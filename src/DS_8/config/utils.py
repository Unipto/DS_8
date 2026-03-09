###############
#    PATHS    #
###############
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]




##################
# KNOWLEDGE BASE #
##################
PATH_KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base.txt"
PATH_IPCC_PDF_FOLDER = PROJECT_ROOT / "data" / "IPCC_PDF"

NASA_MYTH_URLS = [
    "https://science.nasa.gov/climate-change/evidence/",
    "https://science.nasa.gov/climate-change/scientific-consensus/",
    "https://science.nasa.gov/climate-change/causes/",
    "https://science.nasa.gov/climate-change/effects/",
    "https://science.nasa.gov/climate-change/what-is-climate-change/",
    "https://science.nasa.gov/climate-change/extreme-weather/",
    "https://science.nasa.gov/climate-change/faq/",
    "https://science.nasa.gov/earth/explore/wildfires-and-climate-change/",
    "https://science.nasa.gov/earth/earth-observatory/fire-on-ice-the-arctics-changing-fire-regime/",

]


####################
#   DATA LOADING   #
####################
PATH_CLIMATE_FEVER_CSV = PROJECT_ROOT / "data" / "climate_FEVER" / "climate-fever.csv"
import pandas as pd

def load_train_eval_climate_fever_data():
    df = pd.read_csv(PATH_CLIMATE_FEVER_CSV)

    # On garde uniquement supported / refuted
    df = df[df["claim_label"].isin(["SUPPORTS", "REFUTES"])]

    # Sous-échantillon pour des raisons de coût / limitations API
    df_eval = df.sample(n=150, random_state=42).reset_index(drop=True)
    df_train = df.loc[~df["claim_id"].isin(df_eval["claim_id"])]

    return df_train, df_eval


####################
#   RAG PIPELINE   #
####################

BERT_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


def retrieve_top_k_chunks(claim, index, embedding_model, chunks, k=3):
    claim_embedding = embedding_model.encode(
        [claim],
        convert_to_numpy=True
    )

    distances, indices = index.search(claim_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks


def build_fact_check_prompt(claim, retrieved_chunks):
    context = "\n\n".join(
        [f"Source {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are an expert climate fact-checker.

TASK:
Given a claim and a set of reliable scientific sources, determine whether the claim is:
- TRUE
- FALSE
- NOT ENOUGH EVIDENCE

RULES:
- Base your decision ONLY on the provided sources.
- If the sources do not explicitly support or refute the claim, answer NOT ENOUGH EVIDENCE.
- Provide a short justification citing the sources.

CLAIM:
"{claim}"

SOURCES:
{context}

OUTPUT FORMAT (strict):
Verdict: <TRUE / FALSE / NOT ENOUGH EVIDENCE>
Justification: <2–4 sentences>
"""
    return prompt

def fact_check_with_groq(claim, llm_model, index, embedding_model, chunks, client, k=3):
    retrieved_chunks = retrieve_top_k_chunks(claim, index, embedding_model, chunks, k)
    prompt = build_fact_check_prompt(claim, retrieved_chunks)

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a careful and precise scientific assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )
    output = response.choices[0].message.content
    print(output[:300])
    return output, retrieved_chunks

def parse_llm_output(output):
    import re
    output = output.lower()
    match = re.search(pattern=r"verdict:[^\w]*((?:true)|(?:false)|(?:not enough))", string=output)
    print(match)
    if match and match.group(1):
        if "true" in output:
            return "SUPPORTS"
        elif "false" in output:
            return "REFUTES"
        elif "not enough" in output:
            return "NOT_ENOUGH_EVIDENCE"

    return "UNKNOWN"

##############
#  PLOTTING  #
##############
def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve
    
	# Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true == "SUPPORTS", y_pred_proba)

    thresholds = np.asarray(thresholds)
    closest_idx = np.abs(thresholds - PROD_MODEL_THRESHOLD).argmin()
    
    # Figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[fpr[closest_idx], fpr[closest_idx]], y=[0, tpr[closest_idx]], mode="lines", line=dict(dash="dot", color="darkred"), name='Model theoretical FPR and FTR', showlegend=True))

    fig.add_trace(go.Scatter(x=[0, fpr[closest_idx]], y=[tpr[closest_idx], tpr[closest_idx]], mode="lines", line=dict(dash="dot", color="darkred"), name='Model theoretical TPR', showlegend=False))

    # ROC curve (area)
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            fill="tozeroy",
            name="ROC Curve"
        )
    )

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random Classifier"
        )
    )

    # Layout and axes
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    fig.show()