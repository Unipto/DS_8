import pdfplumber

def extract_ipcc_text(pdf_path):
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)
    return "\n".join(text_pages)


from DS_8.config.utils import PATH_IPCC_PDF_FOLDER
import glob

ipcc_block = ""
for f in glob.glob(str(PATH_IPCC_PDF_FOLDER) + "*.pdf"):
    ipcc_text = extract_ipcc_text(f)
    print(type(f))
    print(f.split("/")[-1][:-4])
    ipcc_block += f"""
    [SOURCE: {f}]
    {ipcc_text}

    """

import requests
from bs4 import BeautifulSoup

def extract_nasa_myth(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article")
    if not article:
        return ""

    paragraphs = article.find_all(["p", "h2", "h3"])
    text = "\n".join(p.get_text(strip=True) for p in paragraphs)
    return text


from DS_8.config.utils import NASA_MYTH_URLS


nasa_block = ""
for url in NASA_MYTH_URLS:
    
    nasa_text = (extract_nasa_myth(url))

    nasa_block += f"""
    [SOURCE: NASA_CLIMATE_{url.split("/")[-2].upper().replace("-", "_")}]
    {nasa_text}

    """

from DS_8.config.utils import PATH_KNOWLEDGE_BASE

with open(PATH_KNOWLEDGE_BASE, "w", encoding="utf-8") as f:
    f.write(ipcc_block)
    f.write("\n\n")
    f.write(nasa_block)