import os
import json
import requests
import pdfplumber
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
import urllib3

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
ENLACES_PATH = CURRENT_DIR.parent / "enlaces_validos.json"
PDF_DIR = CURRENT_DIR.parent / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

# LangChain + Milvus
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name="tfm_embeddings_t1",
    connection_args={"host": host, "port": port},
    text_field="content",
    auto_id=True,
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def is_pdf_url(url):
    try:
        resp = requests.head(url, allow_redirects=True, verify=False, timeout=10)
        return "application/pdf" in resp.headers.get("Content-Type", "")
    except:
        return False


def find_pdfs_in_html(url):
    try:
        resp = requests.get(url, verify=False, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        return [
            (
                link
                if link.startswith("http")
                else "https://extranjeros.inclusion.gob.es" + link
            )
            for link in [a["href"] for a in soup.find_all("a", href=True)]
            if ".pdf" in link.lower()
        ]
    except:
        return []


def download_pdf(url, path):
    try:
        r = requests.get(url, verify=False, timeout=15)
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except:
        return False


def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except:
        return ""


def process_and_store_pdf(url):
    filename = url.split("/")[-1] or f"archivo_{abs(hash(url))}.pdf"
    filepath = PDF_DIR / filename

    if not filepath.exists():
        print(f"⬇️ Descargando: {filename}")
        if not download_pdf(url, filepath):
            print(f"❌ Fallo la descarga: {url}")
            return
    else:
        print(f"📂 Ya descargado: {filename}")

    text = extract_text_from_pdf(filepath)
    if not text.strip():
        print(f"⚠️ Texto vacío en {filename}")
        return

    docs = splitter.create_documents([text], metadatas=[{"source": url}])
    vectorstore.add_documents(docs)
    print(f"✅ {len(docs)} fragmentos insertados desde {filename}\n")


if __name__ == "__main__":
    if not ENLACES_PATH.exists():
        print(f"❌ No se encontró el archivo: {ENLACES_PATH}")
        exit(1)

    with open(ENLACES_PATH, "r", encoding="utf-8") as f:
        enlaces = json.load(f)

    print(f"🔍 Verificando {len(enlaces)} enlaces...\n")
    pdfs_directos = [e["url"] for e in enlaces if is_pdf_url(e["url"])]

    if pdfs_directos:
        print(f"📄 Detectados {len(pdfs_directos)} PDF(s) reales por Content-Type\n")
        for url in pdfs_directos:
            process_and_store_pdf(url)
    else:
        print("ℹ️ No se detectaron PDFs por Content-Type, explorando HTMLs...\n")
        pdfs_indirectos = []
        for e in enlaces:
            nuevos = find_pdfs_in_html(e["url"])
            if nuevos:
                print(f"🔗 {e['url']} → {len(nuevos)} PDF(s) encontrados en href")
                pdfs_indirectos.extend(nuevos)

        pdfs_indirectos = list(set(pdfs_indirectos))  # eliminar duplicados

        if pdfs_indirectos:
            print(
                f"\n📄 Procesando {len(pdfs_indirectos)} PDF(s) encontrados en HTML\n"
            )
            for url in pdfs_indirectos:
                process_and_store_pdf(url)
        else:
            print("❌ No se detectaron PDFs ni por Content-Type ni por HTML")
