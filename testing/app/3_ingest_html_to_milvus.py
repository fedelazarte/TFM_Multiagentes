import json
import requests
from bs4 import BeautifulSoup
import urllib3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from dotenv import load_dotenv
import os
from pathlib import Path

# Desactivar warnings SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# Ruta segura al JSON con enlaces válidos
CURRENT_DIR = Path(__file__).resolve().parent
ENLACES_PATH = CURRENT_DIR.parent / "enlaces_validos.json"

# Conexión a Milvus
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name="tfm_embeddings",
    connection_args={"host": host, "port": port},
    text_field="content",
    auto_id=True  # ✅ especificar explícitamente que los IDs son automáticos
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def extract_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, verify=False, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_lines = [line for line in lines if line]
        return "\n".join(clean_lines)
    except Exception as e:
        print(f"❌ Error al procesar {url}: {e}")
        return ""

if __name__ == "__main__":
    if not ENLACES_PATH.exists():
        print(f"❌ No se encontró el archivo: {ENLACES_PATH}")
        exit(1)

    with open(ENLACES_PATH, "r", encoding="utf-8") as f:
        enlaces = json.load(f)

    print(f"🔢 Procesando {len(enlaces)} enlaces...\n")

    for item in enlaces:
        url = item["url"]
        print(f"🌐 {url}")
        raw_text = extract_text_from_url(url)

        if not raw_text.strip():
            print(f"⚠️ Contenido vacío o ilegible: {url}")
            continue

        docs = splitter.create_documents([raw_text], metadatas=[{"source": url}])
        vectorstore.add_documents(docs)
        print(f"✅ {len(docs)} fragmentos insertados.\n")
