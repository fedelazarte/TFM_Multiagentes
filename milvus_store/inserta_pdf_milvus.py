import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

# Configuración
pdf_path = 'C:/Users/Joaquin/Desktop/Proyectos/R31_Multiagentes/milvus_store/pdf_documents/guia_tramites_extranjeria.pdf'
milvus_host = 'localhost'
milvus_port = '19530'
collection_name = 'pdf_tramites_extranjeria'

# Inicializar Milvus
embedding_function = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model="text-embedding-ada-002",
    chunk_size=1000
)

vector_store = Milvus(
    embedding_function=embedding_function,
    collection_name=collection_name,
    connection_args={"host": milvus_host, "port": milvus_port},
    auto_id=True,  # Permite que Milvus asigne automáticamente los IDs
)

# Procesar el PDF
print(f"Procesando: {os.path.basename(pdf_path)}")
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Patrón para detectar encabezados de capítulo
chapter_pattern = re.compile(r'(Capítulo\s+\d+\.?\s+[^\n]*)', re.IGNORECASE)

# Añadir metadatos
for doc in docs:
    doc.metadata['filename'] = os.path.basename(pdf_path)
    doc.metadata['page_number'] = doc.metadata.get('page', None)

    # Buscar encabezado de capítulo en el contenido
    match = chapter_pattern.search(doc.page_content)
    if match:
        doc.metadata['chapter'] = match.group(1).strip()
    else:
        doc.metadata['chapter'] = 'Sin capítulo detectado'

# Insertar en Milvus
print(f"Inserting {len(docs)} documents into Milvus...")
vector_store.add_documents(docs)

print("Inserción completada.")
