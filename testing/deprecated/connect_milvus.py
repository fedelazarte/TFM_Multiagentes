from pymilvus import connections, list_collections
import os
from dotenv import load_dotenv

load_dotenv()

# Parámetros de conexión
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

# Conexión
connections.connect(host=host, port=port)
print(f"✅ Conectado a Milvus en {host}:{port}")

# Listar colecciones
collections = list_collections()
print("Colecciones disponibles:", collections)
