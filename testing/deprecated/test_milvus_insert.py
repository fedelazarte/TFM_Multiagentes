from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# Conectar a Milvus
connections.connect(host=os.getenv("MILVUS_HOST", "localhost"), port=os.getenv("MILVUS_PORT", "19530"))

# Embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Nombre de colección
collection_name = "tfm_embeddings"

# Borrar si ya existe
if utility.has_collection(collection_name):
    print(f"⚠️ Eliminando colección existente: {collection_name}")
    Collection(name=collection_name).drop()

# Definir nuevo esquema con campo llamado 'vector'
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),  # <- Cambiado aquí
]
schema = CollectionSchema(fields=fields, description="Embeddings de prueba")

# Crear colección
collection = Collection(name=collection_name, schema=schema)

# Textos y embeddings
textos = [
    "Soy estudiante mexicano con NIE caducado",
    "Quiero cambiar de universidad",
    "Necesito renovar mi permiso de residencia",
]
embeddings = model.encode(textos).tolist()
data = [textos, embeddings]

# Insertar y flush
collection.insert(data)
collection.flush()

# Indexar para búsquedas
collection.create_index(
    field_name="vector",  # <- Importante: coincide con el nombre del campo
    index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
)
collection.load()

print(f"✅ Colección '{collection_name}' creada e indexada con {collection.num_entities} vectores.")
