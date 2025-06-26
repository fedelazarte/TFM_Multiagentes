from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

# Conectar a Milvus
connections.connect(host="localhost", port="19530")

collection_name = "tfm_embeddings_t1"

# Borrar si ya existe
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
    print(f"🗑️ Colección '{collection_name}' eliminada")

# Definir esquema con auto_id=True
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
]

schema = CollectionSchema(
    fields=fields, description="Embeddings de textos de extranjería"
)

# Crear colección
collection = Collection(name=collection_name, schema=schema)
print(f"✅ Colección '{collection_name}' creada")

# Crear índice para búsquedas
index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()

print("🔎 Índice creado y colección cargada en memoria")
