from pymilvus import Collection, connections, utility

# Conectar a Milvus
connections.connect(host="localhost", port="19530")

collection_name = "tfm_embeddings"

# Verificar y eliminar si existe
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
    print(f"✅ Colección '{collection_name}' eliminada")
else:
    print(f"ℹ️ La colección '{collection_name}' no existe")

