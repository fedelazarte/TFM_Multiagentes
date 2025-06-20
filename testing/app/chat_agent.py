import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.tools import tool
from typing import TypedDict
import langdetect

# Cargar entorno
load_dotenv()

# Configuración básica
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_VERSION"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name="tfm_embeddings",
    connection_args={"host": host, "port": port},
    text_field="content",
    auto_id=True
)
retriever = vectorstore.as_retriever()

# Estado del agente
type AgentState = TypedDict("AgentState", {
    "question": str,
    "lang": str,
    "translated": str,
    "docs": str,
    "answer": str,
    "fallback": bool,
})

# Herramientas (simples agentes funcionales)
@tool
def detectar_idioma(question: str) -> str:
    """Detecta el idioma de una pregunta en texto plano."""
    return langdetect.detect(question)

@tool
def traducir_al_espanol(question: str) -> str:
    """Traduce una pregunta al español manteniendo el sentido legal."""
    prompt = f"Traduce al español esta pregunta conservando el sentido legal: {question}"
    respuesta = llm.invoke(prompt)
    return respuesta.content.strip()

@tool
def buscar_en_vectorstore(question: str) -> str:
    """Recupera contexto legal desde Milvus a partir de una pregunta en español."""
    docs = retriever.invoke(question)
    joined = "\n---\n".join([d.page_content for d in docs[:5]])
    return joined

@tool
def redactar_respuesta(contexto: str, pregunta: str) -> str:
    """Redacta una respuesta legal clara usando contexto y una pregunta."""
    prompt = f"Con base en el siguiente contexto legal:\n{contexto}\n\nResponde claramente a la pregunta:\n{pregunta}"
    respuesta = llm.invoke(prompt)
    return respuesta.content.strip()

def clasificador(state: AgentState):
    lang = detectar_idioma.invoke({"question": state["question"]})
    return {**state, "lang": lang, "original_lang": lang, "question": state["question"]}

def traductor(state: AgentState):
    if state["lang"] == "es":
        return {**state, "translated": state["question"]}
    traducida = traducir_al_espanol.invoke({"question": state["question"]})
    return {**state, "translated": traducida}

def buscador(state: AgentState):
    contexto = buscar_en_vectorstore.invoke({"question": state["translated"]})
    return {**state, "docs": contexto}

def verificador(state: AgentState):
    if not state["docs"].strip():
        return {**state, "fallback": True}
    return {**state, "fallback": False}

def explicador(state: AgentState):
    respuesta = redactar_respuesta.invoke({
        "contexto": state["docs"],
        "pregunta": state["translated"]
    })
    return {**state, "answer": respuesta}

def retraductor(state: AgentState):
    if state.get("original_lang") == "es":
        return state
    prompt = f"Traduce esto al idioma original ({state['original_lang']}), manteniendo el tono formal y claro:\n\n{state['answer']}"
    traduccion = llm.invoke(prompt).content.strip()
    return {**state, "answer": traduccion}

def fallback(state: AgentState):
    return {**state, "answer": "Lo siento, no encontré información suficiente para responder a tu pregunta. ¿Podrías reformularla o ser más específico?"}

# Construcción del grafo
graph = StateGraph(AgentState)
graph.add_node("clasificador", clasificador)
graph.add_node("traductor", traductor)
graph.add_node("buscador", buscador)
graph.add_node("verificador", verificador)
graph.add_node("explicador", explicador)
graph.add_node("retraductor", retraductor)
graph.add_node("fallback", fallback)

graph.set_entry_point("clasificador")
graph.add_edge("clasificador", "traductor")
graph.add_edge("traductor", "buscador")
graph.add_edge("buscador", "verificador")
graph.add_conditional_edges(
    "verificador",
    lambda state: "fallback" if state.get("fallback") else "explicador",
    {"fallback": "fallback", "explicador": "explicador"}
)
graph.add_edge("explicador", "retraductor")
graph.add_edge("retraductor", END)
graph.add_edge("fallback", END)

chain = graph.compile()

# Chat loop
print("🤖 Asistente multiagente listo. Escribe 'salir' para terminar.\n")

while True:
    query = input("🧑 Tú: ")
    if query.lower() in ["salir", "exit", "quit"]:
        break
    state = {"question": query}
    result = chain.invoke(state)
    print(f"\n🤖 Asistente: {result['answer']}\n")
