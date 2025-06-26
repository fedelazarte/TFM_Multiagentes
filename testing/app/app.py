import os

# Deshabilita el watcher de archivos de Streamlit
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch

# Monkey-patch para evitar error de Torch
torch.classes.__path__ = []

from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from langchain.tools import tool
from typing import TypedDict

# Cargar variables de entorno
load_dotenv()

# Configuración de Milvus
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

# Configuración de LLM en Azure
llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_VERSION"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
)

# Modelo de embeddings y vectorstore (forzar CPU)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name="tfm_embeddings",
    connection_args={"host": host, "port": port},
    text_field="content",
    auto_id=True,
)
retriever = vectorstore.as_retriever()

# Lista de países de la UE
PAISES_UE = [
    "alemania",
    "austria",
    "bélgica",
    "bulgaria",
    "chipre",
    "croacia",
    "dinamarca",
    "eslovaquia",
    "eslovenia",
    "españa",
    "estonia",
    "finlandia",
    "francia",
    "grecia",
    "hungría",
    "irlanda",
    "italia",
    "letonia",
    "lituania",
    "luxemburgo",
    "malta",
    "países bajos",
    "polonia",
    "portugal",
    "república checa",
    "rumanía",
    "suecia",
]


# Estado interno del agente
class AgentState(TypedDict):
    question: str
    docs: str
    answer: str
    fallback: bool
    intencion: str


# Herramientas: búsqueda y redacción
@tool
def buscar_en_vectorstore(question: str) -> str:
    """Recupera contexto legal desde Milvus a partir de una pregunta en español."""
    docs = retriever.invoke(question)
    joined = "\n---\n".join([d.page_content for d in docs[:5]])
    return joined


@tool
def redactar_respuesta(contexto: str, pregunta: str) -> str:
    """Redacta una respuesta legal clara usando contexto y una pregunta."""
    prompt = (
        "Eres un asistente legal experto en extranjería en España. "
        "No confundas residencia con asilo si el usuario es ciudadano de la Unión Europea.\n\n"
        f"Contexto legal:\n{contexto}\n\n"
        f"Pregunta ciudadana:\n{pregunta}\n\n"
        "Responde de forma clara, respetuosa y legalmente correcta."
    )
    respuesta = llm.invoke(prompt)
    return respuesta.content.strip()


# Definición de nodos y construcción del grafo


def clasificar_intencion(state: AgentState):
    pregunta = state["question"].lower()
    if "residencia" in pregunta or "residenciarme" in pregunta:
        intencion = "residencia"
    elif "asilo" in pregunta:
        intencion = "asilo"
    elif "nacionalidad" in pregunta:
        intencion = "nacionalidad"
    elif "nie" in pregunta or "número de identidad" in pregunta:
        intencion = "nie"
    else:
        intencion = "otro"
    return {**state, "intencion": intencion}


def corregir_contexto(state: AgentState):
    pregunta = state["question"].lower()
    for pais in PAISES_UE:
        if pais in pregunta:
            nueva = (
                state["question"]
                + f"\nNota: El solicitante es ciudadano de {pais.title()}, país miembro de la Unión Europea."
            )
            return {**state, "question": nueva}
    return state


def buscador(state: AgentState):
    contexto = buscar_en_vectorstore.invoke({"question": state["question"]})
    return {**state, "docs": contexto}


def verificador(state: AgentState):
    if not state.get("docs", "").strip():
        return {**state, "fallback": True}
    return {**state, "fallback": False}


def explicador(state: AgentState):
    respuesta = redactar_respuesta.invoke(
        {"contexto": state["docs"], "pregunta": state["question"]}
    )
    return {**state, "answer": respuesta}


def fallback(state: AgentState):
    return {
        **state,
        "answer": (
            "Lo siento, no encontré información suficiente para responder a tu pregunta. "
            "¿Podrías reformularla o ser más específico?"
        ),
    }


# Compilar la cadena de estados
graph = StateGraph(AgentState)
graph.add_node("clasificar_intencion", clasificar_intencion)
graph.add_node("corregir_contexto", corregir_contexto)
graph.add_node("buscador", buscador)
graph.add_node("verificador", verificador)
graph.add_node("explicador", explicador)
graph.add_node("sin_resultados", fallback)

graph.set_entry_point("clasificar_intencion")
graph.add_edge("clasificar_intencion", "corregir_contexto")
graph.add_edge("corregir_contexto", "buscador")
graph.add_edge("buscador", "verificador")
graph.add_conditional_edges(
    "verificador",
    lambda state: "sin_resultados" if state.get("fallback") else "explicador",
    {"sin_resultados": "sin_resultados", "explicador": "explicador"},
)
graph.add_edge("explicador", END)
graph.add_edge("sin_resultados", END)
chain = graph.compile()

# Interfaz en Streamlit con input arriba y animación
st.set_page_config(page_title="Asistente Legal de Extranjería", layout="centered")
st.title("🤖 Asistente Legal de Extranjería")

# Inicializar historial en sesión
if "history" not in st.session_state:
    st.session_state.history = []


# Formulario de entrada
def main():
    with st.form(key="input_form"):
        pregunta = st.text_input("Escribe tu pregunta sobre extranjería:")
        enviar = st.form_submit_button("Enviar")
        if enviar and pregunta:
            # Imprime en consola la pregunta
            print(f"Pregunta recibida: {pregunta}")
            # Animación de procesamiento
            with st.spinner("Procesando tu pregunta..."):
                state = {"question": pregunta}
                resultado = chain.invoke(state)
                respuesta = resultado.get("answer", "Error al procesar la respuesta.")
            # Imprime en consola la respuesta
            print(f"Respuesta generada: {respuesta}")
            # Guardar en historial
            st.session_state.history.append((pregunta, respuesta))

    # Mostrar historial más reciente arriba
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Historial de conversación")
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Tú:** {q}")
            st.markdown(f"**Asistente:** {a}")
            st.markdown("---")


if __name__ == "__main__":
    main()
