# 📘 Proyecto TFM - Asistente Multiagente para Trámites de Extranjería

Este proyecto implementa un agente conversacional multiagente con recuperación semántica para asistir en trámites de extranjería en España.

Incluye:
- Extracción de contenido desde la web de extranjería
- Procesamiento de HTMLs y PDFs
- Embeddings con HuggingFace
- Almacenamiento vectorial en Milvus
- Asistente multiagente orquestado con LangGraph

---

## 🧱 Requisitos del entorno

### 1. Instalar Python 3.10+ (recomendado 3.10 o 3.11)

#### En macOS con Homebrew
```bash
brew install python@3.11
```

#### En Ubuntu
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

---

### 2. Instalar Docker y Docker Compose

#### macOS / Windows:
Descargar desde: https://www.docker.com/products/docker-desktop/

#### Ubuntu:
```bash
sudo apt install docker.io docker-compose
sudo systemctl enable docker --now
sudo usermod -aG docker $USER
```
Luego cerrá sesión y volvé a entrar para aplicar permisos.

---

### 3. Levantar Milvus localmente

```bash
git clone https://github.com/milvus-io/milvus.git
cd milvus/deployments/docker/standalone
curl -L -o docker-compose.yml https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml

# Levantar Milvus
docker-compose up -d
```

> Se recomienda tener al menos 4 GB libres de RAM.

---

### 4. Crear entorno virtual y activar
```bash
python3 -m venv tfm_env
source tfm_env/bin/activate
```

### 5. Instalar dependencias del proyecto
```bash
pip install -r requirements.txt
```

> Si usás Mac, asegurate de ejecutar:
> ```bash
> /Applications/Python\ 3.x/Install\ Certificates.command
> ```

---

## 📦 Dependencias (`requirements.txt` sugerido)

```text
langchain==0.1.14
langgraph==0.0.39
langchain-community>=0.0.20
langchain-core>=0.1.46
langchain-openai>=0.1.5
langchain-huggingface>=0.0.6
langchain-milvus>=0.0.4
openai>=1.17.0
huggingface-hub
sentence-transformers
pymilvus==2.3.4
pdfplumber
beautifulsoup4
requests
python-dotenv
urllib3
langdetect
```

Para generarlo desde tu entorno actual:
```bash
pip freeze > requirements.txt
```

---

## 🗂 Estructura del proyecto

```bash
R31_Multiagentes/
├── create_collection.py
├── scrape_extranjeria.py
├── valida_enlaces.py
├── ingest_html_to_milvus.py
├── ingest_pdfs_to_milvus.py
├── chat_agent.py
├── pdfs/                 # (auto-generada)
├── enlaces_validos.json  # (generado por valida_enlaces.py)
├── requirements.txt
└── .env                  # contiene claves de Azure OpenAI y Milvus
```

---

## 🚀 Orden de ejecución

### 1. Inicializar la colección en Milvus
```bash
python create_collection.py
```

### 2. Scraping de enlaces HTML
```bash
python scrape_extranjeria.py
```

### 3. Validar enlaces
```bash
python valida_enlaces.py
```

### 4. Ingesta de contenidos HTML
```bash
python ingest_html_to_milvus.py
```

### 5. Ingesta de PDFs detectados en HTML
```bash
python ingest_pdfs_to_milvus.py
```

### 6. Ejecutar el asistente multiagente
```bash
python chat_agent.py
```

---

## 🔐 Variables necesarias en `.env`

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_DEPLOYMENT=gpt-4-TFM
OPENAI_MODEL=gpt-4
OPENAI_VERSION=2025-01-01-preview
OPENAI_ENDPOINT=https://...openai.azure.com/
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## ✅ Notas adicionales
- Los documentos insertados incluyen `metadata["source"]` para trazabilidad.
- El agente admite preguntas en otros idiomas y responde en el mismo idioma detectado.
- El proyecto es compatible con futuras extensiones usando LangGra
