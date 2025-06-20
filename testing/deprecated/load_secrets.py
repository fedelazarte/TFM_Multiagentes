from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Cargar variables de entorno
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_VERSION"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
)

# Test
response = client.chat.completions.create(
    model=os.getenv("OPENAI_DEPLOYMENT"),
    messages=[{"role": "user", "content": "Hola, ¿qué puedes hacer por mí?"}],
)

print(response.choices[0].message.content)