import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, verify=False, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")

        # Elimina scripts, estilos y navegación irrelevante
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
    url = "https://extranjeros.inclusion.gob.es/es/Normativa/index.html"
    texto = extract_text_from_url(url)
    print("\n".join(texto.splitlines()[:20]))  # Mostrar primeros 20 renglones
