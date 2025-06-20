import requests
from bs4 import BeautifulSoup
import urllib3
import json

# ⚠️ Desactiva warnings por deshabilitar verificación SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://extranjeros.inclusion.gob.es"

def get_all_links():
    resp = requests.get(BASE_URL, verify=False)
    soup = BeautifulSoup(resp.text, "lxml")

    enlaces = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        texto = a.get_text(strip=True)
        if href.startswith("http") or href.startswith("/"):
            enlaces.append({
                "texto": texto,
                "url": href if href.startswith("http") else BASE_URL + href
            })

    return enlaces

if __name__ == "__main__":
    links = get_all_links()

    # Mostrar algunos por consola
    for l in links[:10]:
        print(f"{l['texto']}: {l['url']}")

    # Guardar a JSON local
    with open("enlaces_extranj.json", "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Guardados {len(links)} enlaces en enlaces_extranj.json")
