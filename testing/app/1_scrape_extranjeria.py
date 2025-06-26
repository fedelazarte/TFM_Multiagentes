import requests
from bs4 import BeautifulSoup
import urllib3
import json

# ⚠️ Desactiva warnings por deshabilitar verificación SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_all_links(base_url):
    resp = requests.get(base_url, verify=False)
    soup = BeautifulSoup(resp.text, "lxml")

    enlaces = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        texto = a.get_text(strip=True)
        if href.startswith("http") or href.startswith("/"):
            enlaces.append(
                {
                    "texto": texto,
                    "url": (
                        href if href.startswith("http") else base_url.rstrip("/") + href
                    ),
                }
            )

    return enlaces


if __name__ == "__main__":
    # 1) Carga de URLs desde JSON
    with open("enlaces_validos.json", "r", encoding="utf-8") as f:
        entries = json.load(f)

    # 2) Scrapeo de cada URL extraída del JSON
    all_links = []
    for entry in entries:
        url = entry.get("url")
        print(f"🔍 Scrapeando {url} …")
        try:
            enlaces = get_all_links(url)
            all_links.extend(enlaces)
        except Exception as e:
            print(f"❌ Error en {url}: {e}")

    # 3) Mostrar algunos enlaces por consola
    for enlace in all_links[:10]:
        print(f"{enlace['texto']}: {enlace['url']}")

    # 4) Guardar resultados a JSON local
    with open("enlaces_extranj.json", "w", encoding="utf-8") as f:
        json.dump(all_links, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Guardados {len(all_links)} enlaces en enlaces_extranj.json")
