import json
import requests
import urllib3

from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INPUT_FILE = "enlaces_extranj.json"
OUTPUT_FILE = "enlaces_validos.json"

def cargar_enlaces(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validar_enlace(url):
    try:
        resp = requests.head(url, allow_redirects=True, verify=False, timeout=5)
        return resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", "")
    except Exception:
        return False

def eliminar_duplicados(enlaces):
    vistos = set()
    unicos = []
    for item in enlaces:
        url = item["url"]
        if url not in vistos:
            vistos.add(url)
            unicos.append(item)
    return unicos

if __name__ == "__main__":
    print("🔍 Cargando y validando enlaces...")

    raw_enlaces = cargar_enlaces(INPUT_FILE)
    print(f"🔢 Enlaces originales: {len(raw_enlaces)}")

    enlaces_unicos = eliminar_duplicados(raw_enlaces)
    print(f"✅ Sin duplicados: {len(enlaces_unicos)}")

    enlaces_validos = []
    for item in tqdm(enlaces_unicos, desc="Validando enlaces"):
        if validar_enlace(item["url"]):
            enlaces_validos.append(item)

    print(f"\n✅ Enlaces válidos (200 OK y HTML): {len(enlaces_validos)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enlaces_validos, f, indent=2, ensure_ascii=False)

    print(f"📁 Guardados en: {OUTPUT_FILE}")
