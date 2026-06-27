"""
Enriquece dataset_index.csv con nombres comunes en español (es-AR) e inglés (en)
consultando la API pública de iNaturalist por nombre científico.

Columnas nuevas, insertadas inmediatamente después de 'species':
  - nombre_comun_es_ar   <- preferred_common_name con locale=es
  - nombre_comun_en      <- preferred_common_name con locale=en

Estrategia: dos requests por especie (locale=es y locale=en),
1.5 s entre cada request individual.

Idempotente: si el backup ya existe se usa como fuente de verdad
(evita sobreescribir el CSV original al re-ejecutar).
"""

import csv
import shutil
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "dataset_index.csv"
BACKUP_PATH = DATA_DIR / "dataset_index_backup.csv"

INAT_API = "https://api.inaturalist.org/v1/taxa"
RATE_LIMIT_SLEEP = 1.5


def query_inat(scientific_name: str, locale: str) -> dict:
    params = urllib.parse.urlencode({
        "q": scientific_name,
        "rank": "species",
        "locale": locale,
    })
    url = f"{INAT_API}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_common_names(scientific_name: str) -> tuple[str, str]:
    """
    Devuelve (nombre_es_ar, nombre_en) para la especie dada.
    Hace dos requests: uno con locale=es, otro con locale=en.
    """
    nombre_es = ""
    nombre_en = ""

    try:
        data_es = query_inat(scientific_name, "es")
        results_es = data_es.get("results", [])
        if results_es:
            nombre_es = results_es[0].get("preferred_common_name", "")
    except Exception as exc:
        print(f"    ERROR (es) '{scientific_name}': {exc}")

    time.sleep(RATE_LIMIT_SLEEP)

    try:
        data_en = query_inat(scientific_name, "en")
        results_en = data_en.get("results", [])
        if results_en:
            nombre_en = results_en[0].get("preferred_common_name", "")
    except Exception as exc:
        print(f"    ERROR (en) '{scientific_name}': {exc}")

    return nombre_es or "", nombre_en or ""


def main():
    # 1. Backup: si ya existe, usarlo como fuente (idempotencia)
    if BACKUP_PATH.exists():
        source_path = BACKUP_PATH
        print(f"Backup existente encontrado. Leyendo desde: {BACKUP_PATH.name}")
    else:
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        source_path = CSV_PATH
        print(f"Backup creado: {BACKUP_PATH.name}")

    # 2. Leer CSV fuente (siempre el original sin columnas nuevas)
    with open(source_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_orig = list(reader.fieldnames)
        rows = list(reader)

    # 3. Extraer especies únicas (orden de primera aparición)
    seen: dict[str, tuple[str, str] | None] = {}
    for row in rows:
        sp = row["species"]
        if sp not in seen:
            seen[sp] = None
    unique_species = list(seen.keys())
    total = len(unique_species)
    print(f"\n{total} taxones unicos encontrados. Iniciando consultas a iNaturalist...\n")

    # 4. Consultar API (2 requests por especie)
    sin_es_ar = []
    for idx, species in enumerate(unique_species, start=1):
        nombre_es, nombre_en = get_common_names(species)
        seen[species] = (nombre_es, nombre_en)

        es_tag = nombre_es if nombre_es else "(vacio)"
        en_tag = nombre_en if nombre_en else "(vacio)"
        print(f"[{idx:>3}/{total}] {species:<40} es-AR: {es_tag:<35} | en: {en_tag}")

        if not nombre_es:
            sin_es_ar.append(species)

        # Pausa entre especies (el sleep interno ya cubre entre los dos requests)
        if idx < total:
            time.sleep(RATE_LIMIT_SLEEP)

    # 5. Reconstruir CSV con columnas nuevas insertadas después de 'species'
    # Excluir las columnas si ya estaban (re-ejecución)
    clean_fields = [f for f in fieldnames_orig
                    if f not in ("nombre_comun_es_ar", "nombre_comun_en")]
    species_idx = clean_fields.index("species")
    new_fields = (
        clean_fields[: species_idx + 1]
        + ["nombre_comun_es_ar", "nombre_comun_en"]
        + clean_fields[species_idx + 1 :]
    )

    for row in rows:
        sp = row["species"]
        nombre_es, nombre_en = seen[sp]
        row["nombre_comun_es_ar"] = nombre_es
        row["nombre_comun_en"] = nombre_en

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV actualizado: {CSV_PATH.name}")

    # 6. Resumen
    con_es = total - len(sin_es_ar)
    con_en = sum(1 for v in seen.values() if v and v[1])
    print("\n" + "=" * 60)
    print("Resumen")
    print("=" * 60)
    print(f"  Taxones unicos consultados : {total}")
    print(f"  Con nombre en es-AR        : {con_es}")
    print(f"  Con nombre en ingles       : {con_en}")
    print(f"  Sin nombre en es-AR        : {len(sin_es_ar)}")
    if sin_es_ar:
        print("\n  Taxones sin es-AR (para revision manual):")
        for sp in sin_es_ar:
            print(f"    - {sp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
