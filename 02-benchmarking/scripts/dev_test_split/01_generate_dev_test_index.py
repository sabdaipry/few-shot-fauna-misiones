"""Genera un índice nuevo con el query set partido en dev/test estratificado.

Contexto: hoy el query set completo (3674 imágenes) se usa tanto para tomar
todas las decisiones de diseño (backbone, clasificador, métrica de distancia,
umbrales) como para reportar el accuracy final de esas mismas decisiones. Esto
es sesgo de selección (data snooping). Este script separa "query" en:

    query_dev  (70%) — para todas las decisiones de diseño
    query_test (30%) — congelado, solo para el reporte final

El split de "gallery" NO se toca: sigue siendo el catálogo de referencia
completo (888 imágenes), sin cambios.

Regla de especies con soporte insuficiente en query: si una especie tiene
1 sola imagen de query, no se puede garantizar al menos 1 en dev y 1 en test
simultáneamente. Esas especies quedan enteras en query_dev y se excluyen del
cómputo de accuracy en test (deben filtrarse explícitamente aguas abajo).

Para el resto de las especies (n_query >= 2), el split es estratificado con
n_test = clamp(round(n * 0.3), 1, n - 1) — garantiza al menos 1 imagen en
cada lado.

Salida (no sobrescribe dataset_index.csv original):
    02-benchmarking/data/dev_test_split/dataset_index_dev_test_split.csv

Uso:
    python scripts/dev_test_split/01_generate_dev_test_index.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/dev_test_split/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent               # 02-benchmarking/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("dev_test_split", log_dir=_PROJECT_ROOT / "logs")

DATASET_INDEX_PATH = _PROJECT_ROOT / "data" / "dataset_index.csv"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "dev_test_split"
OUTPUT_CSV = OUTPUT_DIR / "dataset_index_dev_test_split.csv"
OUTPUT_EXCLUDED_CSV = OUTPUT_DIR / "excluded_species_query_singletons.csv"

TEST_FRACTION = 0.3
SEED = 29


def split_species_indices(indices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Divide los índices de una especie en (dev_idx, test_idx).

    n_test = clamp(round(n * TEST_FRACTION), 1, n - 1) — garantiza al menos
    una imagen en dev y al menos una en test. Se asume n >= 2 (n == 1 se
    maneja aparte, antes de llamar a esta función).
    """
    n = len(indices)
    shuffled = rng.permutation(indices)
    n_test = int(round(n * TEST_FRACTION))
    n_test = max(1, min(n - 1, n_test))
    test_idx = shuffled[:n_test]
    dev_idx = shuffled[n_test:]
    return dev_idx, test_idx


def main() -> None:
    logger.info("=" * 60)
    logger.info("   GENERACIÓN DE SPLIT DEV/TEST DEL QUERY SET")
    logger.info("=" * 60)

    df = pd.read_csv(DATASET_INDEX_PATH)
    logger.info(f"Índice original cargado: {len(df)} filas")

    new_split = df["split"].copy()
    rng = np.random.default_rng(SEED)

    query_mask = df["split"] == "query"
    query_df = df[query_mask]

    species_counts = query_df.groupby("species").size().sort_values()
    logger.info("Distribución de imágenes de query por especie:")
    logger.info(f"\n{species_counts.value_counts().sort_index().to_string()}")

    excluded_records = []
    n_dev_total = 0
    n_test_total = 0

    for species, group in query_df.groupby("species"):
        idx = group.index.to_numpy()
        n = len(idx)

        if n == 1:
            new_split.loc[idx] = "query_dev"
            excluded_records.append({
                "species": species,
                "query_count": n,
                "reason": "query_count == 1: no se puede garantizar >=1 en dev y >=1 en test",
            })
            n_dev_total += 1
            continue

        dev_idx, test_idx = split_species_indices(idx, rng)
        new_split.loc[dev_idx] = "query_dev"
        new_split.loc[test_idx] = "query_test"
        n_dev_total += len(dev_idx)
        n_test_total += len(test_idx)

    df_out = df.copy()
    df_out["split"] = new_split

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Índice dev/test guardado en: {OUTPUT_CSV}")

    df_excluded = pd.DataFrame(excluded_records)
    df_excluded.to_csv(OUTPUT_EXCLUDED_CSV, index=False)
    logger.info(f"Especies excluidas guardadas en: {OUTPUT_EXCLUDED_CSV}")

    logger.info("")
    logger.info("=== RESUMEN ===")
    logger.info(f"Gallery (sin cambios):            {(df_out['split'] == 'gallery').sum()}")
    logger.info(f"Query_dev:                         {(df_out['split'] == 'query_dev').sum()}")
    logger.info(f"Query_test:                        {(df_out['split'] == 'query_test').sum()}")
    logger.info(f"Especies excluidas de query_test:  {len(df_excluded)}")
    for rec in excluded_records:
        logger.info(f"  - {rec['species']} (query_count={rec['query_count']})")

    # Verificación: ninguna fila de gallery cambió, y todas las filas de
    # query original quedaron etiquetadas como query_dev o query_test.
    assert (df_out.loc[df["split"] == "gallery", "split"] == "gallery").all()
    assert df_out.loc[query_mask, "split"].isin(["query_dev", "query_test"]).all()
    assert len(df_out) == len(df)
    logger.info("Verificación de integridad OK: gallery intacto, todas las filas de query reasignadas.")


if __name__ == "__main__":
    main()
