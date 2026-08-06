"""Re-corre el IC bootstrap (scripts/07_bootstrap_ci.py) sobre las predicciones
de query_dev generadas por scripts/dev_test_split/02_run_benchmark_dev.py.

Reutiliza sin modificarlas las funciones puras de 07_bootstrap_ci.py:
build_class_index_map, stratified_bootstrap_resample, compute_bootstrap_ci.
Mismas limitaciones metodológicas documentadas en el docstring de ese script
aplican acá sin cambios (varianza de muestreo del query_dev set, comparaciones
apareadas no explotadas, clases singleton, clasificadores correlacionados).

Salida (carpeta nueva, no pisa data/benchmark_results/bootstrap_ci.csv):
    data/benchmark_results_dev/bootstrap_ci_dev.csv

Uso:
    python scripts/dev_test_split/04_bootstrap_ci_dev.py [--iterations 1000] [--seed 29]
"""
import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/dev_test_split/
_SCRIPTS_DIR = _SCRIPT_DIR.parent                       # scripts/
_PROJECT_ROOT = _SCRIPTS_DIR.parent                     # 02-benchmarking/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("bootstrap_ci_dev", log_dir=_PROJECT_ROOT / "logs")

_spec = importlib.util.spec_from_file_location(
    "bootstrap_ci_base", _SCRIPTS_DIR / "07_bootstrap_ci.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

BENCHMARK_RESULTS_DIR_DEV = _PROJECT_ROOT / "data" / "benchmark_results_dev"


def main(args) -> None:
    logger.info("==============================================")
    logger.info("   BOOTSTRAP CI (95%) SOBRE PREDICCIONES query_dev")
    logger.info("==============================================")

    pred_files = sorted(BENCHMARK_RESULTS_DIR_DEV.glob("predictions_*_dev.csv"))
    if not pred_files:
        logger.error(
            f"No se encontraron predictions_*_dev.csv en {BENCHMARK_RESULTS_DIR_DEV}. "
            "Correr antes scripts/dev_test_split/02_run_benchmark_dev.py."
        )
        return

    df_ref = pd.read_csv(pred_files[0])
    y_true = df_ref["y_true"].to_numpy()
    labels = np.unique(y_true)
    n_total = len(y_true)
    logger.info(f"Query_dev set de referencia: {pred_files[0].name} ({n_total} filas, {len(labels)} clases)")

    rng = np.random.default_rng(args.seed)
    class_index_map = _base.build_class_index_map(y_true)
    logger.info(f"Generando {args.iterations} resamples bootstrap estratificados (seed={args.seed})...")
    bootstrap_idx = [
        _base.stratified_bootstrap_resample(class_index_map, n_total, rng)
        for _ in range(args.iterations)
    ]

    combos = []
    for pred_path in pred_files:
        model_name = pred_path.stem.replace("predictions_", "").replace("_dev", "")
        df_pred = pd.read_csv(pred_path)

        if not np.array_equal(df_pred["y_true"].to_numpy(), y_true):
            logger.warning(f"{pred_path.name}: y_true difiere del query_dev de referencia, se omite.")
            continue

        # Sin Faiss (ya excluidas del benchmark dev, este filtro es solo defensivo)
        pred_cols = [c for c in df_pred.columns if c.startswith("pred_") and "Faiss" not in c]
        for col in pred_cols:
            clf_name = col.replace("pred_", "")
            combos.append((model_name, clf_name, df_pred[col].to_numpy()))

    logger.info(f"{len(combos)} combinaciones Backbone x Clasificador a evaluar.")

    records = []
    for model_name, clf_name, y_pred in tqdm(combos, desc="Bootstrap CI (dev)"):
        ci_results = _base.compute_bootstrap_ci(y_true, y_pred, labels, bootstrap_idx, args.confidence)
        for metric_name, (mean, ci_lo, ci_hi, ci_width) in ci_results.items():
            records.append({
                "Embedding Model": model_name,
                "Classifier": clf_name,
                "Metric": metric_name,
                "Mean": mean,
                "CI_lower": ci_lo,
                "CI_upper": ci_hi,
                "CI_width": ci_width,
            })

    df_out = pd.DataFrame(records)
    out_path = BENCHMARK_RESULTS_DIR_DEV / "bootstrap_ci_dev.csv"
    df_out.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path} ({len(df_out)} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CI (95%) sobre predictions_*_dev.csv.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--confidence", type=float, default=0.95)
    main(parser.parse_args())
