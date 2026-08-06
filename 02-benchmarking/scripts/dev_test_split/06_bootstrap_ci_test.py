"""Bootstrap CI (95%) sobre las predicciones congeladas de query_test
(scripts/dev_test_split/05_final_eval_test.py), para BioCLIP v2 +
{Nearest Centroid, Linear SVM}.

Reutiliza sin modificarlas las mismas funciones puras de 07_bootstrap_ci.py
que ya se usaron para query completo y query_dev (build_class_index_map,
stratified_bootstrap_resample, compute_bootstrap_ci), misma semilla (29) y
mismas 1000 iteraciones, para que los tres IC sean directamente comparables.

Salida (carpeta nueva, no pisa nada):
    data/benchmark_results_test/bootstrap_ci_test.csv

Uso:
    python scripts/dev_test_split/06_bootstrap_ci_test.py [--iterations 1000] [--seed 29]
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

logger = setup_logger("bootstrap_ci_test", log_dir=_PROJECT_ROOT / "logs")

_spec = importlib.util.spec_from_file_location(
    "bootstrap_ci_base", _SCRIPTS_DIR / "07_bootstrap_ci.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

RESULTS_DIR_TEST = _PROJECT_ROOT / "data" / "benchmark_results_test"
PRED_FILE = RESULTS_DIR_TEST / "predictions_bioclip_v2_test.csv"


def main(args) -> None:
    logger.info("==============================================")
    logger.info("   BOOTSTRAP CI (95%) SOBRE query_test (CONGELADO)")
    logger.info("==============================================")

    if not PRED_FILE.exists():
        logger.error(
            f"No se encontró {PRED_FILE}. "
            "Correr antes scripts/dev_test_split/05_final_eval_test.py."
        )
        return

    df_pred = pd.read_csv(PRED_FILE)
    y_true = df_pred["y_true"].to_numpy()
    labels = np.unique(y_true)
    n_total = len(y_true)
    logger.info(f"query_test: {PRED_FILE.name} ({n_total} filas, {len(labels)} clases)")

    rng = np.random.default_rng(args.seed)
    class_index_map = _base.build_class_index_map(y_true)
    logger.info(f"Generando {args.iterations} resamples bootstrap estratificados (seed={args.seed})...")
    bootstrap_idx = [
        _base.stratified_bootstrap_resample(class_index_map, n_total, rng)
        for _ in range(args.iterations)
    ]

    pred_cols = [c for c in df_pred.columns if c.startswith("pred_") and "Faiss" not in c]
    combos = [("bioclip_v2", col.replace("pred_", ""), df_pred[col].to_numpy()) for col in pred_cols]
    logger.info(f"{len(combos)} combinaciones a evaluar: {[c[1] for c in combos]}")

    records = []
    for model_name, clf_name, y_pred in tqdm(combos, desc="Bootstrap CI (test)"):
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
            logger.info(
                f"{clf_name:<18} {metric_name:<10} mean={mean:.4f}  "
                f"CI95=[{ci_lo:.4f}, {ci_hi:.4f}]  width={ci_width:.4f}"
            )

    df_out = pd.DataFrame(records)
    out_path = RESULTS_DIR_TEST / "bootstrap_ci_test.csv"
    df_out.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path} ({len(df_out)} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CI (95%) sobre predictions_bioclip_v2_test.csv.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--confidence", type=float, default=0.95)
    main(parser.parse_args())
