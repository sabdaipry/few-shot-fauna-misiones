"""Evaluación final única y congelada sobre query_test.

Corre UNA sola vez, después de que arquitectura (BioCLIP v2), métrica
(coseno), clasificadores a reportar (Nearest Centroid — pipeline de
producción real de 04-app, y Linear SVM — ganador nominal por accuracy en
query_dev) y umbral (p95 centroide recalibrado en query_dev, 0.18693) ya
fueron decididos exclusivamente a partir de query_dev. Este script no debe
volver a correrse con otros parámetros derivados de mirar el resultado.

Genera:
1. Accuracy Top-1, Top-5, F1-Macro para BioCLIP v2 + {Nearest Centroid,
   Linear SVM} sobre query_test (gallery de entrenamiento sin cambios).
2. Desglose de errores taxonómicos (leve/medio/severo/crítico) para ambos,
   vía la misma función ya existente src/analysis.py:analyze_taxonomic_errors.
3. Validación del umbral de confianza (p95 centroide = 0.18693, calibrado en
   query_dev) aplicado a query_test congelado: coverage y contaminación
   inter-clase observadas en el test set nunca antes tocado.

Salidas (carpeta nueva, no pisa nada de data/benchmark_results/ ni de
data/benchmark_results_dev/):
    data/benchmark_results_test/benchmark_summary_test.csv
    data/benchmark_results_test/predictions_bioclip_v2_test.csv
    data/benchmark_results_test/taxonomic_error_breakdown_test.csv
    data/benchmark_results_test/threshold_validation_test.csv

Uso:
    python scripts/dev_test_split/05_final_eval_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import normalize

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/dev_test_split/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent               # 02-benchmarking/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmarking import ModelEvaluator
from src.analysis import analyze_taxonomic_errors
from src.utils.logger import setup_logger

logger = setup_logger("final_eval_test", log_dir=_PROJECT_ROOT / "logs")

DATASET_INDEX_DEV_PATH = _PROJECT_ROOT / "data" / "dev_test_split" / "dataset_index_dev_test_split.csv"
FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
RESULTS_DIR_TEST = _PROJECT_ROOT / "data" / "benchmark_results_test"

THRESHOLDS_DEV_JSON = (
    _PROJECT_ROOT.parent / "03-threshold-optimization" / "data" / "dev_test_split" / "thresholds_dev.json"
)
TAX_CLASS_YAML = _PROJECT_ROOT / "data" / "taxonomic_class_mapping.yaml"

CLASSIFIERS_FINAL = ["Nearest Centroid", "Linear SVM"]
BACKBONE_FINAL = "bioclip_v2"


def run_classification_eval() -> pd.DataFrame:
    """BioCLIP v2 + {Nearest Centroid, Linear SVM} sobre query_test."""
    evaluator = ModelEvaluator(
        DATASET_INDEX_DEV_PATH,
        FEATURES_DIR,
        output_dir=RESULTS_DIR_TEST,
        test_split_value="query_test",
    )
    evaluator.evaluate_model(BACKBONE_FINAL, classifier_names=CLASSIFIERS_FINAL)

    summary_path = RESULTS_DIR_TEST / "benchmark_summary.csv"
    summary_path.rename(RESULTS_DIR_TEST / "benchmark_summary_test.csv")

    pred_path = RESULTS_DIR_TEST / f"predictions_{BACKBONE_FINAL}.csv"
    pred_path.rename(RESULTS_DIR_TEST / f"predictions_{BACKBONE_FINAL}_test.csv")

    return pd.read_csv(RESULTS_DIR_TEST / "benchmark_summary_test.csv")


def run_taxonomic_breakdown() -> pd.DataFrame:
    """Desglose de errores taxonómicos para ambos clasificadores sobre query_test."""
    df_pred = pd.read_csv(RESULTS_DIR_TEST / f"predictions_{BACKBONE_FINAL}_test.csv")
    df_index = pd.read_csv(_PROJECT_ROOT / "data" / "dataset_index.csv")

    family_to_class = {}
    if TAX_CLASS_YAML.exists():
        with open(TAX_CLASS_YAML, encoding="utf-8") as f:
            class_to_families = yaml.safe_load(f)
        for cls, fams in class_to_families.items():
            for fam in (fams or []):
                family_to_class[fam] = cls

    y_true = df_pred["y_true"]
    rows = []
    for clf in CLASSIFIERS_FINAL:
        y_pred = df_pred[f"pred_{clf}"]
        counts = analyze_taxonomic_errors(y_true, y_pred, df_index, family_to_class)
        total = sum(counts.values())
        rows.append({
            "Classifier": clf,
            "n_total": total,
            **{f"{k}_n": v for k, v in counts.items()},
            **{f"{k}_%": v / total * 100 for k, v in counts.items()},
        })
    df_tax = pd.DataFrame(rows)
    df_tax.to_csv(RESULTS_DIR_TEST / "taxonomic_error_breakdown_test.csv", index=False)
    return df_tax


def run_threshold_validation() -> pd.DataFrame:
    """Aplica el umbral p95 centroide recalibrado en query_dev (congelado)
    a query_test y mide coverage / contaminación inter-clase observadas."""
    with open(THRESHOLDS_DEV_JSON, encoding="utf-8") as f:
        thresholds_dev = json.load(f)

    threshold = thresholds_dev["umbral_2"]["variants"]["centroid"]["global_percentiles"]["p95"]
    excluded = {e["species"] for e in thresholds_dev["excluded_classes"]}
    logger.info(f"Umbral congelado (p95 centroide, calibrado en query_dev): {threshold:.5f}")
    logger.info(f"Especies excluidas (gallery < 3): {len(excluded)}")

    index_df = pd.read_csv(DATASET_INDEX_DEV_PATH)
    gallery_df = index_df[index_df["split"] == "gallery"]
    test_df = index_df[index_df["split"] == "query_test"]

    def _load(df_subset):
        embs, labels = [], []
        for _, row in df_subset.iterrows():
            species_folder = row["species"].replace(" ", "_")
            stem = Path(row["filepath"]).stem
            npy_path = FEATURES_DIR / BACKBONE_FINAL / row["family"] / row["genus"] / species_folder / f"{stem}.npy"
            if not npy_path.exists():
                continue
            embs.append(np.load(npy_path).astype(np.float32).ravel())
            labels.append(row["species"])
        return np.vstack(embs), np.array(labels)

    gallery_embs, gallery_labels = _load(gallery_df)
    test_embs, test_labels = _load(test_df)

    gallery_normed = normalize(gallery_embs.astype(np.float64), norm="l2")
    centroids = {}
    for sp in np.unique(gallery_labels):
        mask = gallery_labels == sp
        mean_vec = gallery_normed[mask].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids[sp] = mean_vec / norm if norm > 0 else mean_vec

    test_normed = normalize(test_embs.astype(np.float64), norm="l2")

    dmin_intra = np.full(len(test_embs), np.nan)
    dmin_inter = np.full(len(test_embs), np.nan)
    for i, label in enumerate(test_labels):
        if label in centroids:
            dmin_intra[i] = 1.0 - float(test_normed[i] @ centroids[label])
        other_centroids = np.vstack([c for sp, c in centroids.items() if sp != label])
        dmin_inter[i] = float((1.0 - (test_normed[i] @ other_centroids.T)).min())

    mask = np.array([sp not in excluded for sp in test_labels]) & ~np.isnan(dmin_intra)
    intra_f = dmin_intra[mask]
    inter_f = dmin_inter[mask]

    coverage = float(np.mean(intra_f <= threshold))
    inter_contam = float(np.mean(inter_f <= threshold))
    sep_gap = float(np.median(inter_f) - threshold)

    df_out = pd.DataFrame([{
        "threshold_source": "query_dev p95 centroide (congelado)",
        "threshold_value": threshold,
        "n_query_test_included": int(mask.sum()),
        "n_query_test_excluded_species": int((~mask).sum()),
        "coverage": coverage,
        "inter_contamination": inter_contam,
        "separation_gap": sep_gap,
    }])
    df_out.to_csv(RESULTS_DIR_TEST / "threshold_validation_test.csv", index=False)
    logger.info(
        f"Validación de umbral sobre query_test: coverage={coverage:.4f} "
        f"inter_contam={inter_contam:.4f} gap={sep_gap:.4f}"
    )
    return df_out


def main() -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("   EVALUACIÓN FINAL ÚNICA SOBRE query_test (CONGELADA)")
    logger.info("=" * 60)

    if not DATASET_INDEX_DEV_PATH.exists():
        logger.error(f"No existe {DATASET_INDEX_DEV_PATH}.")
        return

    RESULTS_DIR_TEST.mkdir(parents=True, exist_ok=True)

    logger.info("--- Clasificación: BioCLIP v2 + {Nearest Centroid, Linear SVM} sobre query_test ---")
    df_summary = run_classification_eval()
    logger.info(f"\n{df_summary[['Embedding Model', 'Classifier', 'Accuracy', 'Top-5 Accuracy', 'F1-Macro']].to_string(index=False)}")

    logger.info("")
    logger.info("--- Desglose de errores taxonómicos sobre query_test ---")
    df_tax = run_taxonomic_breakdown()
    logger.info(f"\n{df_tax.to_string(index=False)}")

    logger.info("")
    logger.info("--- Validación del umbral (p95 centroide, calibrado en query_dev) sobre query_test ---")
    run_threshold_validation()

    logger.info("")
    logger.info("EVALUACIÓN FINAL SOBRE query_test COMPLETADA — no volver a tocar query_test.")


if __name__ == "__main__":
    main()
