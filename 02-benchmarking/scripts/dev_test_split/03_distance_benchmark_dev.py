"""Re-corre el benchmark de métricas de distancia (scripts/08_distance_benchmark.py)
usando solo query_dev, para no contaminar la selección de métrica con query_test.

Reutiliza las funciones de distancia, evaluación 1-NN y ploteo de
scripts/08_distance_benchmark.py sin modificarlo — solo cambia qué filas del
índice se cargan como "query" (acá: únicamente split == 'query_dev'; gallery
sin cambios).

Salidas (carpeta nueva, no pisa nada de data/benchmark_results/):
    data/benchmark_results_dev/distance_benchmark_dev.csv
    data/benchmark_results_dev/distance_distributions_dev.npz
    data/reports_dev/figures/distance_benchmark_accuracy_dev.{svg,png}
    data/reports_dev/figures/distance_benchmark_latency_dev.{svg,png}
    data/reports_dev/figures/distance_distributions/distance_distributions_{backbone}_dev.{svg,png}

Uso:
    python scripts/dev_test_split/03_distance_benchmark_dev.py
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/dev_test_split/
_SCRIPTS_DIR = _SCRIPT_DIR.parent                       # scripts/
_PROJECT_ROOT = _SCRIPTS_DIR.parent                     # 02-benchmarking/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("distance_benchmark_dev", log_dir=_PROJECT_ROOT / "logs")

# ---------------------------------------------------------------------------
# Importar funciones de 08_distance_benchmark.py sin ejecutar su main()
# (nombre de módulo empieza con dígito, no se puede hacer `import` normal)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "distance_benchmark_base", _SCRIPTS_DIR / "08_distance_benchmark.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

DATASET_INDEX_DEV_PATH = _PROJECT_ROOT / "data" / "dev_test_split" / "dataset_index_dev_test_split.csv"
BENCHMARK_RESULTS_DIR_DEV = _PROJECT_ROOT / "data" / "benchmark_results_dev"
FIGURES_DIR_DEV = _PROJECT_ROOT / "data" / "reports_dev" / "figures"
FIGURES_DIR_DIST_DEV = FIGURES_DIR_DEV / "distance_distributions"

BACKBONES = _base.BACKBONES         # {bioclip_v2, dinov3_small, dinov2_small, resnet50, convnextv2_tiny}
METRICS = _base.METRICS             # [cosine, euclidean, euclidean_l2norm, manhattan]
FEATURES_DIR = _PROJECT_ROOT / "data" / "features"


def load_embeddings_dev(backbone: str, index_df: pd.DataFrame, features_dir: Path):
    """Igual que _base.load_embeddings, pero solo toma split in {'gallery', 'query_dev'}
    y relabela 'query_dev' -> 'query' en memoria para reusar el resto del pipeline
    de 08_distance_benchmark.py sin modificarlo."""
    df = index_df[index_df["split"].isin(["gallery", "query_dev"])].copy()
    df["split"] = df["split"].replace({"query_dev": "query"})
    return _base.load_embeddings(backbone, df, features_dir)


def main() -> None:
    logger.info("")
    logger.info("==============================================")
    logger.info("   BENCHMARK DE MÉTRICAS DE DISTANCIA SOBRE query_dev")
    logger.info("==============================================")

    if not DATASET_INDEX_DEV_PATH.exists():
        logger.error(
            f"No existe {DATASET_INDEX_DEV_PATH}. "
            "Correr antes scripts/dev_test_split/01_generate_dev_test_index.py"
        )
        return

    BENCHMARK_RESULTS_DIR_DEV.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR_DEV.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR_DIST_DEV.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(DATASET_INDEX_DEV_PATH)
    logger.info(
        f"Índice cargado: {len(index_df)} registros "
        f"({(index_df['split']=='gallery').sum()} gallery, "
        f"{(index_df['split']=='query_dev').sum()} query_dev, "
        f"{(index_df['split']=='query_test').sum()} query_test [no se usa])"
    )

    records: list[dict] = []
    all_distributions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for backbone, dim in BACKBONES.items():
        logger.info("")
        logger.info(f"--- Backbone: {backbone} (dim={dim}) ---")

        splits = load_embeddings_dev(backbone, index_df, FEATURES_DIR)
        gallery_embs, gallery_labels = splits["gallery"]
        query_embs, query_labels = splits["query"]
        logger.info(f"  Gallery: {gallery_embs.shape}  |  Query_dev: {query_embs.shape}")

        dist_fns = {
            "cosine": _base.dist_cosine,
            "euclidean": _base.dist_euclidean,
            "euclidean_l2norm": _base.dist_euclidean_l2norm,
            "manhattan": _base.dist_manhattan,
        }

        all_distributions[backbone] = {}

        for metric in METRICS:
            logger.info(f"  Evaluando métrica: {metric}")
            fn = dist_fns[metric]

            accuracy, latency_dict = _base.run_1nn_evaluation(
                query_embs, query_labels, gallery_embs, gallery_labels, fn
            )
            logger.info(
                f"    Accuracy: {accuracy:.4f}  |  "
                f"mean={latency_dict['mean_ms']:.3f}ms  "
                f"median={latency_dict['median_ms']:.3f}ms  "
                f"p95={latency_dict['p95_ms']:.3f}ms"
            )

            intra, inter = _base.compute_distance_distributions(
                query_embs, query_labels, gallery_embs, gallery_labels, fn
            )
            all_distributions[backbone][metric] = (intra, inter)

            records.append({
                "backbone": backbone,
                "backbone_dim": dim,
                "metric": metric,
                "accuracy": accuracy,
                "mean_ms": latency_dict["mean_ms"],
                "median_ms": latency_dict["median_ms"],
                "p95_ms": latency_dict["p95_ms"],
            })

    results_df = pd.DataFrame(records)
    results_df["is_best"] = False
    for backbone in results_df["backbone"].unique():
        mask = results_df["backbone"] == backbone
        best_idx = results_df.loc[mask, "accuracy"].idxmax()
        results_df.loc[best_idx, "is_best"] = True

    csv_path = BENCHMARK_RESULTS_DIR_DEV / "distance_benchmark_dev.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nCSV guardado: {csv_path}")

    npz_arrays = {}
    for backbone, metric_dists in all_distributions.items():
        for metric, (intra, inter) in metric_dists.items():
            npz_arrays[f"{backbone}_{metric}_intra"] = intra
            npz_arrays[f"{backbone}_{metric}_inter"] = inter
    npz_path = BENCHMARK_RESULTS_DIR_DEV / "distance_distributions_dev.npz"
    np.savez(npz_path, **npz_arrays)
    logger.info(f"Distribuciones guardadas: {npz_path} ({len(npz_arrays)} arrays)")

    try:
        _base.plot_accuracy_bar(
            results_df.rename(columns={}), FIGURES_DIR_DEV
        )
        # Renombrar a sufijo _dev para no confundir con las figuras originales
        for stem in ("distance_benchmark_accuracy",):
            for ext in ("svg", "png"):
                src = FIGURES_DIR_DEV / f"{stem}.{ext}"
                if src.exists():
                    src.rename(FIGURES_DIR_DEV / f"{stem}_dev.{ext}")
        _base.plot_latency_bar(results_df, FIGURES_DIR_DEV)
        for stem in ("distance_benchmark_latency",):
            for ext in ("svg", "png"):
                src = FIGURES_DIR_DEV / f"{stem}.{ext}"
                if src.exists():
                    src.rename(FIGURES_DIR_DEV / f"{stem}_dev.{ext}")
        _base.plot_distance_distributions(all_distributions, results_df, FIGURES_DIR_DIST_DEV)
    except Exception as e:
        logger.warning(f"No se pudieron generar figuras: {e}")

    logger.info("")
    logger.info("=== RESUMEN — ordenado por accuracy descendente (query_dev) ===")
    summary = results_df.sort_values("accuracy", ascending=False)
    header = (
        f"{'backbone':<18} {'metric':<22} {'accuracy':>9} "
        f"{'mean_ms':>10} {'median_ms':>11} {'p95_ms':>10} {'is_best':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for _, row in summary.iterrows():
        best_flag = " *" if row["is_best"] else ""
        logger.info(
            f"{row['backbone']:<18} {row['metric']:<22} "
            f"{row['accuracy']:>9.4f} {row['mean_ms']:>9.3f}ms "
            f"{row['median_ms']:>10.3f}ms {row['p95_ms']:>9.3f}ms"
            f"{best_flag}"
        )
    logger.info("")
    logger.info("BENCHMARK DE DISTANCIA SOBRE query_dev COMPLETADO")


if __name__ == "__main__":
    main()
