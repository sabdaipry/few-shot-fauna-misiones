"""Re-corre la calibración de umbrales (scripts/01_calibrate_thresholds.py)
usando solo query_dev, para no contaminar la calibración con query_test.

Reutiliza (sin modificarlas) las funciones geométricas y de calibración de
01_calibrate_thresholds.py: compute_dmin_1nn, compute_dmin_inter,
compute_centroids, compute_dmin_centroid, calibrate_threshold,
compute_global_percentiles, compute_per_class_percentiles, funciones de
ploteo, etc. Solo cambia qué filas se cargan como "query" (acá: únicamente
split == 'query_dev'; gallery sin cambios) y las rutas de entrada/salida.

Salidas (carpeta nueva, no pisa nada de data/thresholds.json ni de
data/reports/figures/):
    03-threshold-optimization/data/dev_test_split/thresholds_dev.json
    03-threshold-optimization/data/reports_dev/figures/dinov2_small/ (4 figuras)
    03-threshold-optimization/data/reports_dev/figures/bioclip_v2/   (5 figuras)

Uso:
    python 03-threshold-optimization/scripts/dev_test_split/01_calibrate_thresholds_dev.py
"""
import json
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent            # scripts/dev_test_split/
_SCRIPTS_DIR = _SCRIPT_DIR.parent                         # scripts/
_MODULE_ROOT = _SCRIPTS_DIR.parent                         # 03-threshold-optimization/
_REPO_ROOT = _MODULE_ROOT.parent                           # raíz del repo
_BENCHMARKING_ROOT = _REPO_ROOT / "02-benchmarking"

sys.path.insert(0, str(_BENCHMARKING_ROOT))
from src.utils.logger import setup_logger

_LOG_DIR = _MODULE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger("calibrate_thresholds_dev", log_dir=_LOG_DIR)

# ---------------------------------------------------------------------------
# Importar funciones de 01_calibrate_thresholds.py sin ejecutar su main()
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "calibrate_thresholds_base", _SCRIPTS_DIR / "01_calibrate_thresholds.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DATASET_INDEX_DEV_PATH = _BENCHMARKING_ROOT / "data" / "dev_test_split" / "dataset_index_dev_test_split.csv"
FEATURES_DIR           = _BENCHMARKING_ROOT / "data" / "features"
NPZ_PATH_DEV           = _BENCHMARKING_ROOT / "data" / "benchmark_results_dev" / "distance_distributions_dev.npz"
OUTPUT_JSON            = _MODULE_ROOT / "data" / "dev_test_split" / "thresholds_dev.json"
FIGURES_BASE           = _MODULE_ROOT / "data" / "reports_dev" / "figures"

PERCENTILES = _base.PERCENTILES
MIN_GALLERY_SUPPORT = _base.MIN_GALLERY_SUPPORT


def compute_class_support_dev(index_df: pd.DataFrame) -> pd.DataFrame:
    """Igual que _base.compute_class_support pero usando 'query_dev' en vez
    de 'query' (gallery no cambia; query_test no debe influir en nada acá)."""
    gallery = index_df[index_df["split"] == "gallery"].groupby("species").size()
    query = index_df[index_df["split"] == "query_dev"].groupby("species").size()
    all_species = index_df["species"].unique()

    records = []
    for sp in all_species:
        g = int(gallery.get(sp, 0))
        q = int(query.get(sp, 0))
        records.append({"species": sp, "gallery_count": g, "query_count": q,
                         "excluded": g < MIN_GALLERY_SUPPORT})
    return pd.DataFrame(records).sort_values("gallery_count").reset_index(drop=True)


def load_embeddings_dev(backbone: str, index_df: pd.DataFrame, features_dir: Path):
    """Igual que _base.load_embeddings, pero solo toma split in {'gallery',
    'query_dev'} y relabela 'query_dev' -> 'query' en memoria para reusar
    el resto de las funciones de 01_calibrate_thresholds.py sin tocarlo."""
    df = index_df[index_df["split"].isin(["gallery", "query_dev"])].copy()
    df["split"] = df["split"].replace({"query_dev": "query"})
    return _base.load_embeddings(backbone, df, features_dir)


def main() -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("   CALIBRACIÓN DE UMBRALES SOBRE query_dev")
    logger.info("=" * 60)

    if not DATASET_INDEX_DEV_PATH.exists():
        logger.error(
            f"No existe {DATASET_INDEX_DEV_PATH}. "
            "Correr antes scripts/dev_test_split/01_generate_dev_test_index.py "
            "(en 02-benchmarking)."
        )
        return
    if not NPZ_PATH_DEV.exists():
        logger.error(
            f"No existe {NPZ_PATH_DEV}. "
            "Correr antes scripts/dev_test_split/03_distance_benchmark_dev.py "
            "(en 02-benchmarking)."
        )
        return

    output_data_dir = OUTPUT_JSON.parent
    figures_dino = FIGURES_BASE / "dinov2_small"
    figures_bio = FIGURES_BASE / "bioclip_v2"
    for d in (output_data_dir, figures_dino, figures_bio):
        d.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(DATASET_INDEX_DEV_PATH)
    logger.info(f"Índice dev/test cargado: {len(index_df)} registros")

    support_df = compute_class_support_dev(index_df)
    excluded = _base.get_excluded_classes(support_df)

    logger.info(f"Clases incluidas: {len(support_df) - len(excluded)}")
    logger.info(f"Clases excluidas (gallery < {MIN_GALLERY_SUPPORT}): {len(excluded)}")
    for sp in sorted(excluded):
        row = support_df[support_df["species"] == sp].iloc[0]
        logger.warning(
            f"  EXCLUIDA: {sp}  (gallery={row['gallery_count']}, query_dev={row['query_count']})"
        )

    npz_data = np.load(NPZ_PATH_DEV)
    logger.info(f"NPZ (dev) cargado: {NPZ_PATH_DEV.name}")

    excluded_json = []
    for sp in sorted(excluded):
        row = support_df[support_df["species"] == sp].iloc[0]
        excluded_json.append({
            "species": sp,
            "gallery_count": int(row["gallery_count"]),
            "query_dev_count": int(row["query_count"]),
            "reason": f"gallery_count < {MIN_GALLERY_SUPPORT}",
        })

    n_query_included = int(
        index_df[
            (index_df["split"] == "query_dev") & (~index_df["species"].isin(excluded))
        ].shape[0]
    )
    logger.info(f"Imágenes query_dev incluidas: {n_query_included}")

    # -----------------------------------------------------------------------
    # UMBRAL 1 — DINOv2 Small
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- UMBRAL 1: DINOv2 Small (coseno) — query_dev ---")

    splits_dino = load_embeddings_dev("dinov2_small", index_df, FEATURES_DIR)
    gallery_embs_dino, gallery_labels_dino = splits_dino["gallery"]
    query_embs_dino, query_labels_dino = splits_dino["query"]
    logger.info(f"  Gallery: {gallery_embs_dino.shape}  |  Query_dev: {query_embs_dino.shape}")

    dmin_intra_dino, labels_dino = _base.compute_dmin_1nn(
        query_embs_dino, query_labels_dino, gallery_embs_dino, gallery_labels_dino,
    )
    dmin_inter_dino, _ = _base.compute_dmin_inter(
        query_embs_dino, query_labels_dino, gallery_embs_dino, gallery_labels_dino,
    )

    umbral_1_dict = _base.calibrate_threshold(
        dmin_intra=dmin_intra_dino,
        dmin_inter=dmin_inter_dino,
        species_labels=labels_dino,
        excluded=excluded,
        backbone="dinov2_small",
        description="Filtro fauna conocida / desconocida (open-set detection) — query_dev",
        n_query_included=n_query_included,
    )

    logger.info("  Percentiles globales Umbral 1 (query_dev):")
    for k, v in umbral_1_dict["global_percentiles"].items():
        stats = umbral_1_dict["per_percentile_stats"][k]
        logger.info(
            f"    {k}: threshold={v:.4f}  coverage={stats['coverage']:.4f}"
            f"  inter_contam={stats['inter_contamination']:.4f}  gap={stats['separation_gap']:.4f}"
        )

    try:
        _base.plot_intra_inter_distribution(
            dmin_intra_npz=npz_data["dinov2_small_cosine_intra"],
            dmin_inter_npz=npz_data["dinov2_small_cosine_inter"],
            global_percentiles=umbral_1_dict["global_percentiles"],
            backbone_name="DINOv2 Small (dev)",
            out_dir=figures_dino,
        )
        _base.plot_percentile_candidates_stats(
            per_percentile_stats=umbral_1_dict["per_percentile_stats"],
            backbone_name="DINOv2 Small (dev)", variant_label="1-NN", out_dir=figures_dino,
        )
        _base.plot_per_class_coverage(
            dmin_intra=dmin_intra_dino, species_labels=labels_dino, excluded=excluded,
            global_percentiles=umbral_1_dict["global_percentiles"],
            backbone_name="DINOv2 Small (dev)", out_dir=figures_dino,
        )
        _base.plot_per_class_dmin_boxplots(
            dmin_intra=dmin_intra_dino, species_labels=labels_dino, excluded=excluded,
            global_percentiles=umbral_1_dict["global_percentiles"],
            backbone_name="DINOv2 Small (dev)", out_dir=figures_dino,
        )
    except Exception as e:
        logger.warning(f"No se pudieron generar figuras de Umbral 1: {e}")

    # -----------------------------------------------------------------------
    # UMBRAL 2 — BioCLIP v2
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- UMBRAL 2: BioCLIP v2 (coseno) — query_dev ---")

    splits_bio = load_embeddings_dev("bioclip_v2", index_df, FEATURES_DIR)
    gallery_embs_bio, gallery_labels_bio = splits_bio["gallery"]
    query_embs_bio, query_labels_bio = splits_bio["query"]
    logger.info(f"  Gallery: {gallery_embs_bio.shape}  |  Query_dev: {query_embs_bio.shape}")

    dmin_intra_1nn, labels_bio = _base.compute_dmin_1nn(
        query_embs_bio, query_labels_bio, gallery_embs_bio, gallery_labels_bio,
    )
    dmin_inter_bio, _ = _base.compute_dmin_inter(
        query_embs_bio, query_labels_bio, gallery_embs_bio, gallery_labels_bio,
    )

    centroids = _base.compute_centroids(gallery_embs_bio, gallery_labels_bio)
    logger.info(f"  Centroides calculados para {len(centroids)} especies (a partir del gallery, sin cambios).")

    dmin_intra_centroid, _ = _base.compute_dmin_centroid(
        query_embs_bio, query_labels_bio, centroids,
    )

    calib_1nn = _base.calibrate_threshold(
        dmin_intra=dmin_intra_1nn, dmin_inter=dmin_inter_bio,
        species_labels=labels_bio, excluded=excluded, backbone="bioclip_v2",
        description="Confianza de predicción taxonómica — variante 1-NN — query_dev",
        n_query_included=n_query_included,
    )
    calib_centroid = _base.calibrate_threshold(
        dmin_intra=dmin_intra_centroid, dmin_inter=dmin_inter_bio,
        species_labels=labels_bio, excluded=excluded, backbone="bioclip_v2",
        description="Confianza de predicción taxonómica — variante centroide — query_dev",
        n_query_included=n_query_included,
    )

    logger.info("  Percentiles globales Umbral 2 — variante 1-NN (query_dev):")
    for k, v in calib_1nn["global_percentiles"].items():
        stats = calib_1nn["per_percentile_stats"][k]
        logger.info(
            f"    {k}: threshold={v:.4f}  coverage={stats['coverage']:.4f}"
            f"  inter_contam={stats['inter_contamination']:.4f}  gap={stats['separation_gap']:.4f}"
        )
    logger.info("  Percentiles globales Umbral 2 — variante centroide (query_dev):")
    for k, v in calib_centroid["global_percentiles"].items():
        stats = calib_centroid["per_percentile_stats"][k]
        logger.info(
            f"    {k}: threshold={v:.4f}  coverage={stats['coverage']:.4f}"
            f"  inter_contam={stats['inter_contamination']:.4f}  gap={stats['separation_gap']:.4f}"
        )

    umbral_2_dict = {
        "backbone": "bioclip_v2",
        "metric": "cosine",
        "description": "Confianza de predicción taxonómica — query_dev",
        "n_included_classes": calib_1nn["n_included_classes"],
        "n_query_included": n_query_included,
        "variants": {
            "nearest_neighbor": {
                "description": calib_1nn["description"],
                "global_percentiles": calib_1nn["global_percentiles"],
                "per_percentile_stats": calib_1nn["per_percentile_stats"],
            },
            "centroid": {
                "description": calib_centroid["description"],
                "global_percentiles": calib_centroid["global_percentiles"],
                "per_percentile_stats": calib_centroid["per_percentile_stats"],
            },
        },
        "per_class_percentiles": calib_1nn["per_class_percentiles"],
        "recommended_percentile": None,
        "recommended_threshold": None,
    }

    try:
        _base.plot_intra_inter_distribution(
            dmin_intra_npz=npz_data["bioclip_v2_cosine_intra"],
            dmin_inter_npz=npz_data["bioclip_v2_cosine_inter"],
            global_percentiles=calib_1nn["global_percentiles"],
            backbone_name="BioCLIP v2 (dev)", out_dir=figures_bio,
        )
        _base.plot_percentile_candidates_stats(
            per_percentile_stats=calib_1nn["per_percentile_stats"],
            backbone_name="BioCLIP v2 (dev)", variant_label="1-NN", out_dir=figures_bio,
            fig_stem="02_percentile_candidates_stats_1nn",
        )
        _base.plot_percentile_candidates_stats(
            per_percentile_stats=calib_centroid["per_percentile_stats"],
            backbone_name="BioCLIP v2 (dev)", variant_label="Centroide", out_dir=figures_bio,
            fig_stem="02_percentile_candidates_stats_centroid",
        )
        _base.plot_per_class_coverage(
            dmin_intra=dmin_intra_1nn, species_labels=labels_bio, excluded=excluded,
            global_percentiles=calib_1nn["global_percentiles"],
            backbone_name="BioCLIP v2 (dev)", out_dir=figures_bio,
        )
        _base.plot_per_class_dmin_boxplots(
            dmin_intra=dmin_intra_1nn, species_labels=labels_bio, excluded=excluded,
            global_percentiles=calib_1nn["global_percentiles"],
            backbone_name="BioCLIP v2 (dev)", out_dir=figures_bio,
        )
        _base.plot_per_class_coverage(
            dmin_intra=dmin_intra_centroid, species_labels=labels_bio, excluded=excluded,
            global_percentiles=calib_centroid["global_percentiles"],
            backbone_name="BioCLIP v2 (dev)", out_dir=figures_bio,
            variant_label="Centroide", fig_stem="03_per_class_coverage_centroid",
        )
        _base.plot_per_class_dmin_boxplots(
            dmin_intra=dmin_intra_centroid, species_labels=labels_bio, excluded=excluded,
            global_percentiles=calib_centroid["global_percentiles"],
            backbone_name="BioCLIP v2 (dev)", out_dir=figures_bio,
            variant_label="Centroide", fig_stem="04_per_class_dmin_boxplots_centroid",
        )
        _base.plot_1nn_vs_centroid_comparison(
            dmin_1nn=dmin_intra_1nn, dmin_centroid=dmin_intra_centroid,
            species_labels=labels_bio, excluded=excluded,
            pcts_1nn=calib_1nn["global_percentiles"], pcts_centroid=calib_centroid["global_percentiles"],
            out_dir=figures_bio,
        )
    except Exception as e:
        logger.warning(f"No se pudieron generar figuras de Umbral 2: {e}")

    # -----------------------------------------------------------------------
    # Guardar thresholds_dev.json
    # -----------------------------------------------------------------------
    thresholds_out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "min_gallery_support": MIN_GALLERY_SUPPORT,
            "percentile_candidates": PERCENTILES,
            "metric": "cosine",
            "seed": _base.SEED,
            "split_used": "query_dev (70% estratificado del query set original, seed=29)",
        },
        "excluded_classes": excluded_json,
        "umbral_1": umbral_1_dict,
        "umbral_2": umbral_2_dict,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds_out, f, indent=2, ensure_ascii=False)
    logger.info(f"\nthresholds_dev.json guardado: {OUTPUT_JSON}")

    # -----------------------------------------------------------------------
    # Resumen final
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=== RESUMEN DE CALIBRACIÓN (query_dev) ===")
    header = f"{'Umbral':<12} {'Variante':<18} {'Percentil':>10} {'Threshold':>12} {'Coverage':>10} {'Inter%':>10} {'Gap':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for p in PERCENTILES:
        key = f"p{p}"
        stats = umbral_1_dict["per_percentile_stats"][key]
        logger.info(
            f"{'Umbral 1':<12} {'1-NN':<18} {key:>10} "
            f"{stats['threshold_value']:>12.4f} {stats['coverage']:>10.4f} "
            f"{stats['inter_contamination']:>10.4f} {stats['separation_gap']:>10.4f}"
        )

    logger.info("")
    for variant_key, variant_label in [("nearest_neighbor", "1-NN"), ("centroid", "Centroide")]:
        for p in PERCENTILES:
            key = f"p{p}"
            stats = umbral_2_dict["variants"][variant_key]["per_percentile_stats"][key]
            logger.info(
                f"{'Umbral 2':<12} {variant_label:<18} {key:>10} "
                f"{stats['threshold_value']:>12.4f} {stats['coverage']:>10.4f} "
                f"{stats['inter_contamination']:>10.4f} {stats['separation_gap']:>10.4f}"
            )

    logger.info("")
    logger.info("CALIBRACIÓN SOBRE query_dev COMPLETADA")


if __name__ == "__main__":
    main()
