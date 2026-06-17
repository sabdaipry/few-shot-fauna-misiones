"""Calibración de umbrales para el pipeline de clasificación few-shot de fauna silvestre.

Computa los umbrales de decisión para dos etapas del pipeline en cascada:

  Umbral 1 (DINOv2 Small + coseno):
      Filtro open-set — distingue fauna conocida de fauna desconocida.
  Umbral 2 (BioCLIP v2 + coseno):
      Confianza de predicción taxonómica — dos variantes: vecino más cercano
      y centroide de clase.

Para cada umbral se calculan percentiles globales (p90, p95, p97, p99) sobre
las distribuciones de d_min intra-clase del query set, excluyendo las clases
con soporte de gallery < 3 imágenes.

Salidas:
    03-threshold-optimization/data/thresholds.json
    03-threshold-optimization/data/reports/figures/dinov2_small/  (4 figuras)
    03-threshold-optimization/data/reports/figures/bioclip_v2/    (5 figuras)

Uso:
    python 03-threshold-optimization/scripts/01_calibrate_thresholds.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Ajuste de rutas para importar el logger desde 02-benchmarking/src/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent           # 03-threshold-optimization/scripts/
_MODULE_ROOT = _SCRIPT_DIR.parent                        # 03-threshold-optimization/
_REPO_ROOT = _MODULE_ROOT.parent                         # raíz del repo
_BENCHMARKING_ROOT = _REPO_ROOT / "02-benchmarking"

sys.path.insert(0, str(_BENCHMARKING_ROOT))
from src.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_LOG_DIR = _MODULE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger("calibrate_thresholds", log_dir=_LOG_DIR)

# ---------------------------------------------------------------------------
# Constantes y paths
# ---------------------------------------------------------------------------
DATASET_INDEX_PATH = _BENCHMARKING_ROOT / "data" / "dataset_index.csv"
FEATURES_DIR       = _BENCHMARKING_ROOT / "data" / "features"
NPZ_PATH           = _BENCHMARKING_ROOT / "data" / "benchmark_results" / "distance_distributions.npz"
OUTPUT_JSON        = _MODULE_ROOT / "data" / "thresholds.json"
FIGURES_BASE       = _MODULE_ROOT / "data" / "reports" / "figures"

BACKBONES: list[str] = ["dinov2_small", "bioclip_v2"]
PERCENTILES: list[int] = [90, 95, 97, 99]
MIN_GALLERY_SUPPORT: int = 3
SEED: int = 29

COLOR_INTRA = "#2d6a4f"   # verde oscuro
COLOR_INTER = "#95d5b2"   # verde claro

_GREENS = matplotlib.colormaps["Greens"]
_PERCENTILE_COLORS = {
    90: _GREENS(0.40),
    95: _GREENS(0.55),
    97: _GREENS(0.70),
    99: _GREENS(0.90),
}


# ===========================================================================
# SECCIÓN 1 — Carga de datos
# ===========================================================================

def load_dataset_index(path: Path) -> pd.DataFrame:
    """Carga el índice del dataset desde CSV.

    Args:
        path: Ruta al archivo dataset_index.csv.

    Returns:
        DataFrame con columnas filepath, species, genus, family, split,
        ivc_score, ivc_category.
    """
    df = pd.read_csv(path)
    logger.info(f"Índice cargado: {len(df)} registros desde {path.name}")
    return df


def compute_class_support(index_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula soporte en gallery y query por especie.

    Args:
        index_df: DataFrame del índice del dataset.

    Returns:
        DataFrame con columnas species, gallery_count, query_count, excluded,
        ordenado por gallery_count ascendente.
    """
    gallery = index_df[index_df["split"] == "gallery"].groupby("species").size()
    query   = index_df[index_df["split"] == "query"].groupby("species").size()
    all_species = index_df["species"].unique()

    records = []
    for sp in all_species:
        g = int(gallery.get(sp, 0))
        q = int(query.get(sp, 0))
        records.append({"species": sp, "gallery_count": g, "query_count": q,
                        "excluded": g < MIN_GALLERY_SUPPORT})

    return pd.DataFrame(records).sort_values("gallery_count").reset_index(drop=True)


def get_excluded_classes(support_df: pd.DataFrame) -> set[str]:
    """Extrae el conjunto de especies excluidas por soporte insuficiente.

    Args:
        support_df: DataFrame retornado por compute_class_support.

    Returns:
        set de nombres de especies excluidas.
    """
    return set(support_df.loc[support_df["excluded"], "species"].tolist())


def load_embeddings(
    backbone: str,
    index_df: pd.DataFrame,
    features_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Carga los embeddings .npy para un backbone y retorna gallery y query.

    La ruta de cada archivo se construye como:
        features_dir/{backbone}/{family}/{genus}/{species_underscored}/{stem}.npy

    Args:
        backbone: Nombre del backbone (subcarpeta en features_dir).
        index_df: DataFrame del índice del dataset.
        features_dir: Ruta raíz de la carpeta de features.

    Returns:
        Dict con claves 'gallery' y 'query'. Cada valor es una tupla
        (embeddings float32 shape (N, D), labels array de strings).
    """
    result: dict[str, tuple[list, list]] = {
        "gallery": ([], []),
        "query":   ([], []),
    }
    backbone_dir = features_dir / backbone
    missing = 0

    for _, row in index_df.iterrows():
        species_folder = row["species"].replace(" ", "_")
        stem = Path(row["filepath"]).stem
        npy_path = (backbone_dir / row["family"] / row["genus"]
                    / species_folder / f"{stem}.npy")

        if not npy_path.exists():
            logger.warning(f"No encontrado: {npy_path}")
            missing += 1
            continue

        emb  = np.load(npy_path).astype(np.float32).ravel()
        split = row["split"]
        result[split][0].append(emb)
        result[split][1].append(row["species"])

    if missing:
        logger.warning(f"[{backbone}] {missing} archivos .npy omitidos.")

    return {
        split: (np.vstack(embs), np.array(labels))
        for split, (embs, labels) in result.items()
        if embs
    }


# ===========================================================================
# SECCIÓN 2 — Cómputo geométrico
# ===========================================================================

def _cosine_distance_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Matriz de distancia coseno (N_query × M_gallery).

    Args:
        query:   Embeddings query, shape (N, D).
        gallery: Embeddings gallery, shape (M, D).

    Returns:
        Matriz float64 de shape (N, M) con valores en [0, 2].
    """
    q_norm = normalize(query.astype(np.float64), norm="l2")
    g_norm = normalize(gallery.astype(np.float64), norm="l2")
    return 1.0 - (q_norm @ g_norm.T)


def compute_dmin_1nn(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distancia coseno mínima de cada query a su vecino más cercano intra-clase.

    Para cada imagen query calcula la distancia al gallery más cercano de la
    misma especie. Si la especie no tiene representación en gallery se emite nan.

    Args:
        query_embs:    Embeddings query, shape (N, D).
        query_labels:  Etiquetas de especie para cada query, shape (N,).
        gallery_embs:  Embeddings gallery, shape (M, D).
        gallery_labels: Etiquetas de especie para cada gallery, shape (M,).

    Returns:
        Tupla (dmin_values, species_labels) — arrays paralelos de longitud N.
        dmin_values es float64 con nan para clases sin representación en gallery.
    """
    dist_matrix = _cosine_distance_matrix(query_embs, gallery_embs)
    dmin  = np.full(len(query_embs), np.nan, dtype=np.float64)

    for i, label in enumerate(query_labels):
        same_mask = gallery_labels == label
        if same_mask.any():
            dmin[i] = dist_matrix[i, same_mask].min()

    return dmin, query_labels.copy()


def compute_dmin_inter(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distancia coseno mínima de cada query al vecino más cercano inter-clase.

    Para cada imagen query calcula la distancia al gallery más cercano de una
    especie diferente. Si no existe ningún gallery de distinta especie se emite nan.

    Args:
        query_embs:    Embeddings query, shape (N, D).
        query_labels:  Etiquetas de especie para cada query, shape (N,).
        gallery_embs:  Embeddings gallery, shape (M, D).
        gallery_labels: Etiquetas de especie para cada gallery, shape (M,).

    Returns:
        Tupla (dmin_inter_values, species_labels) — arrays paralelos de longitud N.
    """
    dist_matrix = _cosine_distance_matrix(query_embs, gallery_embs)
    dmin_inter  = np.full(len(query_embs), np.nan, dtype=np.float64)

    for i, label in enumerate(query_labels):
        diff_mask = gallery_labels != label
        if diff_mask.any():
            dmin_inter[i] = dist_matrix[i, diff_mask].min()

    return dmin_inter, query_labels.copy()


def compute_centroids(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula el centroide L2-normalizado de cada especie en gallery.

    El centroide se obtiene promediando los embeddings L2-normalizados de cada
    clase y L2-normalizando el resultado.

    Args:
        gallery_embs:   Embeddings gallery, shape (M, D).
        gallery_labels: Etiquetas de especie, shape (M,).

    Returns:
        Dict {species: centroid_vector shape (D,)} en float64.
    """
    gallery_normed = normalize(gallery_embs.astype(np.float64), norm="l2")
    centroids: dict[str, np.ndarray] = {}

    for sp in np.unique(gallery_labels):
        mask = gallery_labels == sp
        mean_vec = gallery_normed[mask].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids[sp] = mean_vec / norm if norm > 0 else mean_vec

    return centroids


def compute_dmin_centroid(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Distancia coseno de cada query al centroide de su propia especie.

    Args:
        query_embs:   Embeddings query, shape (N, D).
        query_labels: Etiquetas de especie para cada query, shape (N,).
        centroids:    Dict {species: centroid_vector} de gallery.

    Returns:
        Tupla (dmin_centroid_values, species_labels). nan si la especie no
        tiene centroide (no aparece en gallery).
    """
    query_normed = normalize(query_embs.astype(np.float64), norm="l2")
    dmin = np.full(len(query_embs), np.nan, dtype=np.float64)

    for i, label in enumerate(query_labels):
        if label in centroids:
            c = centroids[label]
            dmin[i] = 1.0 - float(query_normed[i] @ c)

    return dmin, query_labels.copy()


# ===========================================================================
# SECCIÓN 3 — Calibración
# ===========================================================================

def _filter_included(
    dmin_values: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Filtra valores de clases excluidas y NaN.

    Args:
        dmin_values:    Array de distancias mínimas.
        species_labels: Array de etiquetas de especie paralelo.
        excluded:       Set de especies a excluir.

    Returns:
        Tupla (dmin_filtered, labels_filtered) sin clases excluidas ni NaN.
    """
    mask = np.array([sp not in excluded for sp in species_labels])
    mask &= ~np.isnan(dmin_values)
    return dmin_values[mask], species_labels[mask]


def compute_global_percentiles(
    dmin_values: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
    percentiles: list[int] = PERCENTILES,
) -> dict[str, float]:
    """Computa percentiles globales sobre la distribución filtrada de d_min.

    Args:
        dmin_values:    Array de distancias mínimas intra-clase.
        species_labels: Etiquetas paralelas a dmin_values.
        excluded:       Especies a excluir del cómputo.
        percentiles:    Lista de percentiles a computar, ej. [90, 95, 97, 99].

    Returns:
        Dict {f"p{p}": valor_float} por cada percentil.
    """
    vals, _ = _filter_included(dmin_values, species_labels, excluded)
    return {f"p{p}": float(np.percentile(vals, p)) for p in percentiles}


def compute_per_class_percentiles(
    dmin_values: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
    percentiles: list[int] = PERCENTILES,
) -> pd.DataFrame:
    """Computa percentiles de d_min por especie (solo clases incluidas).

    Args:
        dmin_values:    Array de distancias mínimas intra-clase.
        species_labels: Etiquetas paralelas a dmin_values.
        excluded:       Especies a excluir.
        percentiles:    Lista de percentiles a computar.

    Returns:
        DataFrame con columnas: species, n_query, p90, p95, p97, p99.
        Una fila por especie incluida.
    """
    vals, labs = _filter_included(dmin_values, species_labels, excluded)
    records = []
    for sp in np.unique(labs):
        sp_vals = vals[labs == sp]
        row: dict = {"species": sp, "n_query": len(sp_vals)}
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(sp_vals, p))
        records.append(row)
    return pd.DataFrame(records).sort_values("species").reset_index(drop=True)


def compute_threshold_stats(
    dmin_intra: np.ndarray,
    dmin_inter: np.ndarray,
    species_labels_intra: np.ndarray,
    species_labels_inter: np.ndarray,
    excluded: set[str],
    threshold: float,
) -> dict[str, float]:
    """Evalúa un candidato de umbral con métricas de separación.

    Args:
        dmin_intra:          Array de d_min intra-clase (todos los queries).
        dmin_inter:          Array de d_min inter-clase (todos los queries).
        species_labels_intra: Etiquetas paralelas a dmin_intra.
        species_labels_inter: Etiquetas paralelas a dmin_inter.
        excluded:            Especies a excluir del cómputo.
        threshold:           Valor del umbral candidato.

    Returns:
        Dict con:
            threshold_value:      Valor absoluto del umbral.
            coverage:             Fracción de intra_filtered ≤ threshold (recall).
            inter_contamination:  Fracción de inter_filtered ≤ threshold (FPR aprox.).
            separation_gap:       median(inter_filtered) − threshold.
    """
    intra_f, _ = _filter_included(dmin_intra, species_labels_intra, excluded)
    inter_f, _ = _filter_included(dmin_inter, species_labels_inter, excluded)

    coverage     = float(np.mean(intra_f <= threshold))
    inter_contam = float(np.mean(inter_f <= threshold))
    sep_gap      = float(np.median(inter_f) - threshold)

    return {
        "threshold_value":     float(threshold),
        "coverage":            round(coverage, 6),
        "inter_contamination": round(inter_contam, 6),
        "separation_gap":      round(sep_gap, 6),
    }


def calibrate_threshold(
    dmin_intra: np.ndarray,
    dmin_inter: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
    backbone: str,
    description: str,
    n_query_included: int,
    percentiles: list[int] = PERCENTILES,
) -> dict:
    """Orquesta la calibración completa para un backbone y una variante.

    Args:
        dmin_intra:        Array de d_min intra-clase (longitud N_query total).
        dmin_inter:        Array de d_min inter-clase (longitud N_query total).
        species_labels:    Etiquetas por imagen query, paralelas a los arrays.
        excluded:          Especies excluidas del cómputo de percentiles.
        backbone:          Nombre del backbone.
        description:       Descripción del rol del umbral.
        n_query_included:  Número de imágenes query de clases incluidas.
        percentiles:       Candidatos de percentil.

    Returns:
        Dict con estructura lista para insertar en thresholds.json.
    """
    global_pcts = compute_global_percentiles(dmin_intra, species_labels, excluded, percentiles)
    per_class   = compute_per_class_percentiles(dmin_intra, species_labels, excluded, percentiles)

    per_pct_stats: dict[str, dict] = {}
    for p in percentiles:
        key = f"p{p}"
        stats = compute_threshold_stats(
            dmin_intra, dmin_inter, species_labels, species_labels, excluded,
            threshold=global_pcts[key],
        )
        per_pct_stats[key] = stats

    # per_class_percentiles como dict anidado {species: {pXX: val, n_query: n}}
    pc_dict: dict[str, dict] = {}
    for _, row in per_class.iterrows():
        sp = row["species"]
        pc_dict[sp] = {
            "n_query": int(row["n_query"]),
            **{f"p{p}": float(row[f"p{p}"]) for p in percentiles},
        }

    return {
        "backbone":               backbone,
        "metric":                 "cosine",
        "description":            description,
        "n_included_classes":     len(per_class),
        "n_query_included":       n_query_included,
        "global_percentiles":     global_pcts,
        "per_percentile_stats":   per_pct_stats,
        "per_class_percentiles":  pc_dict,
        "recommended_percentile": None,
        "recommended_threshold":  None,
    }


# ===========================================================================
# SECCIÓN 4 — Visualización
# ===========================================================================

def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Guarda una figura como .svg y .png."""
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    logger.info(f"Figura guardada: {out_dir.name}/{stem}.{{svg,png}}")
    plt.close(fig)


def plot_intra_inter_distribution(
    dmin_intra_npz: np.ndarray,
    dmin_inter_npz: np.ndarray,
    global_percentiles: dict[str, float],
    backbone_name: str,
    out_dir: Path,
) -> None:
    """Histograma de densidad intra vs inter-clase con líneas de percentil.

    Usa los arrays del NPZ (distribución completa, incluyendo clases excluidas)
    como overlay de contexto global. Los percentiles corresponden a la
    distribución filtrada (solo clases incluidas).

    Args:
        dmin_intra_npz:     Array de d_min intra del NPZ (3674 valores).
        dmin_inter_npz:     Array de d_min inter del NPZ (3674 valores).
        global_percentiles: Dict {f"p{p}": valor} de la calibración filtrada.
        backbone_name:      Nombre del backbone para el título.
        out_dir:            Directorio de salida para la figura.
    """
    intra_clean = dmin_intra_npz[~np.isnan(dmin_intra_npz)]
    inter_clean = dmin_inter_npz[~np.isnan(dmin_inter_npz)]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(intra_clean, bins=80, alpha=0.65, density=True,
            color=COLOR_INTRA, label=f"Intra-clase (n={len(intra_clean)})")
    ax.hist(inter_clean, bins=80, alpha=0.65, density=True,
            color=COLOR_INTER, label=f"Inter-clase (n={len(inter_clean)})")

    for p, color in _PERCENTILE_COLORS.items():
        val = global_percentiles[f"p{p}"]
        ax.axvline(val, color=color, linewidth=1.8, linestyle="--",
                   label=f"p{p} = {val:.4f}")

    ax.set_xlabel("Distancia coseno", fontsize=14)
    ax.set_ylabel("Densidad", fontsize=14)
    ax.set_title(
        f"Distribuciones de distancia intra vs inter-clase — {backbone_name}",
        fontsize=18, fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, out_dir, "01_intra_inter_distribution")


def plot_percentile_candidates_stats(
    per_percentile_stats: dict[str, dict],
    backbone_name: str,
    variant_label: str,
    out_dir: Path,
    fig_stem: str = "02_percentile_candidates_stats",
) -> None:
    """Tabla visual (heatmap 4×3) con métricas por candidato de percentil.

    Muestra threshold_value, coverage, inter_contamination y separation_gap
    para cada uno de los 4 candidatos de percentil.

    Args:
        per_percentile_stats: Dict {f"p{p}": stats_dict} de calibrate_threshold.
        backbone_name:        Nombre del backbone para el título.
        variant_label:        Etiqueta de variante ("1-NN" o "Centroide").
        out_dir:              Directorio de salida.
        fig_stem:             Nombre base del archivo de salida.
    """
    keys    = [f"p{p}" for p in PERCENTILES]
    metrics = ["threshold_value", "coverage", "inter_contamination", "separation_gap"]
    labels  = ["Umbral (coseno)", "Coverage", "Inter contamination", "Separation gap"]

    data = np.array([
        [per_percentile_stats[k][m] for k in keys]
        for m in metrics
    ])

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.5))

    for ax, row_data, metric_label in zip(axes, data, labels):
        colors = [_PERCENTILE_COLORS[p] for p in PERCENTILES]
        bars = ax.bar(keys, row_data, color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, row_data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(row_data.max()) * 0.03,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
        ax.set_title(metric_label, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        if metric_label == "Separation gap":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle(
        f"Estadísticas por percentil candidato — {backbone_name} ({variant_label})",
        fontsize=18, fontweight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, out_dir, fig_stem)


def plot_per_class_coverage(
    dmin_intra: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
    global_percentiles: dict[str, float],
    backbone_name: str,
    out_dir: Path,
) -> None:
    """Barras horizontales de cobertura por especie bajo el umbral p95.

    Muestra una barra por especie (82 incluidas) con la fracción de imágenes
    query con d_min ≤ umbral p95. Líneas verticales de referencia en los
    4 candidatos de percentil.

    Args:
        dmin_intra:         Array de d_min intra-clase.
        species_labels:     Etiquetas paralelas a dmin_intra.
        excluded:           Especies a excluir.
        global_percentiles: Dict {f"p{p}": valor} de calibrate_threshold.
        backbone_name:      Nombre del backbone para el título.
        out_dir:            Directorio de salida.
    """
    vals, labs = _filter_included(dmin_intra, species_labels, excluded)
    threshold_p95 = global_percentiles["p95"]

    species_list = np.unique(labs)
    coverages = []
    for sp in species_list:
        sp_vals = vals[labs == sp]
        coverages.append(float(np.mean(sp_vals <= threshold_p95)))

    order = np.argsort(coverages)[::-1]
    sorted_species   = [species_list[i] for i in order]
    sorted_coverages = [coverages[i] for i in order]

    n = len(sorted_species)
    fig_height = max(18, n * 0.28)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = np.arange(n)
    bar_colors = [_GREENS(0.55)] * n
    ax.barh(y_pos, sorted_coverages, color=bar_colors, edgecolor="white", height=0.7)

    for p, color in _PERCENTILE_COLORS.items():
        ax.axvline(p / 100.0, color=color, linewidth=1.5, linestyle="--",
                   label=f"p{p} = {global_percentiles[f'p{p}']:.4f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_species, fontsize=12)
    ax.set_xlabel("Cobertura (fracción de query ≤ umbral p95)", fontsize=14)
    ax.set_title(
        f"Cobertura por especie bajo umbral p95 — {backbone_name}",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="x", labelsize=13)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, out_dir, "03_per_class_coverage")


def plot_per_class_dmin_boxplots(
    dmin_intra: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
    global_percentiles: dict[str, float],
    backbone_name: str,
    out_dir: Path,
) -> None:
    """Box plots horizontales de distribución de d_min intra por especie.

    Las cajas se ordenan por mediana ascendente. Líneas horizontales para
    los 4 candidatos de percentil global.

    Args:
        dmin_intra:         Array de d_min intra-clase.
        species_labels:     Etiquetas paralelas a dmin_intra.
        excluded:           Especies a excluir.
        global_percentiles: Dict {f"p{p}": valor} de calibrate_threshold.
        backbone_name:      Nombre del backbone para el título.
        out_dir:            Directorio de salida.
    """
    vals, labs = _filter_included(dmin_intra, species_labels, excluded)
    species_list = np.unique(labs)

    medians = [float(np.median(vals[labs == sp])) for sp in species_list]
    order   = np.argsort(medians)
    sorted_species = [species_list[i] for i in order]
    data_list = [vals[labs == sp] for sp in sorted_species]

    n = len(sorted_species)
    fig_height = max(18, n * 0.28)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bp = ax.boxplot(
        data_list,
        vert=False,
        patch_artist=True,
        positions=np.arange(n),
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_GREENS(0.45))
        patch.set_alpha(0.8)
    for element in ("whiskers", "caps", "medians", "fliers"):
        for item in bp[element]:
            item.set_color(_GREENS(0.85))
            if element == "medians":
                item.set_linewidth(2)

    for p, color in _PERCENTILE_COLORS.items():
        val = global_percentiles[f"p{p}"]
        ax.axvline(val, color=color, linewidth=1.5, linestyle="--",
                   label=f"p{p} = {val:.4f}")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(sorted_species, fontsize=12)
    ax.set_xlabel("Distancia coseno d_min intra-clase", fontsize=14)
    ax.set_title(
        f"Distribución de d_min por especie — {backbone_name}",
        fontsize=18, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=13)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, out_dir, "04_per_class_dmin_boxplots")


def plot_1nn_vs_centroid_comparison(
    dmin_1nn: np.ndarray,
    dmin_centroid: np.ndarray,
    species_labels: np.ndarray,
    excluded: set[str],
    pcts_1nn: dict[str, float],
    pcts_centroid: dict[str, float],
    out_dir: Path,
) -> None:
    """Comparación de las dos variantes de Umbral 2: 1-NN vs centroide.

    Panel izquierdo: histogramas superpuestos de ambas variantes con líneas
    de percentil p95 de cada una.
    Panel derecho: scatter d_min_1nn vs d_min_centroid por imagen de query
    (clases incluidas), con diagonal de referencia y correlación de Pearson.

    Args:
        dmin_1nn:      Array de d_min variante 1-NN.
        dmin_centroid: Array de d_min variante centroide.
        species_labels: Etiquetas paralelas a los arrays.
        excluded:       Especies a excluir.
        pcts_1nn:       Dict {f"p{p}": valor} de la variante 1-NN.
        pcts_centroid:  Dict {f"p{p}": valor} de la variante centroide.
        out_dir:        Directorio de salida.
    """
    vals_1nn, labs_1nn = _filter_included(dmin_1nn,      species_labels, excluded)
    vals_c,   labs_c   = _filter_included(dmin_centroid, species_labels, excluded)

    # Alinear por especie para el scatter (same order guaranteed by filter)
    fig, (ax_hist, ax_scat) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel izquierdo: histogramas ---
    ax_hist.hist(vals_1nn, bins=80, alpha=0.65, density=True,
                 color=COLOR_INTRA, label="1-NN intra")
    ax_hist.hist(vals_c, bins=80, alpha=0.65, density=True,
                 color=COLOR_INTER, label="Centroide intra")
    ax_hist.axvline(pcts_1nn["p95"], color=_GREENS(0.85), linewidth=2,
                    linestyle="--", label=f"p95 1-NN = {pcts_1nn['p95']:.4f}")
    ax_hist.axvline(pcts_centroid["p95"], color=_GREENS(0.55), linewidth=2,
                    linestyle="-.", label=f"p95 Centroide = {pcts_centroid['p95']:.4f}")
    ax_hist.set_xlabel("Distancia coseno d_min", fontsize=14)
    ax_hist.set_ylabel("Densidad", fontsize=14)
    ax_hist.set_title("Distribuciones comparadas", fontsize=16, fontweight="bold")
    ax_hist.tick_params(axis="both", labelsize=13)
    ax_hist.legend(fontsize=12)
    ax_hist.grid(alpha=0.3)

    # --- Panel derecho: scatter ---
    # Tomar el mínimo de longitud para alinear (los arrays deben tener misma longitud)
    n = min(len(vals_1nn), len(vals_c))
    rng = np.random.default_rng(SEED)
    if n > 2000:
        idx = rng.choice(n, size=2000, replace=False)
        x_plot, y_plot = vals_1nn[idx], vals_c[idx]
    else:
        x_plot, y_plot = vals_1nn[:n], vals_c[:n]

    corr = float(np.corrcoef(vals_1nn[:n], vals_c[:n])[0, 1])
    lim_max = max(vals_1nn[:n].max(), vals_c[:n].max()) * 1.05
    ax_scat.scatter(x_plot, y_plot, alpha=0.3, s=12, color=_GREENS(0.6))
    ax_scat.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="y = x")
    ax_scat.set_xlabel("d_min 1-NN", fontsize=14)
    ax_scat.set_ylabel("d_min Centroide", fontsize=14)
    ax_scat.set_title(
        f"1-NN vs Centroide  |  r = {corr:.3f}",
        fontsize=16, fontweight="bold",
    )
    ax_scat.tick_params(axis="both", labelsize=13)
    ax_scat.legend(fontsize=12)
    ax_scat.grid(alpha=0.3)
    ax_scat.set_xlim(0, lim_max)
    ax_scat.set_ylim(0, lim_max)

    fig.suptitle(
        "BioCLIP v2 — Comparación variantes Umbral 2: 1-NN vs Centroide",
        fontsize=18, fontweight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, out_dir, "05_1nn_vs_centroid_comparison")


# ===========================================================================
# SECCIÓN 5 — Main
# ===========================================================================

def main() -> None:
    """Ejecuta la calibración completa de umbrales del pipeline few-shot."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("   CALIBRACIÓN DE UMBRALES — PIPELINE FEW-SHOT FAUNA")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. SETUP
    # -----------------------------------------------------------------------
    output_data_dir = _MODULE_ROOT / "data"
    figures_dino    = FIGURES_BASE / "dinov2_small"
    figures_bio     = FIGURES_BASE / "bioclip_v2"
    for d in (output_data_dir, figures_dino, figures_bio):
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 2. CARGA DE DATOS BASE
    # -----------------------------------------------------------------------
    index_df    = load_dataset_index(DATASET_INDEX_PATH)
    support_df  = compute_class_support(index_df)
    excluded    = get_excluded_classes(support_df)

    logger.info(f"Clases incluidas: {len(support_df) - len(excluded)}")
    logger.info(f"Clases excluidas (gallery < {MIN_GALLERY_SUPPORT}): {len(excluded)}")
    for sp in sorted(excluded):
        row = support_df[support_df["species"] == sp].iloc[0]
        logger.warning(
            f"  EXCLUIDA: {sp}  "
            f"(gallery={row['gallery_count']}, query={row['query_count']})"
        )

    npz_data = np.load(NPZ_PATH)
    logger.info(f"NPZ cargado: {NPZ_PATH.name}")

    # Construir lista de excluidos para el JSON
    excluded_json = []
    for sp in sorted(excluded):
        row = support_df[support_df["species"] == sp].iloc[0]
        excluded_json.append({
            "species":       sp,
            "gallery_count": int(row["gallery_count"]),
            "query_count":   int(row["query_count"]),
            "reason":        f"gallery_count < {MIN_GALLERY_SUPPORT}",
        })

    n_query_included = int(
        index_df[
            (index_df["split"] == "query") &
            (~index_df["species"].isin(excluded))
        ].shape[0]
    )
    logger.info(f"Imágenes query incluidas: {n_query_included}")

    # -----------------------------------------------------------------------
    # 3. UMBRAL 1 — DINOv2 Small
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- UMBRAL 1: DINOv2 Small (coseno) ---")

    splits_dino = load_embeddings("dinov2_small", index_df, FEATURES_DIR)
    gallery_embs_dino,  gallery_labels_dino  = splits_dino["gallery"]
    query_embs_dino,    query_labels_dino    = splits_dino["query"]
    logger.info(
        f"  Gallery: {gallery_embs_dino.shape}  |  Query: {query_embs_dino.shape}"
    )

    logger.info("  Computando d_min intra-clase (1-NN)...")
    dmin_intra_dino, labels_dino = compute_dmin_1nn(
        query_embs_dino, query_labels_dino,
        gallery_embs_dino, gallery_labels_dino,
    )

    logger.info("  Computando d_min inter-clase...")
    dmin_inter_dino, _ = compute_dmin_inter(
        query_embs_dino, query_labels_dino,
        gallery_embs_dino, gallery_labels_dino,
    )

    umbral_1_dict = calibrate_threshold(
        dmin_intra   = dmin_intra_dino,
        dmin_inter   = dmin_inter_dino,
        species_labels = labels_dino,
        excluded     = excluded,
        backbone     = "dinov2_small",
        description  = "Filtro fauna conocida / desconocida (open-set detection)",
        n_query_included = n_query_included,
    )

    logger.info("  Percentiles globales Umbral 1:")
    for k, v in umbral_1_dict["global_percentiles"].items():
        stats = umbral_1_dict["per_percentile_stats"][k]
        logger.info(
            f"    {k}: threshold={v:.4f}  coverage={stats['coverage']:.4f}"
            f"  inter_contam={stats['inter_contamination']:.4f}"
            f"  gap={stats['separation_gap']:.4f}"
        )

    logger.info("  Generando figuras Umbral 1...")
    plot_intra_inter_distribution(
        dmin_intra_npz   = npz_data["dinov2_small_cosine_intra"],
        dmin_inter_npz   = npz_data["dinov2_small_cosine_inter"],
        global_percentiles = umbral_1_dict["global_percentiles"],
        backbone_name    = "DINOv2 Small",
        out_dir          = figures_dino,
    )
    plot_percentile_candidates_stats(
        per_percentile_stats = umbral_1_dict["per_percentile_stats"],
        backbone_name        = "DINOv2 Small",
        variant_label        = "1-NN",
        out_dir              = figures_dino,
    )
    plot_per_class_coverage(
        dmin_intra         = dmin_intra_dino,
        species_labels     = labels_dino,
        excluded           = excluded,
        global_percentiles = umbral_1_dict["global_percentiles"],
        backbone_name      = "DINOv2 Small",
        out_dir            = figures_dino,
    )
    plot_per_class_dmin_boxplots(
        dmin_intra         = dmin_intra_dino,
        species_labels     = labels_dino,
        excluded           = excluded,
        global_percentiles = umbral_1_dict["global_percentiles"],
        backbone_name      = "DINOv2 Small",
        out_dir            = figures_dino,
    )

    # -----------------------------------------------------------------------
    # 4. UMBRAL 2 — BioCLIP v2
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- UMBRAL 2: BioCLIP v2 (coseno) ---")

    splits_bio = load_embeddings("bioclip_v2", index_df, FEATURES_DIR)
    gallery_embs_bio,  gallery_labels_bio  = splits_bio["gallery"]
    query_embs_bio,    query_labels_bio    = splits_bio["query"]
    logger.info(
        f"  Gallery: {gallery_embs_bio.shape}  |  Query: {query_embs_bio.shape}"
    )

    logger.info("  Computando d_min intra-clase variante 1-NN...")
    dmin_intra_1nn, labels_bio = compute_dmin_1nn(
        query_embs_bio, query_labels_bio,
        gallery_embs_bio, gallery_labels_bio,
    )

    logger.info("  Computando d_min inter-clase...")
    dmin_inter_bio, _ = compute_dmin_inter(
        query_embs_bio, query_labels_bio,
        gallery_embs_bio, gallery_labels_bio,
    )

    logger.info("  Computando centroides de gallery...")
    centroids = compute_centroids(gallery_embs_bio, gallery_labels_bio)
    logger.info(f"  Centroides calculados para {len(centroids)} especies.")

    logger.info("  Computando d_min intra-clase variante centroide...")
    dmin_intra_centroid, _ = compute_dmin_centroid(
        query_embs_bio, query_labels_bio, centroids,
    )

    # Calibrar variante 1-NN
    calib_1nn = calibrate_threshold(
        dmin_intra     = dmin_intra_1nn,
        dmin_inter     = dmin_inter_bio,
        species_labels = labels_bio,
        excluded       = excluded,
        backbone       = "bioclip_v2",
        description    = "Confianza de predicción taxonómica — variante 1-NN",
        n_query_included = n_query_included,
    )

    # Calibrar variante centroide
    calib_centroid = calibrate_threshold(
        dmin_intra     = dmin_intra_centroid,
        dmin_inter     = dmin_inter_bio,
        species_labels = labels_bio,
        excluded       = excluded,
        backbone       = "bioclip_v2",
        description    = "Confianza de predicción taxonómica — variante centroide",
        n_query_included = n_query_included,
    )

    logger.info("  Percentiles globales Umbral 2 — variante 1-NN:")
    for k, v in calib_1nn["global_percentiles"].items():
        stats = calib_1nn["per_percentile_stats"][k]
        logger.info(
            f"    {k}: threshold={v:.4f}  coverage={stats['coverage']:.4f}"
            f"  inter_contam={stats['inter_contamination']:.4f}"
            f"  gap={stats['separation_gap']:.4f}"
        )
    logger.info("  Percentiles globales Umbral 2 — variante centroide:")
    for k, v in calib_centroid["global_percentiles"].items():
        stats = calib_centroid["per_percentile_stats"][k]
        logger.info(
            f"    {k}: threshold={v:.4f}  coverage={stats['coverage']:.4f}"
            f"  inter_contam={stats['inter_contamination']:.4f}"
            f"  gap={stats['separation_gap']:.4f}"
        )

    # Ensamblar umbral_2 con estructura simétrica + variants anidadas
    umbral_2_dict: dict = {
        "backbone":    "bioclip_v2",
        "metric":      "cosine",
        "description": "Confianza de predicción taxonómica",
        "n_included_classes":    calib_1nn["n_included_classes"],
        "n_query_included":      n_query_included,
        "variants": {
            "nearest_neighbor": {
                "description":         calib_1nn["description"],
                "global_percentiles":  calib_1nn["global_percentiles"],
                "per_percentile_stats": calib_1nn["per_percentile_stats"],
            },
            "centroid": {
                "description":         calib_centroid["description"],
                "global_percentiles":  calib_centroid["global_percentiles"],
                "per_percentile_stats": calib_centroid["per_percentile_stats"],
            },
        },
        "per_class_percentiles":  calib_1nn["per_class_percentiles"],
        "recommended_percentile": None,
        "recommended_threshold":  None,
    }

    logger.info("  Generando figuras Umbral 2...")
    plot_intra_inter_distribution(
        dmin_intra_npz   = npz_data["bioclip_v2_cosine_intra"],
        dmin_inter_npz   = npz_data["bioclip_v2_cosine_inter"],
        global_percentiles = calib_1nn["global_percentiles"],
        backbone_name    = "BioCLIP v2",
        out_dir          = figures_bio,
    )
    plot_percentile_candidates_stats(
        per_percentile_stats = calib_1nn["per_percentile_stats"],
        backbone_name        = "BioCLIP v2",
        variant_label        = "1-NN",
        out_dir              = figures_bio,
        fig_stem             = "02_percentile_candidates_stats_1nn",
    )
    plot_percentile_candidates_stats(
        per_percentile_stats = calib_centroid["per_percentile_stats"],
        backbone_name        = "BioCLIP v2",
        variant_label        = "Centroide",
        out_dir              = figures_bio,
        fig_stem             = "02_percentile_candidates_stats_centroid",
    )
    plot_per_class_coverage(
        dmin_intra         = dmin_intra_1nn,
        species_labels     = labels_bio,
        excluded           = excluded,
        global_percentiles = calib_1nn["global_percentiles"],
        backbone_name      = "BioCLIP v2",
        out_dir            = figures_bio,
    )
    plot_per_class_dmin_boxplots(
        dmin_intra         = dmin_intra_1nn,
        species_labels     = labels_bio,
        excluded           = excluded,
        global_percentiles = calib_1nn["global_percentiles"],
        backbone_name      = "BioCLIP v2",
        out_dir            = figures_bio,
    )
    plot_1nn_vs_centroid_comparison(
        dmin_1nn       = dmin_intra_1nn,
        dmin_centroid  = dmin_intra_centroid,
        species_labels = labels_bio,
        excluded       = excluded,
        pcts_1nn       = calib_1nn["global_percentiles"],
        pcts_centroid  = calib_centroid["global_percentiles"],
        out_dir        = figures_bio,
    )

    # -----------------------------------------------------------------------
    # 5. GUARDAR thresholds.json
    # -----------------------------------------------------------------------
    thresholds_out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "min_gallery_support":   MIN_GALLERY_SUPPORT,
            "percentile_candidates": PERCENTILES,
            "metric":                "cosine",
            "seed":                  SEED,
        },
        "excluded_classes": excluded_json,
        "umbral_1":         umbral_1_dict,
        "umbral_2":         umbral_2_dict,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds_out, f, indent=2, ensure_ascii=False)
    logger.info(f"\nthresholds.json guardado: {OUTPUT_JSON}")

    # -----------------------------------------------------------------------
    # 6. RESUMEN FINAL
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=== RESUMEN DE CALIBRACIÓN ===")
    header = f"{'Umbral':<12} {'Variante':<18} {'Percentil':>10} {'Threshold':>12} {'Coverage':>10} {'Inter%':>10} {'Gap':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for p in PERCENTILES:
        key   = f"p{p}"
        stats = umbral_1_dict["per_percentile_stats"][key]
        logger.info(
            f"{'Umbral 1':<12} {'1-NN':<18} {key:>10} "
            f"{stats['threshold_value']:>12.4f} "
            f"{stats['coverage']:>10.4f} "
            f"{stats['inter_contamination']:>10.4f} "
            f"{stats['separation_gap']:>10.4f}"
        )

    logger.info("")
    for variant_key, variant_label in [("nearest_neighbor", "1-NN"), ("centroid", "Centroide")]:
        for p in PERCENTILES:
            key   = f"p{p}"
            stats = umbral_2_dict["variants"][variant_key]["per_percentile_stats"][key]
            logger.info(
                f"{'Umbral 2':<12} {variant_label:<18} {key:>10} "
                f"{stats['threshold_value']:>12.4f} "
                f"{stats['coverage']:>10.4f} "
                f"{stats['inter_contamination']:>10.4f} "
                f"{stats['separation_gap']:>10.4f}"
            )

    logger.info("")
    logger.info("CALIBRACIÓN COMPLETADA")


if __name__ == "__main__":
    main()
