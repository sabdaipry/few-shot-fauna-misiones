"""Benchmark de métricas de distancia para clasificación 1-NN few-shot.

Evalúa 5 métricas de distancia (coseno, euclidiana, euclidiana L2-norm,
Manhattan, Mahalanobis) sobre los embeddings de gallery/query de bioclip_v2 y
dinov2_small. Para cada combinación backbone × métrica mide accuracy 1-NN y
latencia de búsqueda por imagen.

Guarda:
- data/benchmark_results/distance_benchmark.csv
- data/benchmark_results/distance_distributions.npz  (intra/inter para TODAS las métricas)
- figures/distance_benchmark_accuracy.{svg,png}
- figures/distance_benchmark_latency.{svg,png}
- figures/distance_distributions_{backbone}.{svg,png}  (una figura 2×2 por backbone)

Uso:
    python scripts/08_distance_benchmark.py
"""

import sys
import time
import logging
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

# Ajuste de rutas para imports desde la raíz del repo
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("distance_benchmark", log_dir=_PROJECT_ROOT / "logs")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATASET_INDEX_PATH = _PROJECT_ROOT / "data" / "dataset_index.csv"
FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
BENCHMARK_RESULTS_DIR = _PROJECT_ROOT / "data" / "benchmark_results"
FIGURES_DIR = _PROJECT_ROOT / "data" / "reports" / "figures"
FIGURES_DIR_DIST = _PROJECT_ROOT / "data" / "reports" / "figures" / "distance_distributions"

BACKBONES: dict[str, int] = {
    "bioclip_v2":      768,
    "dinov3_small":    384,
    "dinov2_small":    384,
    "resnet50":        2048,
    "convnextv2_tiny": 768,
}

METRICS: list[str] = [
    "cosine",
    "euclidean",
    "euclidean_l2norm",
    "manhattan",
]

WARMUP_ITERS = 10

# Paleta fija para los plots
_GREENS = cm.get_cmap("Greens")
_METRIC_COLORS = {
    "cosine":           _GREENS(0.9),
    "euclidean":        _GREENS(0.7),
    "euclidean_l2norm": _GREENS(0.55),
    "manhattan":        _GREENS(0.4),
}
_BACKBONE_COLORS = {
    "bioclip_v2":      _GREENS(0.9),
    "dinov3_small":    _GREENS(0.75),
    "dinov2_small":    _GREENS(0.6),
    "resnet50":        _GREENS(0.45),
    "convnextv2_tiny": _GREENS(0.3),
}


# ---------------------------------------------------------------------------
# Carga de embeddings
# ---------------------------------------------------------------------------

def load_embeddings(
    backbone: str,
    index_df: pd.DataFrame,
    features_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Carga los embeddings .npy para un backbone y retorna gallery y query.

    Args:
        backbone: Nombre del backbone (subcarpeta en features_dir).
        index_df: DataFrame con columnas filepath, species, genus, family, split.
        features_dir: Ruta raíz de la carpeta de features.

    Returns:
        Dict con claves 'gallery' y 'query', cada una con (embeddings, labels)
        donde embeddings es float32 de shape (N, D) y labels es array de strings.
    """
    result: dict[str, tuple[list, list]] = {"gallery": ([], []), "query": ([], [])}
    backbone_dir = features_dir / backbone

    missing = 0
    for _, row in index_df.iterrows():
        species_folder = row["species"].replace(" ", "_")
        stem = Path(row["filepath"]).stem
        npy_path = backbone_dir / row["family"] / row["genus"] / species_folder / f"{stem}.npy"

        if not npy_path.exists():
            logger.warning(f"No encontrado: {npy_path}")
            missing += 1
            continue

        emb = np.load(npy_path).astype(np.float32).ravel()
        split = row["split"]
        result[split][0].append(emb)
        result[split][1].append(row["species"])

    if missing:
        logger.warning(f"[{backbone}] {missing} archivos .npy no encontrados y omitidos.")

    return {
        split: (np.vstack(embs), np.array(labels))
        for split, (embs, labels) in result.items()
        if embs
    }


# ---------------------------------------------------------------------------
# Funciones de distancia  →  todas retornan matrix (n_query, n_gallery)
# ---------------------------------------------------------------------------

def dist_cosine(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Distancia coseno: 1 − similitud coseno, vectorizada."""
    q_norm = normalize(query, norm="l2")
    g_norm = normalize(gallery, norm="l2")
    return 1.0 - (q_norm @ g_norm.T)


def dist_euclidean(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Distancia euclidiana estándar."""
    return cdist(query, gallery, metric="euclidean")


def dist_euclidean_l2norm(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Distancia euclidiana sobre embeddings L2-normalizados."""
    return cdist(normalize(query, norm="l2"), normalize(gallery, norm="l2"), metric="euclidean")


def dist_manhattan(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Distancia Manhattan (L1)."""
    return cdist(query, gallery, metric="cityblock")


# ---------------------------------------------------------------------------
# Evaluación 1-NN
# ---------------------------------------------------------------------------

def predict_1nn(dist_matrix: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """Retorna la etiqueta del vecino más cercano para cada fila de dist_matrix."""
    nearest_idx = np.argmin(dist_matrix, axis=1)
    return gallery_labels[nearest_idx]


def measure_latency(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    dist_fn: Callable,
    warmup: int = WARMUP_ITERS,
    n_samples: int = 200,
    n_repeats: int = 10,
    seed: int = 29,
) -> dict[str, float]:
    """Mide latencia de búsqueda con muestra estratificada y repeticiones.

    Selecciona n_samples imágenes del query de forma aleatoria (seed fija),
    las mide n_repeats veces cada una, y retorna media, mediana y p95.

    Args:
        query_embs: Embeddings query, shape (N, D).
        gallery_embs: Embeddings gallery, shape (M, D).
        dist_fn: Función dist(query_1xD, gallery_MxD).
        warmup: Iteraciones de calentamiento no medidas.
        n_samples: Cantidad de imágenes a muestrear para medir latencia.
        n_repeats: Repeticiones por imagen.
        seed: Semilla para reproducibilidad.

    Returns:
        Dict con claves 'mean_ms', 'median_ms', 'p95_ms'.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(query_embs), size=min(n_samples, len(query_embs)), replace=False)
    sample = query_embs[idx]

    for _ in range(warmup):
        dist_fn(query_embs[:1], gallery_embs)

    times_ms = []
    for q in sample:
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            dist_fn(q[np.newaxis], gallery_embs)
            times_ms.append((time.perf_counter() - t0) * 1000)

    times_arr = np.array(times_ms)
    return {
        "mean_ms":   float(np.mean(times_arr)),
        "median_ms": float(np.median(times_arr)),
        "p95_ms":    float(np.percentile(times_arr, 95)),
    }


def run_1nn_evaluation(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    dist_fn: Callable,
) -> tuple[float, dict[str, float]]:
    """Calcula accuracy 1-NN y métricas de latencia de búsqueda por imagen.

    Args:
        query_embs: Embeddings query.
        query_labels: Etiquetas de especie para query.
        gallery_embs: Embeddings gallery.
        gallery_labels: Etiquetas de especie para gallery.
        dist_fn: Función de distancia.

    Returns:
        (accuracy, latency_dict) con claves 'mean_ms', 'median_ms', 'p95_ms'.
    """
    dist_matrix = dist_fn(query_embs, gallery_embs)
    preds = predict_1nn(dist_matrix, gallery_labels)
    accuracy = float(np.mean(preds == query_labels))
    latency_dict = measure_latency(query_embs, gallery_embs, dist_fn)
    return accuracy, latency_dict


# ---------------------------------------------------------------------------
# Distribuciones intra / inter clase
# ---------------------------------------------------------------------------

def compute_distance_distributions(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    dist_fn: Callable,
) -> tuple[np.ndarray, np.ndarray]:
    """Computa distribuciones intra e inter clase para una métrica dada.

    Intra: distancia de cada imagen query al vecino más cercano de la MISMA
    especie en gallery.
    Inter: distancia de cada imagen query al vecino más cercano de UNA ESPECIE
    DIFERENTE en gallery.

    Args:
        query_embs: Embeddings query, shape (N, D).
        query_labels: Etiquetas de especie para query.
        gallery_embs: Embeddings gallery, shape (M, D).
        gallery_labels: Etiquetas de especie para gallery.
        dist_fn: Función de distancia.

    Returns:
        (intra_distances, inter_distances) — arrays 1D de longitud N.
    """
    dist_matrix = dist_fn(query_embs, gallery_embs)

    intra = np.empty(len(query_embs))
    inter = np.empty(len(query_embs))

    for i, label in enumerate(query_labels):
        same_mask = gallery_labels == label
        diff_mask = ~same_mask

        row = dist_matrix[i]

        if same_mask.any():
            intra[i] = row[same_mask].min()
        else:
            intra[i] = np.nan

        if diff_mask.any():
            inter[i] = row[diff_mask].min()
        else:
            inter[i] = np.nan

    return intra, inter


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_accuracy_bar(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Gráfico de barras agrupadas backbone × métrica mostrando accuracy."""
    backbones = results_df["backbone"].unique()
    metrics = METRICS
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, backbone in enumerate(backbones):
        sub = results_df[results_df["backbone"] == backbone].set_index("metric")
        vals = [sub.loc[m, "accuracy"] if m in sub.index else 0.0 for m in metrics]
        offset = (i - (len(backbones) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=backbone,
                      color=_BACKBONE_COLORS.get(backbone))
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003 + v * 0.003,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontsize=16)
    ax.set_ylabel("Accuracy (1-NN)", fontsize=16)
    ax.set_title("Accuracy 1-NN por backbone y métrica de distancia",
                 fontsize=18, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"distance_benchmark_accuracy.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Figura guardada: {out_dir}/distance_benchmark_accuracy.{{svg,png}}")


def plot_latency_bar(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Gráfico de barras agrupadas backbone × métrica mostrando latencia (escala log)."""
    backbones = results_df["backbone"].unique()
    metrics = METRICS
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    all_vals: list[float] = []
    bar_info: list[tuple] = []

    for i, backbone in enumerate(backbones):
        sub = results_df[results_df["backbone"] == backbone].set_index("metric")
        vals = [sub.loc[m, "median_ms"] if m in sub.index else np.nan for m in metrics]
        offset = (i - (len(backbones) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=backbone,
                      color=_BACKBONE_COLORS.get(backbone))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                all_vals.append(v)
                bar_info.append((bar, v))

    ax.set_yscale("log")
    max_val = max(all_vals) if all_vals else 1.0
    ax.set_ylim(top=max_val * 3)

    for bar, v in bar_info:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.05 + 0.003,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            rotation=90,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontsize=16)
    ax.set_ylabel("Latencia mediana por imagen (ms, escala log)", fontsize=16)
    ax.set_title("Latencia de búsqueda 1-NN por backbone y métrica",
                 fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()

    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"distance_benchmark_latency.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Figura guardada: {out_dir}/distance_benchmark_latency.{{svg,png}}")


def plot_distance_distributions(
    distributions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    results_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Genera una figura 2×2 por backbone con histogramas intra vs inter-clase
    para cada métrica. Una figura por backbone, guardada como
    distance_distributions_{backbone}.{svg,png}.

    Args:
        distributions: {backbone: {metric: (intra, inter)}}.
        results_df: DataFrame con columnas backbone, metric, accuracy.
        out_dir: Directorio de salida para las figuras.
    """
    COLOR_INTRA = "#2d6a4f"  # verde oscuro
    COLOR_INTER = "#95d5b2"  # verde claro

    for backbone, metric_dists in distributions.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, METRICS):
            if metric not in metric_dists:
                ax.set_visible(False)
                continue

            intra, inter = metric_dists[metric]
            intra_clean = intra[~np.isnan(intra)]
            inter_clean = inter[~np.isnan(inter)]

            row = results_df[
                (results_df["backbone"] == backbone) & (results_df["metric"] == metric)
            ]
            acc = row.iloc[0]["accuracy"] if not row.empty else float("nan")

            ax.hist(intra_clean, bins=60, alpha=0.6, label="Intra-clase",
                    color=COLOR_INTRA, density=True)
            ax.hist(inter_clean, bins=60, alpha=0.6, label="Inter-clase",
                    color=COLOR_INTER, density=True)
            ax.set_xlabel("Distancia", fontsize=14)
            ax.set_ylabel("Densidad", fontsize=14)
            ax.set_title(f"{metric}  |  Acc={acc:.3f}", fontweight="bold", fontsize=16)
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(alpha=0.3)

        for ax in axes_flat[len(METRICS):]:
            ax.set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=13,
                   bbox_to_anchor=(1.0, 0.98))

        fig.suptitle(
            f"Distribuciones de distancia intra vs inter-clase — {backbone}",
            fontsize=18, fontweight="bold",
        )
        fig.tight_layout()

        fname = backbone.replace("/", "_").replace(" ", "_")
        for ext in ("svg", "png"):
            fig.savefig(
                out_dir / f"distance_distributions_{fname}.{ext}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)
        logger.info(
            f"Figura guardada: {out_dir}/distance_distributions_{fname}.{{svg,png}}"
        )


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta el benchmark completo de métricas de distancia."""
    logger.info("")
    logger.info("==============================================")
    logger.info("   FASE 8: BENCHMARK DE MÉTRICAS DE DISTANCIA")
    logger.info("==============================================")

    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR_DIST.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(DATASET_INDEX_PATH)
    logger.info(
        f"Índice cargado: {len(index_df)} registros "
        f"({(index_df['split']=='gallery').sum()} gallery, "
        f"{(index_df['split']=='query').sum()} query)"
    )

    records: list[dict] = []
    # {backbone: {metric: (intra, inter)}}
    all_distributions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for backbone, dim in BACKBONES.items():
        logger.info("")
        logger.info(f"--- Backbone: {backbone} (dim={dim}) ---")

        splits = load_embeddings(backbone, index_df, FEATURES_DIR)
        gallery_embs, gallery_labels = splits["gallery"]
        query_embs, query_labels = splits["query"]
        logger.info(
            f"  Gallery: {gallery_embs.shape}  |  Query: {query_embs.shape}"
        )

        dist_fns: dict[str, Callable] = {
            "cosine": dist_cosine,
            "euclidean": dist_euclidean,
            "euclidean_l2norm": dist_euclidean_l2norm,
            "manhattan": dist_manhattan,
        }

        all_distributions[backbone] = {}

        for metric in METRICS:
            logger.info(f"  Evaluando métrica: {metric}")
            fn = dist_fns[metric]

            accuracy, latency_dict = run_1nn_evaluation(
                query_embs, query_labels, gallery_embs, gallery_labels, fn
            )

            logger.info(
                f"    Accuracy: {accuracy:.4f}  |  "
                f"mean={latency_dict['mean_ms']:.3f}ms  "
                f"median={latency_dict['median_ms']:.3f}ms  "
                f"p95={latency_dict['p95_ms']:.3f}ms"
            )

            logger.info(f"    Calculando distribuciones intra/inter-clase para {metric}...")
            intra, inter = compute_distance_distributions(
                query_embs, query_labels, gallery_embs, gallery_labels, fn
            )
            all_distributions[backbone][metric] = (intra, inter)

            records.append(
                {
                    "backbone":     backbone,
                    "backbone_dim": dim,
                    "metric":       metric,
                    "accuracy":     accuracy,
                    "mean_ms":      latency_dict["mean_ms"],
                    "median_ms":    latency_dict["median_ms"],
                    "p95_ms":       latency_dict["p95_ms"],
                }
            )

    # Marcar is_best por backbone
    results_df = pd.DataFrame(records)
    results_df["is_best"] = False
    for backbone in results_df["backbone"].unique():
        mask = results_df["backbone"] == backbone
        best_idx = results_df.loc[mask, "accuracy"].idxmax()
        results_df.loc[best_idx, "is_best"] = True

    # Guardar CSV
    csv_path = BENCHMARK_RESULTS_DIR / "distance_benchmark.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nCSV guardado: {csv_path}")

    # Guardar distribuciones .npz (TODAS las métricas)
    npz_arrays: dict[str, np.ndarray] = {}
    for backbone, metric_dists in all_distributions.items():
        for metric, (intra, inter) in metric_dists.items():
            npz_arrays[f"{backbone}_{metric}_intra"] = intra
            npz_arrays[f"{backbone}_{metric}_inter"] = inter
    npz_path = BENCHMARK_RESULTS_DIR / "distance_distributions.npz"
    np.savez(npz_path, **npz_arrays)
    logger.info(f"Distribuciones guardadas: {npz_path} ({len(npz_arrays)} arrays)")

    # Figuras
    plot_accuracy_bar(results_df, FIGURES_DIR)
    plot_latency_bar(results_df, FIGURES_DIR)
    plot_distance_distributions(all_distributions, results_df, FIGURES_DIR_DIST)

    # Resumen en consola
    logger.info("")
    logger.info("=== RESUMEN — ordenado por accuracy descendente ===")
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
    logger.info("FASE 8 COMPLETADA")


if __name__ == "__main__":
    main()
