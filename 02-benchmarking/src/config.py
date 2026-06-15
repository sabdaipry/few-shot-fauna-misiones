"""
Configuración centralizada del benchmark.

Este módulo define PROJECT_ROOT, la lista canónica de modelos a evaluar
y las constantes de paths derivadas, para que todos los scripts de
02-benchmarking/scripts/ las importen en lugar de duplicarlas.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
MODELS_TO_TEST = [
    "resnet50",
    "convnextv2_tiny", "convnextv2_base",
    "dinov2_small", "dinov2_base",
    "dinov2_small_gap", "dinov2_base_gap",
    "dinov3_small", "dinov3_base",
    "dinov3_small_gap", "dinov3_base_gap",
    "siglip_base", "siglip_so400m",
    "siglip2_base", "siglip2_so400m",
    "bioclip_v1", "bioclip_v2",
    "clip_base", "clip_large",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

DATASET_INDEX_PATH = DATA_DIR / "dataset_index.csv"
FEATURES_DIR = DATA_DIR / "features"
BENCHMARK_RESULTS_DIR = DATA_DIR / "benchmark_results"
REPORTS_DIR = DATA_DIR / "reports"

BACKBONES_TIMES_PATH = DATA_DIR / "backbones_times.csv"
SCALABILITY_RESULTS_PATH = DATA_DIR / "scalability_results.csv"
INCREMENTAL_RESULTS_PATH = DATA_DIR / "incremental_results.csv"
OUTLIER_RESULTS_PATH = DATA_DIR / "outlier_results.csv"
