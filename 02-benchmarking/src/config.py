"""
Configuración centralizada del benchmark.

Este módulo define PROJECT_ROOT, la lista canónica de modelos a evaluar
y las constantes de paths derivadas, para que todos los scripts de
02-benchmarking/scripts/ las importen en lugar de duplicarlas.
"""
from pathlib import Path
from src.backbones import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Modelos — el orden está definido en MODEL_REGISTRY (src/backbones.py)
# ---------------------------------------------------------------------------
MODELS_TO_TEST = list(MODEL_REGISTRY.keys())

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
