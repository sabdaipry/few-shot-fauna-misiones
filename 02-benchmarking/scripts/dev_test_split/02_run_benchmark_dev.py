"""Corre el benchmark de clasificación few-shot sobre el subconjunto query_dev
(re-validación metodológica: separa las decisiones de diseño del reporte final).

Diferencias respecto a scripts/03_run_benchmark.py (que sigue intacto y usa el
query set completo):
    - Índice: dataset_index_dev_test_split.csv en vez de dataset_index.csv.
    - Test set: solo filas con split == 'query_dev' (query_test queda fuera
      por completo, ni se carga).
    - Backbones: los 15 estándar (se excluyen las 4 variantes DINO _gap,
      que fueron un experimento diagnóstico aparte).
    - Clasificadores: los 7 "reales" (Nearest Centroid, KNN k=1/3/5, Linear
      SVM, RBF SVM, Random Forest). Se excluyen las 4 variantes Faiss —
      fueron una prueba de latencia aparte, no un método de clasificación
      distinto, y no deben mezclarse en la tabla de accuracy.

Salidas (carpeta nueva, no pisa nada de data/benchmark_results/):
    data/benchmark_results_dev/benchmark_summary_dev.csv
    data/benchmark_results_dev/predictions_<model>_dev.csv

Uso:
    python scripts/dev_test_split/02_run_benchmark_dev.py
"""
import sys
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/dev_test_split/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent               # 02-benchmarking/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmarking import ModelEvaluator
from src.utils.logger import setup_logger
from src.config import MODELS_TO_TEST, FEATURES_DIR

logger = setup_logger("benchmarking-runner-dev", log_dir=_PROJECT_ROOT / "logs")

DATASET_INDEX_DEV_PATH = _PROJECT_ROOT / "data" / "dev_test_split" / "dataset_index_dev_test_split.csv"
BENCHMARK_RESULTS_DIR_DEV = _PROJECT_ROOT / "data" / "benchmark_results_dev"

# 15 backbones estándar: se excluyen las 4 variantes _gap (experimento
# diagnóstico aparte, no forma parte del benchmark principal).
STANDARD_BACKBONES = [m for m in MODELS_TO_TEST if not m.endswith("_gap")]

# 7 clasificadores "reales": se excluyen las 4 variantes Faiss (prueba de
# latencia aparte, resultados de accuracy idénticos a sus contrapartes
# sklearn en este volumen de datos).
CLASSIFIERS_7 = [
    "Nearest Centroid",
    "KNN (k=1)",
    "KNN (k=3)",
    "KNN (k=5)",
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
]


def main():
    logger.info("")
    logger.info("==============================================")
    logger.info("   BENCHMARK SOBRE query_dev (re-validación metodológica)")
    logger.info("==============================================")
    logger.info(f"Backbones ({len(STANDARD_BACKBONES)}): {STANDARD_BACKBONES}")
    logger.info(f"Clasificadores ({len(CLASSIFIERS_7)}): {CLASSIFIERS_7}")

    if not DATASET_INDEX_DEV_PATH.exists():
        logger.error(
            f"No existe {DATASET_INDEX_DEV_PATH}. "
            "Correr antes scripts/dev_test_split/01_generate_dev_test_index.py"
        )
        return

    evaluator = ModelEvaluator(
        DATASET_INDEX_DEV_PATH,
        FEATURES_DIR,
        output_dir=BENCHMARK_RESULTS_DIR_DEV,
        test_split_value="query_dev",
    )

    outer_pbar = tqdm(STANDARD_BACKBONES, desc="Progreso Total", unit="model")
    for model in outer_pbar:
        outer_pbar.set_description(f"Evaluando Modelo: {model}")
        evaluator.evaluate_model(model, classifier_names=CLASSIFIERS_7)

    # Renombrar salidas para que queden claramente identificadas como "_dev"
    # (ModelEvaluator siempre escribe benchmark_summary.csv / predictions_<m>.csv
    # dentro del output_dir que se le pase; acá ese output_dir ya es una
    # carpeta nueva y separada, así que renombramos los archivos finales
    # dentro de ella para que el sufijo _dev sea explícito en el nombre).
    summary_path = BENCHMARK_RESULTS_DIR_DEV / "benchmark_summary.csv"
    if summary_path.exists():
        summary_path.rename(BENCHMARK_RESULTS_DIR_DEV / "benchmark_summary_dev.csv")

    for pred_path in BENCHMARK_RESULTS_DIR_DEV.glob("predictions_*.csv"):
        if pred_path.stem.endswith("_dev"):
            continue
        new_name = pred_path.with_name(pred_path.stem + "_dev.csv")
        pred_path.rename(new_name)

    logger.info("=" * 50)
    logger.info("BENCHMARK SOBRE query_dev FINALIZADO")
    logger.info("=" * 50)
    logger.info(f"Resultados guardados en: {BENCHMARK_RESULTS_DIR_DEV}")


if __name__ == "__main__":
    main()
