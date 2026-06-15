import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.benchmarking import ModelEvaluator
from src.utils.logger import setup_logger
from src.config import MODELS_TO_TEST, DATASET_INDEX_PATH, FEATURES_DIR, BENCHMARK_RESULTS_DIR

logger = setup_logger("benchmarking-runner")

def main():
    print("\n")
    logger.info("==============================================")
    logger.info("   FASE 3: BENCHMARKING DE CLASIFICADORES")
    logger.info("==============================================")

    # Instanciar el evaluador
    evaluator = ModelEvaluator(DATASET_INDEX_PATH, FEATURES_DIR, BENCHMARK_RESULTS_DIR)

    # Barra Maestra (Modelos)
    # position=0 asegura que esta barra se quede arriba
    outer_pbar = tqdm(MODELS_TO_TEST, desc="Progreso Total", unit="model")

    # Correr evaluación
    for model in outer_pbar:
        outer_pbar.set_description(f"Evaluando Modelo: {model}")
        evaluator.evaluate_model(model)

    logger.info("="*50)
    logger.info("BENCHMARKING FINALIZADO")
    logger.info("="*50)
    logger.info(f"Resultados guardados en: {BENCHMARK_RESULTS_DIR}")
    logger.info("   - benchmark_summary.csv (Tabla general)")
    logger.info("   - predictions_<model_name>.csv (Detalle por imagen para gráficos)")
    
if __name__ == "__main__":
    main()