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

logger = setup_logger("benchmarking-runner")

def main():
    print("\n")
    logger.info("==============================================")
    logger.info("   FASE 3: BENCHMARKING DE CLASIFICADORES")
    logger.info("==============================================")

    # Configuración
    INDEX_PATH = "data/dataset_index.csv"
    FEATURES_DIR = "data/features"
    OUTPUT_DIR = "data/benchmark_results" # Nueva carpeta para resultados ordenados

    # LISTA COMPLETA DE TUS MODELOS (Verificados)
    models_to_test = [
        "resnet50",
        "convnextv2_tiny", "convnextv2_base",
        "dinov2_small", "dinov2_base",
        "dinov2_small_gap", "dinov2_base_gap",
        "dinov3_small", "dinov3_base",
        "dinov3_small_gap", "dinov3_base_gap",
        "siglip_base", "siglip_so400m",
        "siglip2_base", "siglip2_so400m",
        "bioclip_v1", "bioclip_v2",
        "clip_base", "clip_large"
    ]

    # Instanciar el evaluador
    evaluator = ModelEvaluator(INDEX_PATH, FEATURES_DIR, OUTPUT_DIR)

    # Barra Maestra (Modelos)
    # position=0 asegura que esta barra se quede arriba
    outer_pbar = tqdm(models_to_test, desc="Progreso Total", unit="model")

    # Correr evaluación
    for model in outer_pbar:
        outer_pbar.set_description(f"Evaluando Modelo: {model}")
        evaluator.evaluate_model(model)

    logger.info("="*50)
    logger.info("BENCHMARKING FINALIZADO")
    logger.info("="*50)
    logger.info(f"Resultados guardados en: {OUTPUT_DIR}")
    logger.info("   - benchmark_summary.csv (Tabla general)")
    logger.info("   - predictions_<model_name>.csv (Detalle por imagen para gráficos)")
    
if __name__ == "__main__":
    main()