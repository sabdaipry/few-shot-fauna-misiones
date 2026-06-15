"""Extrae embeddings de imágenes usando los backbones configurados y los
guarda como archivos .npy replicando la estructura de directorios del dataset."""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch

# --- CONFIGURACIÓN DE RUTAS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

try:
    from src.utils.logger import setup_logger
    from src.backbones import create_extractor
    from src.config import MODELS_TO_TEST
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

# Configuración
DATASET_INDEX = project_root / "data/dataset_index.csv"
FEATURES_DIR = project_root / "data/features"

logger = setup_logger("feature_extractor", log_dir=project_root / "logs")

def extract_features(model_name):
    """Extrae y guarda embeddings de todas las imágenes del índice para el modelo dado."""
    # 1. Cargar índice
    if not DATASET_INDEX.exists():
        logger.error("No se encontró dataset_index.csv. Corre el script 01 primero.")
        return

    df = pd.read_csv(DATASET_INDEX)
    total_images = len(df)
    
    # 2. Preparar Modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"-> Iniciando extracción con {model_name} en {device}...")
    
    try:
        extractor = create_extractor(model_name, device)
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return

    # 3. Preparar directorio de salida
    # Estructura: data/features/nombre_modelo/
    output_base_dir = FEATURES_DIR / extractor.name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"-> Guardando embeddings en: {output_base_dir}")

    # 4. Loop de extracción
    success_count = 0
    skip_count = 0
    error_count = 0

    # Usamos tqdm para la barra de progreso
    pbar = tqdm(total=total_images, desc="Procesando", unit="img")

    for _, row in df.iterrows():
        # Ruta original de la imagen
        # NOTA: row['filepath'] es relativo o absoluto según cómo se generó.
        # Asumimos que es relativo a project_root o absoluto. 
        # Lo más seguro es construirlo relativo al root si no existe directo.
        img_path = Path(row['filepath'])
        if not img_path.exists():
            # Intentar buscarlo relativo al root si falló
            img_path = project_root / row['filepath']
        
        if not img_path.exists():
            error_count += 1
            pbar.update(1)
            continue

        # Definir ruta de salida manteniendo estructura
        # Ejemplo: data/features/resnet50/Felidae/Panthera/onca/img01.npy
        
        # Truco: obtener la parte relativa desde la carpeta 'images' para replicar estructura
        try:
            # Buscamos 'images' en el path para cortar desde ahí
            parts = img_path.parts
            if 'images' in parts:
                idx = parts.index('images')
                rel_path = Path(*parts[idx+1:]) # Todo lo que sigue a images
            else:
                # Si no hay carpeta images, usamos el nombre de archivo plano (menos ideal)
                rel_path = Path(img_path.name)
        except ValueError:
            rel_path = Path(img_path.name)

        save_path = output_base_dir / rel_path.with_suffix('.npy')
        
        # Crear subdirectorios si no existen
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # CHECK DE CACHÉ: Si ya existe, saltar
        if save_path.exists():
            skip_count += 1
            pbar.update(1)
            continue

        # Inferencia
        embedding = extractor.get_embedding(str(img_path))

        if embedding is not None:
            np.save(save_path, embedding)
            success_count += 1
        else:
            error_count += 1
        
        pbar.update(1)

    pbar.close()
    
    logger.info("="*50)
    logger.info(f" Extracción finalizada para {model_name}".upper())
    logger.info("="*50)
    logger.info(f" Procesados: {success_count}")
    logger.info(f" Saltados (Ya existían): {skip_count}")
    logger.info(f" Errores: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extractor de Características")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODELS_TO_TEST,
                        help="Modelo a utilizar")
    
    args = parser.parse_args()
    
    extract_features(args.model)