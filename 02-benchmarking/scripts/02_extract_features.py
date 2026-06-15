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
    from src.backbones import (
        ResNet50Extractor, 
        DinoV2Extractor,
        DinoV3Extractor,
        BioClipExtractor,
        ConvNextV2Extractor,
        SigLIPExtractor,
        SigLIP2Extractor,
        CLIPExtractor
    )
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

# Configuración
DATASET_INDEX = project_root / "data/dataset_index.csv"
FEATURES_DIR = project_root / "data/features"

logger = setup_logger("feature_extractor", log_dir=project_root / "logs")

def get_model(model_name, device):
    """Fábrica de modelos"""
    if model_name == "resnet50": return ResNet50Extractor(device=device)
    elif model_name == "dinov2_small": return DinoV2Extractor(version='small', pooling='cls', device=device)
    elif model_name == "dinov2_small_gap": return DinoV2Extractor(version='small', pooling='gap', device=device)
    elif model_name == "dinov2_base": return DinoV2Extractor(version='base', pooling='cls', device=device)
    elif model_name == "dinov2_base_gap": return DinoV2Extractor(version='base', pooling='gap', device=device)
    elif model_name == "dinov3_small": return DinoV3Extractor(version='small', pooling='cls', device=device)
    elif model_name == "dinov3_small_gap": return DinoV3Extractor(version='small', pooling='gap', device=device)
    elif model_name == "dinov3_base": return DinoV3Extractor(version='base', pooling='cls', device=device)
    elif model_name == "dinov3_base_gap": return DinoV3Extractor(version='base', pooling='gap', device=device)
    elif model_name == "bioclip_v1": return BioClipExtractor(version='v1', device=device)
    elif model_name == "bioclip_v2": return BioClipExtractor(version='v2', device=device)
    elif model_name == "convnextv2_tiny": return ConvNextV2Extractor(version='tiny', device=device)
    elif model_name == "convnextv2_base": return ConvNextV2Extractor(version='base', device=device)
    elif model_name == "siglip_so400m": return SigLIPExtractor(version='so400m', device=device)
    elif model_name == "siglip_base": return SigLIPExtractor(version='base', device=device)
    elif model_name == "siglip2_base": return SigLIP2Extractor(version='base', device=device)
    elif model_name == "siglip2_so400m": return SigLIP2Extractor(version='so400m', device=device)
    elif model_name == "clip_base": return CLIPExtractor(version='base', device=device)
    elif model_name == "clip_large": return CLIPExtractor(version='large', device=device)
    else: raise ValueError(f"Modelo no reconocido: {model_name}")

def extract_features(model_name):
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
        extractor = get_model(model_name, device)
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
                        choices=[
                            "resnet50",
                            "dinov2_small", "dinov2_base", 
                            "dinov2_small_gap", "dinov2_base_gap",
                            "dinov3_small", "dinov3_base", 
                            "dinov3_small_gap", "dinov3_base_gap",
                            "bioclip_v1", "bioclip_v2",
                            "convnextv2_tiny", "convnextv2_base", 
                            "siglip_so400m", "siglip_base",
                            "siglip2_base", "siglip2_so400m",
                            "clip_base", "clip_large"
                        ],
                        help="Modelo a utilizar")
    
    args = parser.parse_args()
    
    extract_features(args.model)