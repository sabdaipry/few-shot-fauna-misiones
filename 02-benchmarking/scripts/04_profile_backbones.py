"""Perfila la latencia de inferencia de cada backbone midiendo el tiempo
de forward pass puro (sin carga de modelo) sobre una imagen de muestra."""
import sys
import os
import time
import argparse
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging

# --- SILENCIAR RUIDO (EJECUTAR ANTES DE IMPORTAR TRANSFORMERS) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

# --- CONFIGURACIÓN DE RUTAS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.config import MODELS_TO_TEST, DATASET_INDEX_PATH, BACKBONES_TIMES_PATH
from src.backbones import create_extractor

logger = setup_logger("profile-backbones")
# Obtenemos el logger de 'backbones' (el archivo src/backbones.py)
# y le subimos el nivel a ERROR. Así evitamos que imprima "-> Cargando modelo..." en cada iteración.
logging.getLogger("backbones").setLevel(logging.ERROR)

def prepare_input_and_forward(extractor, image_path, device):
    """
    Prepara la entrada y devuelve la función lambda para ejecutar el forward.
    Maneja las diferencias de nombre de atributos (transform vs processor).
    """
    img = Image.open(image_path).convert('RGB')
    name = extractor.__class__.__name__

    # --- 1. RESNET (Torchvision) ---
    if "ResNet" in name:
        # En implementaciones estándar, suele ser self.transform
        tf = getattr(extractor, 'transform', getattr(extractor, 'preprocess', None))
        
        if tf is None:
            # Fallback manual si no encuentra el atributo
            tf = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        img_t = tf(img).unsqueeze(0).to(device)
        return lambda: extractor.model(img_t)

    # --- 2. BioClip / CLIP (OpenCLIP) ---
    if "BioClip" in name or "CLIPExtractor" in name:
        # En OpenCLIP, la transformación a veces se guarda en self.processor o self.preprocess
        tf = getattr(extractor, 'processor', getattr(extractor, 'preprocess', None))
        
        if tf is None:
            raise ValueError(f"No se encontró transformación para {name}")

        img_t = tf(img).unsqueeze(0).to(device)
        return lambda: extractor.model.encode_image(img_t)

    # --- 3. SigLIP (Hugging Face) ---
    if "SigLIP" in name:
        inputs = extractor.processor(images=img, return_tensors="pt").to(device)
        return lambda: extractor.model.get_image_features(**inputs)

    # --- 4. DINO / ConvNext (Hugging Face) ---
    if "Dino" in name or "ConvNext" in name:
        inputs = extractor.processor(images=img, return_tensors="pt").to(device)
        return lambda: extractor.model(**inputs)

    raise ValueError(f"No se pudo configurar inferencia para {name}")
    
def profile_model_latency(model_name, image_path, device, num_iters=50, warmup=10):
    """Mide latencia media y desvío (ms) del forward pass de un backbone sobre una imagen."""
    # Silencia todos los loggers raíz durante la carga del modelo para evitar
    # el ruido de transformers/timm ("Some weights were not...", etc.)
    logging.getLogger().setLevel(logging.ERROR)

    try:
        # 1. Cargar Extractor
        extractor = create_extractor(model_name, device.type)

        try:
            extractor.load_model() # Asegurar que los pesos estén cargados
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            return None, None

        # 2. Preparar función de inferencia pura (Inputa ya en GPU/CPU)
        forward_fn = prepare_input_and_forward(extractor, image_path, device)

        # 3. Warm-up
        with torch.no_grad():
            for _ in range(warmup):
                _ = forward_fn()

        # 4. Profiling
        times = []
        with torch.no_grad():
            for _ in range(num_iters):
                if device.type == 'cuda': torch.cuda.synchronize()
                start = time.time()
                
                _ = forward_fn()
                
                if device.type == 'cuda': torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000) # ms
        
        # Liberar memoria VRAM si es posible
        del extractor
        del forward_fn
        if device.type == 'cuda': torch.cuda.empty_cache()

        return np.mean(times), np.std(times)
    except Exception as e:
        # Volvemos a activar el logger para reportar el error
        logger.setLevel(logging.INFO)
        logger.error(f"Error perfilando {model_name} en {image_path}: {e}")
        return None, None
    
def main():
    """Perfila todos los modelos de MODELS_TO_TEST y guarda los tiempos en CSV."""
    # --- ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Profiling de Latencia de Backbones")
    parser.add_argument("--iters", type=int, default=50, help="Cantidad de iteraciones para promediar")
    parser.add_argument("--warmup", type=int, default=10, help="Iteraciones de calentamiento")
    args = parser.parse_args()


    logger.info("")
    logger.info("==============================================")
    logger.info("   FASE 4: PROFILING DE BACKBONES")
    logger.info("==============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware de Profiling: {device.type.upper()}")

    # Obtener imagen de muestra
    if not os.path.exists(DATASET_INDEX_PATH):
        logger.error(f"No se encontró el índice del dataset en {DATASET_INDEX_PATH}")
        return
    df = pd.read_csv(DATASET_INDEX_PATH)
    sample_image_path = df.iloc[0]['filepath']

    results = []

    for i, model in enumerate(MODELS_TO_TEST):
        logger.info(f"[{i+1}/{len(MODELS_TO_TEST)}] Profiling {model} ...")
        avg, std = profile_model_latency(model, sample_image_path, device)

        if avg is not None:
            results.append({
                'Embedding Model': model,
                'Backbone Time (ms)': avg,
                'Backbone Std': std,
                'Device': device.type
            })
            # Feedback visual inmediato
            logger.info(f"-> {model}: {avg:.2f} ms ± {std:.2f} ms on {device.type.upper()}")
        else:
            logger.warning(f"-> No se pudo obtener resultados para {model}.")

    if results:
        pd.DataFrame(results).to_csv(BACKBONES_TIMES_PATH, index=False)
        logger.info(f"Resultados guardados en: {BACKBONES_TIMES_PATH}")
    else:
        logger.warning("No se obtuvieron resultados de profiling.")

if __name__ == "__main__":
    main()