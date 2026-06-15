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

logger = setup_logger("profile-backbones")
# Obtenemos el logger de 'backbones' (el archivo src/backbones.py)
# y le subimos el nivel a ERROR. Así evitamos que imprima "-> Cargando modelo..." en cada iteración.
logging.getLogger("backbones").setLevel(logging.ERROR)

# --- FÁBRICA DE MODELOS ---
def get_model_instance(model_name, device):
    """Instancia el modelo según el nombre."""
    if model_name == "resnet50": return ResNet50Extractor(device=device)
    elif model_name == "dinov2_small": return DinoV2Extractor(version='small', pooling='cls', device=device)
    elif model_name == "dinov2_base": return DinoV2Extractor(version='base', pooling='cls', device=device)
    elif model_name == "dinov2_small_gap": return DinoV2Extractor(version='small', pooling='mean', device=device)
    elif model_name == "dinov2_base_gap": return DinoV2Extractor(version='base', pooling='mean', device=device)
    elif model_name == "dinov3_small": return DinoV3Extractor(version='small', pooling='cls', device=device)
    elif model_name == "dinov3_base": return DinoV3Extractor(version='base', pooling='cls', device=device)
    elif model_name == "dinov3_small_gap": return DinoV3Extractor(version='small', pooling='mean', device=device)
    elif model_name == "dinov3_base_gap": return DinoV3Extractor(version='base', pooling='mean', device=device)
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
    else: raise ValueError(f"Modelo desconocido: {model_name}")
    
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
    # Desactivar logs temporalmente para la carga interna
    logging.getLogger().setLevel(logging.ERROR)

    try:
        # 1. Cargar Extractor
        extractor = get_model_instance(model_name, device)

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

    # --- ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Profiling de Latencia de Backbones")
    parser.add_argument("--iters", type=int, default=50, help="Cantidad de iteraciones para promediar")
    parser.add_argument("--warmup", type=int, default=10, help="Iteraciones de calentamiento")
    args = parser.parse_args()


    print("\n")
    logger.info("==============================================")
    logger.info("   FASE 4: PROFILING DE BACKBONES")
    logger.info("==============================================")

    INDEX_PATH = "data/dataset_index.csv"
    OUTPUT_FILE = "data/backbones_times.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware de Profiling: {device.type.upper()}")

    # Obtener imagen de muestra
    if not os.path.exists(INDEX_PATH):
        logger.error(f"No se encontró el índice del dataset en {INDEX_PATH}")
        return
    df = pd.read_csv(INDEX_PATH)
    sample_image_path = df.iloc[0]['filepath']

    models = [
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

    results = []

    
    for i, model in enumerate(models):
        logger.info(f"[{i+1}/{len(models)}] Profiling {model} ...")
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
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Resultados guardados en: {OUTPUT_FILE}")
    else:
        logger.warning("No se obtuvieron resultados de profiling.")

if __name__ == "__main__":
    main()