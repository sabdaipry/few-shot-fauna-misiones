"""
Extractores de embeddings para los distintos backbones evaluados en el benchmark.

Incluye implementaciones para:
- CNNs convolucionales: ResNet50 (baseline), ConvNeXt V2
- Vision Transformers (ViT) auto-supervisados: DINOv2, DINOv3
- Modelos multimodales imagen-texto: SigLIP v1, SigLIP v2, CLIP (OpenAI/LAION)
- Modelos de dominio biológico: BioCLIP v1 y v2

Todos los extractores heredan de BaseModel y exponen la misma interfaz:
load_model() para inicializar pesos y get_embedding(image_path) para inferencia.
"""

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from abc import ABC, abstractmethod
from pathlib import Path

import timm
from transformers import (
    AutoImageProcessor, AutoModel,
    SiglipModel, SiglipProcessor,
    CLIPModel,
)
import open_clip

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

# Importamos nuestro logger
from src.utils.logger import setup_logger

logger = setup_logger("backbones")


class BaseModel(ABC):
    """
    Clase abstracta que define la interfaz obligatoria para todos los modelos.

    Toda subclase debe implementar load_model() para inicializar pesos y
    get_embedding(image_path) para devolver el vector de características.
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self.name = "base_model"

    @staticmethod
    def _unwrap_image_features(outputs):
        """
        Normaliza la salida de get_image_features(), que según la versión de
        Hugging Face puede devolver un Tensor directo o un objeto contenedor.

        Orden de preferencia: Tensor → pooler_output → GAP sobre last_hidden_state.

        Args:
            outputs: Tensor o objeto de salida de HF get_image_features().

        Returns:
            torch.Tensor con las características de imagen.
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.mean(dim=1)
        return outputs

    @abstractmethod
    def load_model(self):
        """Carga los pesos del modelo."""
        pass

    @abstractmethod
    def get_embedding(self, image_path):
        """
        Recibe un path de imagen y devuelve un vector numpy (embedding).
        """
        pass


# --- 1. RESNET50 (BASELINE) ---
class ResNet50Extractor(BaseModel):
    """
    Extractor basado en ResNet50 preentrenado en ImageNet-1k (via timm).
    Funciona como baseline convolucional para comparar contra los ViTs.
    """
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.name = "resnet50"
        self.load_model()

    def load_model(self):
        """
        Carga ResNet50 (resnet50.a1_in1k) via timm y prepara las
        transformaciones de entrada usando los pesos DEFAULT de torchvision.
        """
        logger.info(f"-> Cargando {self.name} (Torchvision ImageNet)...")
        try:
            weights = models.ResNet50_Weights.DEFAULT
            # 'resnet50.a1_in1k' es la versión moderna y estable de ResNet50
            self.model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0)
            self.model.to(self.device)
            self.model.eval()
            self.processor = weights.transforms()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando ResNet50 ImageNet: {e}")
            raise

    def get_embedding(self, image_path):
        """
        Extrae el vector de características de la capa final (sin clasificador).

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.processor(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(img_t)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# --- CLASE INTERMEDIA: EXTRACTORES HF ViT (CLS / GAP) ---
class HFViTExtractor(BaseModel):
    """
    Clase intermedia para extractores basados en Vision Transformers de Hugging Face
    que devuelven last_hidden_state (p. ej. DINOv2, DINOv3).

    Implementa get_embedding() con soporte de pooling CLS y GAP, dejando
    load_model() abstracto para que cada subclase elija sus propios pesos.
    """
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.pooling = 'cls'

    def get_embedding(self, image_path):
        """
        Procesa una imagen y devuelve el embedding según la estrategia de pooling.

        Estrategias:
            - 'cls': toma el token [CLS] (índice 0 de la secuencia).
            - 'gap': promedia los tokens de parches (índices 1:) → Global Average Pooling.

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # El processor de HF devuelve un diccionario {'pixel_values': tensor}
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # outputs.last_hidden_state: (Batch, Sequence, Dim)
            # Sequence = 1 (CLS) + N_patches
            last_hidden_states = outputs.last_hidden_state
            if self.pooling == 'cls':
                return last_hidden_states[0, 0, :].cpu().numpy().flatten()
            elif self.pooling == 'gap':
                patch_tokens = last_hidden_states[0, 1:, :]
                avg_pool = torch.mean(patch_tokens, dim=0)
                return avg_pool.cpu().numpy().flatten()
            else:
                raise ValueError(f"Valor de pooling no soportado: {self.pooling!r}")
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# --- 2. DINOv2 (Meta) ---
class DinoV2Extractor(HFViTExtractor):
    """
    Extractor basado en DINOv2 (Meta, 2023) via Hugging Face Transformers.

    Soporta versiones small (384d), base (768d) y large (1024d), y dos estrategias
    de pooling: CLS (token de clasificación) y GAP (promedio de parches).
    """
    def __init__(self, version='base', pooling='cls', device='cpu'):
        super().__init__(device)
        # small: 384 dims, base: 768 dims, large: 1024 dims
        self.model_map = {
            'small': 'facebook/dinov2-small',
            'base':  'facebook/dinov2-base',
            'large': 'facebook/dinov2-large'
        }
        self.hf_name = self.model_map.get(version, 'facebook/dinov2-base')
        self.pooling = pooling

        suffix = '_gap' if pooling == 'gap' else ''
        self.name = f"dinov2_{version}{suffix}"
        self.load_model()

    def load_model(self):
        """Descarga e inicializa AutoImageProcessor y AutoModel de DINOv2 desde Hugging Face."""
        logger.info(f"-> Cargando {self.name} desde Hugging Face ({self.hf_name})...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_name)
            self.model = AutoModel.from_pretrained(self.hf_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando DINOv2: {e}")
            raise


# --- 3. DINOv3 (Meta - Nuevo) ---
class DinoV3Extractor(HFViTExtractor):
    """
    Extractor basado en DINOv3 (Meta, 2024) via Hugging Face Transformers.

    Requiere autenticación con Hugging Face (huggingface-cli login) y
    trust_remote_code=True para el código del modelo. Soporta las mismas
    estrategias de pooling que DinoV2Extractor (CLS y GAP).
    """
    def __init__(self, version='base', pooling='cls', device='cpu'):
        super().__init__(device)
        self.model_map = {
            'small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'base':  'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'large': 'facebook/dinov3-vitl16-pretrain-lvd1689m'
        }
        self.hf_name = self.model_map.get(version, 'facebook/dinov3-vitb16-pretrain-lvd1689m')
        self.pooling = pooling

        suffix = '_gap' if pooling == 'gap' else ''
        self.name = f"dinov3_{version}{suffix}"
        self.load_model()

    def load_model(self):
        """
        Descarga e inicializa AutoImageProcessor y AutoModel de DINOv3 desde Hugging Face.
        Requiere token de autenticación HF y trust_remote_code=True.
        """
        logger.info(f"-> Cargando {self.name} ({self.hf_name})...")
        try:
            # token=True usa el token de 'huggingface-cli login'
            # trust_remote_code=True requerido para modelos nuevos/privados
            self.processor = AutoImageProcessor.from_pretrained(
                self.hf_name, token=True, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.hf_name, token=True, trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando DINOv3: {e}")
            raise


# --- 4. BioCLIP ---
class BioClipExtractor(BaseModel):
    """
    Extractor basado en BioCLIP (Imageomics, 2024) via OpenCLIP.

    BioCLIP es un modelo CLIP especializado en taxonomía biológica, entrenado
    con el dataset TreeOfLife-10M. Soporta v1 (ViT-B/16, rápido) y
    v2 (ViT-L/14, SOTA).
    """
    def __init__(self, version='v1', device='cpu'):
        super().__init__(device)
        # v1: El original (ViT-B/16) - Rapido
        # v2: El nuevo (ViT-L/14) - Lento pero SOTA
        self.model_map = {
            'v1': 'hf-hub:imageomics/bioclip',
            'v2': 'hf-hub:imageomics/bioclip-2'
        }
        self.name = f"bioclip_{version}"
        self.hf_name = self.model_map.get(version, 'hf-hub:imageomics/bioclip')
        self.fallback_processor = "openai/clip-vit-base-patch16"
        self.load_model()

    def load_model(self):
        """
        Carga BioCLIP via open_clip usando el prefijo 'hf-hub:' para descargar
        desde Hugging Face Hub. El processor resultante es el transform de validación.
        """
        logger.info(f"-> Cargando {self.name}...")
        try:
            # create_model_and_transforms devuelve: modelo, transform_train, transform_val
            # Nos quedamos con transform_val (el tercero) para inferencia
            self.model, _, self.processor = open_clip.create_model_and_transforms(self.hf_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando BioCLIP: {e}")
            raise

    def get_embedding(self, image_path):
        """
        Extrae el embedding visual usando encode_image() de OpenCLIP.

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # OpenCLIP devuelve un Tensor directo (no un diccionario)
            img_t = self.processor(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model.encode_image(img_t)
            return outputs.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# --- 5. ConvNeXt V2 ---
class ConvNextV2Extractor(BaseModel):
    """
    Extractor basado en ConvNeXt V2 (Meta, 2023) via Hugging Face Transformers.

    Arquitectura convolucional moderna que compite con ViTs en rendimiento con
    menor costo computacional. El embedding se obtiene aplicando Global Average
    Pooling sobre las dimensiones espaciales del last_hidden_state.
    """
    def __init__(self, version='tiny', device='cpu'):
        super().__init__(device)
        self.name = f"convnextv2_{version}"
        self.hf_name = f"facebook/convnextv2-{version}-1k-224"
        self.load_model()

    def load_model(self):
        """Descarga e inicializa AutoImageProcessor y AutoModel de ConvNeXt V2 desde Hugging Face."""
        logger.info(f"-> Cargando {self.name}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_name)
            self.model = AutoModel.from_pretrained(self.hf_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando ConvNeXt V2: {e}")
            raise

    def get_embedding(self, image_path):
        """
        Extrae el embedding aplicando GAP sobre las dimensiones espaciales
        del last_hidden_state (dim=[-2, -1]).

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            return last_hidden_state.mean(dim=[-2, -1]).cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# --- 6. SigLIP v1 ---
class SigLIPExtractor(BaseModel):
    """
    Extractor basado en SigLIP v1 (Google, 2023) via Hugging Face Transformers.

    SigLIP reemplaza la pérdida contrastiva de CLIP por Sigmoid loss, logrando
    mejor rendimiento con batches más pequeños. Soporta variantes so400m (alta
    precisión, 400M parámetros) y base (rápida, 86M parámetros).
    """
    def __init__(self, version='so400m', device='cpu'):
        super().__init__(device)
        # so400m: Shape Optimized 400M (alta precision)
        # base:   Base Patch 16 (86M params, rapido)
        self.model_map = {
            'so400m': 'google/siglip-so400m-patch14-384',
            'base':   'google/siglip-base-patch16-224'
        }
        self.hf_name = self.model_map.get(version, 'google/siglip-so400m-patch14-384')
        self.name = f"siglip_{version}"
        self.load_model()

    def load_model(self):
        """Descarga e inicializa SiglipProcessor y SiglipModel desde Hugging Face."""
        logger.info(f"-> Cargando {self.name}...")
        try:
            self.processor = SiglipProcessor.from_pretrained(self.hf_name)
            self.model = SiglipModel.from_pretrained(self.hf_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando SigLIP 1: {e}")
            raise

    def get_embedding(self, image_path):
        """
        Extrae el embedding usando get_image_features() y normaliza la salida
        con _unwrap_image_features() para manejar distintas versiones de la API HF.

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            outputs = self._unwrap_image_features(outputs)
            return outputs.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# --- 7. SigLIP v2 ---
class SigLIP2Extractor(BaseModel):
    """
    Extractor basado en SigLIP 2 (Google, 2025) via Hugging Face Transformers.

    Soporta NaFlex (Native Aspect Ratio Flexible) para evitar deformación en
    crops de cámaras trampa. Usa AutoImageProcessor para manejar la lógica
    NaFlex automáticamente.
    """
    def __init__(self, version='so400m', device='cpu'):
        super().__init__(device)
        self.model_map = {
            'so400m': 'google/siglip2-so400m-patch14-384',  # Nativo 384px
            'base':   'google/siglip2-base-patch16-224'     # Nativo 224px
        }
        self.hf_name = self.model_map.get(version, 'google/siglip2-so400m-patch14-384')
        self.name = f"siglip2_{version}"
        self.load_model()

    def load_model(self):
        """
        Descarga e inicializa AutoImageProcessor y AutoModel de SigLIP 2 desde
        Hugging Face. AutoImageProcessor gestiona automáticamente la lógica NaFlex.
        """
        logger.info(f"-> Cargando {self.name}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_name)
            self.model = AutoModel.from_pretrained(self.hf_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando SigLIP 2: {e}")
            raise

    def get_embedding(self, image_path):
        """
        Extrae el embedding usando get_image_features() con parches variables (NaFlex)
        y normaliza la salida con _unwrap_image_features().

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # El procesador NaFlex genera parches variables según la imagen
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            outputs = self._unwrap_image_features(outputs)
            return outputs.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# --- 8. Vanilla CLIP (OpenAI) ---
class CLIPExtractor(BaseModel):
    """
    Extractor basado en CLIP estándar (OpenAI/LAION) via OpenCLIP.

    Sirve como baseline generalista para comparar contra los especialistas (BioCLIP).
    Soporta variantes base (ViT-B/16, comparable con BioCLIP v1) y large
    (ViT-L/14, comparable con BioCLIP v2), ambas preentrenadas en LAION-2B.
    """
    def __init__(self, version='large', device='cpu'):
        super().__init__(device)
        self.model_map = {
            # Base: ViT-B/16 (Mas rapido, comparable con BioCLIP v1)
            'base':  ('ViT-B-16', 'laion2b_s34b_b88k'),
            # Large: ViT-L/14 (comparable con BioCLIP v2)
            'large': ('ViT-L-14', 'laion2b_s32b_b82k')
        }

        if version not in self.model_map:
            raise ValueError(f"Version {version} no soportada en CLIPExtractor")

        self.model_name, self.pretrained = self.model_map[version]
        self.name = f"clip_{version}"
        self.load_model()

    def load_model(self):
        """
        Carga CLIP via open_clip especificando nombre del modelo y dataset de
        preentrenamiento. El processor resultante es el transform de validación.
        """
        logger.info(f"-> Cargando {self.name}...")
        try:
            # open_clip requiere nombre del modelo Y dataset de preentrenamiento
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"-> {self.name} cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando CLIP: {e}")
            raise

    def get_embedding(self, image_path):
        """
        Extrae el embedding visual usando encode_image() de OpenCLIP.

        Args:
            image_path: Ruta a la imagen de entrada.

        Returns:
            np.ndarray aplanado con el embedding, o None si ocurre un error.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # El processor de OpenCLIP devuelve un Tensor directo
            img_t = self.processor(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model.encode_image(img_t)
            return outputs.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error en {self.name} con {image_path}: {e}")
            return None


# ---------------------------------------------------------------------------
# Registry y factory centralizada
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[type, dict]] = {
    "resnet50":         (ResNet50Extractor,   {}),
    "convnextv2_tiny":  (ConvNextV2Extractor, {"version": "tiny"}),
    "convnextv2_base":  (ConvNextV2Extractor, {"version": "base"}),
    "dinov2_small":     (DinoV2Extractor,     {"version": "small", "pooling": "cls"}),
    "dinov2_base":      (DinoV2Extractor,     {"version": "base",  "pooling": "cls"}),
    "dinov2_small_gap": (DinoV2Extractor,     {"version": "small", "pooling": "gap"}),
    "dinov2_base_gap":  (DinoV2Extractor,     {"version": "base",  "pooling": "gap"}),
    "dinov3_small":     (DinoV3Extractor,     {"version": "small", "pooling": "cls"}),
    "dinov3_base":      (DinoV3Extractor,     {"version": "base",  "pooling": "cls"}),
    "dinov3_small_gap": (DinoV3Extractor,     {"version": "small", "pooling": "gap"}),
    "dinov3_base_gap":  (DinoV3Extractor,     {"version": "base",  "pooling": "gap"}),
    "siglip_base":      (SigLIPExtractor,     {"version": "base"}),
    "siglip_so400m":    (SigLIPExtractor,     {"version": "so400m"}),
    "siglip2_base":     (SigLIP2Extractor,    {"version": "base"}),
    "siglip2_so400m":   (SigLIP2Extractor,    {"version": "so400m"}),
    "bioclip_v1":       (BioClipExtractor,    {"version": "v1"}),
    "bioclip_v2":       (BioClipExtractor,    {"version": "v2"}),
    "clip_base":        (CLIPExtractor,       {"version": "base"}),
    "clip_large":       (CLIPExtractor,       {"version": "large"}),
}


def create_extractor(model_name: str, device: str = "cpu") -> BaseModel:
    """Instancia un extractor por nombre, normalizando device a str."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Modelo no reconocido: {model_name!r}. Disponibles: {available}"
        )
    cls, kwargs = MODEL_REGISTRY[model_name]
    return cls(device=device, **kwargs)
