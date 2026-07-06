"""
Worker de análisis en profundidad para SAREKO.

Extrae los mapas de activación (Attention Rollout y GradCAM) del frame
representativo y recolecta las distancias temporales de un BiologicalEvent,
todo en segundo plano.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

_APP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

VIDEO_EXTS = {".mp4", ".avi", ".mov"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_catalog_cache = None  # CatalogManager, cargado una única vez por proceso


def _get_catalog():
    """Devuelve un CatalogManager compartido (evita recargar el pickle cada análisis)."""
    global _catalog_cache
    if _catalog_cache is None:
        from inference.pipeline import CatalogManager
        _catalog_cache = CatalogManager()
    return _catalog_cache


def _load_representative_frame(filepath: Path, frame_idx: int) -> Optional[Image.Image]:
    """Carga el frame representativo desde disco como PIL.Image RGB."""
    try:
        ext = filepath.suffix.lower()
        if ext in IMAGE_EXTS:
            return Image.open(filepath).convert("RGB")
        if ext in VIDEO_EXTS:
            import cv2
            cap = cv2.VideoCapture(str(filepath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            import cv2 as _cv2
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
    except Exception:
        pass
    return None


class DeepAnalysisWorker(QThread):
    """
    Análisis en profundidad de un BiologicalEvent.

    Señales
    -------
    maps_ready(np.ndarray, np.ndarray, np.ndarray)  — imagen original,
                                                       mapa de attention rollout,
                                                       mapa de GradCAM
    distances_ready(list, list)              — frame_timestamps + frame_distances
    error_occurred(str)
    embedder_created(object)                 — BioCLIPEmbedder creado internamente si
                                               no se proporcionó uno desde MainWindow
    """

    maps_ready       = Signal(object, object, object)
    distances_ready  = Signal(list, list)
    error_occurred   = Signal(str)
    embedder_created = Signal(object)

    def __init__(
        self,
        event,
        embedder,
        filepath: Optional[Path],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._event    = event
        self._embedder = embedder       # puede ser None en el primer uso
        self._filepath = filepath

    # ------------------------------------------------------------------

    def run(self) -> None:
        event = self._event

        # Distancias temporales — disponibles directamente en el evento
        distances  = list(getattr(event, "frame_distances",  []))
        timestamps = list(getattr(event, "frame_timestamps", []))
        if distances:
            self.distances_ready.emit(timestamps, distances)

        # Mapas de activación (attention rollout + GradCAM)
        filepath  = self._filepath
        frame_idx = getattr(event, "representative_frame_idx", 0)

        if filepath is None:
            self.error_occurred.emit("No hay ruta de archivo disponible para el frame representativo.")
            return

        pil_image = _load_representative_frame(filepath, frame_idx)
        if pil_image is None:
            self.error_occurred.emit("No se pudo cargar el frame representativo desde disco.")
            return

        try:
            embedder = self._embedder
            if embedder is None:
                from inference.pipeline import BioCLIPEmbedder
                embedder = BioCLIPEmbedder()
                self.embedder_created.emit(embedder)

            original_array = np.array(pil_image)
            rollout_map    = embedder.get_attention_map(pil_image)

            top5 = getattr(event, "top5_candidates", [])
            gradcam_map = np.zeros(
                (original_array.shape[0], original_array.shape[1]), dtype=np.float32
            )
            if top5:
                centroid = _get_catalog().get_centroids().get(top5[0].get("species", ""))
                if centroid is not None:
                    gradcam_map = embedder.get_gradcam(pil_image, centroid)

            self.maps_ready.emit(original_array, rollout_map, gradcam_map)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
