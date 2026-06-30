"""
Worker de análisis en profundidad para SAREKO.

Extrae el mapa de atención del frame representativo y recolecta
las distancias temporales de un BiologicalEvent, todo en segundo plano.
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
    attention_ready(np.ndarray, np.ndarray)  — imagen original + mapa de atención
    distances_ready(list, list)              — frame_timestamps + frame_distances
    error_occurred(str)
    embedder_created(object)                 — BioCLIPEmbedder creado internamente si
                                               no se proporcionó uno desde MainWindow
    """

    attention_ready  = Signal(object, object)
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

        # Mapa de atención
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
            attn_map = embedder.get_attention_map(pil_image)
            self.attention_ready.emit(original_array, attn_map)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
