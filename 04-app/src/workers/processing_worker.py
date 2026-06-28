"""
Worker de procesamiento en segundo plano para SAREKO.

Corre en un QThread separado; nunca bloquea el hilo principal.
Emite señales Qt para actualizar la GUI en tiempo real.
"""

import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

# Aseguramos que 04-app/ esté en sys.path para importar inference/
_APP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from inference.pipeline import (
    BioCLIPEmbedder,
    CatalogManager,
    SpeciesClassifier,
    VideoProcessor,
)
from PIL import Image

VIDEO_EXTS = {".mp4", ".avi", ".mov"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ProcessingWorker(QThread):
    """
    Procesa una lista de archivos (videos e imágenes) con el pipeline de inferencia.

    Señales
    -------
    progress_updated(int)          — porcentaje global 0–100
    file_started(str)              — nombre del archivo que empieza
    file_completed(str, list)      — archivo + lista de BiologicalEvent / ClassificationResult
    stage_updated(str, int)        — nombre de etapa + % completado de esa etapa
    error_occurred(str, str)       — archivo + mensaje de error
    finished()                     — todo el lote completado (señal base de QThread)
    metrics_updated(dict)          — métricas en tiempo real para la card de seguimiento
    """

    progress_updated = Signal(int)
    file_started     = Signal(str)
    file_completed   = Signal(str, list, dict)
    stage_updated    = Signal(str, int)
    error_occurred   = Signal(str, str)
    metrics_updated  = Signal(dict)

    def __init__(
        self,
        files:      list[Path],
        N:          int  = 30,
        K:          int  = 10,
        M:          int  = 6,
        batch_size: int  = 8,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._files      = files
        self.N           = N
        self.K           = K
        self.M           = M
        self.batch_size  = batch_size
        self._stop_flag  = False

        # Estado de métricas acumuladas
        self._total_frames_done = 0
        self._start_time:    Optional[float] = None
        self._bioclip_time:  float = 0.0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop_flag = True

    # ------------------------------------------------------------------
    # Hilo principal
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._stop_flag = False
        self._start_time = time.monotonic()
        total_files = len(self._files)

        # Etapa 0: carga del catálogo y el modelo (una sola vez)
        self.stage_updated.emit("Cargando catálogo de especies", 0)
        try:
            catalog    = CatalogManager()
            embedder   = BioCLIPEmbedder()
            classifier = SpeciesClassifier(catalog)
        except Exception as exc:
            self.error_occurred.emit("inicialización", str(exc))
            return

        self.stage_updated.emit("Catálogo cargado", 100)

        for file_idx, file_path in enumerate(self._files):
            if self._stop_flag:
                break

            self.file_started.emit(file_path.name)
            self._emit_metrics(
                tipo       = "archivo único" if total_files == 1 else "lote",
                progreso   = int(file_idx / total_files * 100),
                archivo    = file_path.name,
                estado     = "procesando",
                frames_arch= 0,
            )

            try:
                if file_path.suffix.lower() in VIDEO_EXTS:
                    events, meta = self._process_video(
                        file_path, embedder, classifier,
                        file_idx, total_files,
                    )
                elif file_path.suffix.lower() in IMAGE_EXTS:
                    events, meta = self._process_image(
                        file_path, embedder, classifier,
                        file_idx, total_files,
                    )
                else:
                    raise ValueError(f"Formato no soportado: {file_path.suffix}")

                self.file_completed.emit(file_path.name, events, meta)

            except Exception as exc:
                self.error_occurred.emit(file_path.name, str(exc))

            file_pct = int((file_idx + 1) / total_files * 100)
            self.progress_updated.emit(file_pct)

        # Porcentaje final explícito
        if not self._stop_flag:
            self.progress_updated.emit(100)
            self.stage_updated.emit("Escritura de resultados", 100)
            self._emit_metrics(
                tipo      = "archivo único" if total_files == 1 else "lote",
                progreso  = 100,
                archivo   = self._files[-1].name if self._files else "",
                estado    = "completado",
                frames_arch = 0,
            )

    # ------------------------------------------------------------------
    # Procesamiento de video
    # ------------------------------------------------------------------

    def _process_video(
        self,
        path:       Path,
        embedder:   BioCLIPEmbedder,
        classifier: SpeciesClassifier,
        file_idx:   int,
        total:      int,
    ) -> list:
        import math
        import cv2

        self.stage_updated.emit("Extracción de frames", 0)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir: {path}")

        fps            = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        est_sampled    = max(1, math.ceil(total_frames / self.N))
        cap.release()

        self.stage_updated.emit("Extracción de frames", 100)
        self.stage_updated.emit("Generación de embeddings (BioCLIP)", 0)

        frames_done  = [0]
        file_base_pct = int(file_idx / total * 100)
        file_span_pct = int(1 / total * 100)

        def _progress(done: int, tot: int) -> None:
            frames_done[0] = done
            stage_pct = int(done / max(tot, 1) * 100)
            self.stage_updated.emit("Generación de embeddings (BioCLIP)", stage_pct)

            global_pct = file_base_pct + int(stage_pct / 100 * file_span_pct)
            self.progress_updated.emit(min(global_pct, 99))

            elapsed = time.monotonic() - self._start_time
            self._emit_metrics(
                tipo        = "archivo único" if total == 1 else "lote",
                progreso    = global_pct,
                archivo     = path.name,
                estado      = "procesando",
                frames_arch = done,
                tiempo_total= elapsed,
            )

        t0 = time.monotonic()
        processor = VideoProcessor(
            embedder, classifier,
            N=self.N, K=self.K, M=self.M,
            batch_size=self.batch_size,
        )
        events = processor.process(path, progress_callback=_progress)
        file_processing_sec = time.monotonic() - t0
        self._bioclip_time += file_processing_sec
        self._total_frames_done += frames_done[0]

        self.stage_updated.emit("Clasificación por similitud coseno", 100)
        self.stage_updated.emit("Árbitro KNN (si aplica)", 100)
        self.stage_updated.emit("Consenso temporal", 100)
        self.stage_updated.emit("Escritura de resultados", 50)

        return events, {
            "duration_sec":     video_duration,
            "processing_sec":   file_processing_sec,
            "frames_processed": frames_done[0],
        }

    # ------------------------------------------------------------------
    # Procesamiento de imagen
    # ------------------------------------------------------------------

    def _process_image(
        self,
        path:       Path,
        embedder:   BioCLIPEmbedder,
        classifier: SpeciesClassifier,
        file_idx:   int,
        total:      int,
    ) -> list:
        self.stage_updated.emit("Extracción de frames", 100)
        self.stage_updated.emit("Generación de embeddings (BioCLIP)", 0)

        t0 = time.monotonic()
        pil_img = Image.open(path).convert("RGB")
        emb     = embedder.embed_image(pil_img)
        file_processing_sec = time.monotonic() - t0
        self._bioclip_time += file_processing_sec

        self.stage_updated.emit("Generación de embeddings (BioCLIP)", 100)
        self.stage_updated.emit("Clasificación por similitud coseno", 0)

        result = classifier.classify(emb)

        self.stage_updated.emit("Clasificación por similitud coseno", 100)
        self.stage_updated.emit("Árbitro KNN (si aplica)", 100)
        self.stage_updated.emit("Consenso temporal", 100)
        self.stage_updated.emit("Escritura de resultados", 100)

        self._total_frames_done += 1

        elapsed = time.monotonic() - self._start_time
        file_pct = int((file_idx + 1) / total * 100)
        self._emit_metrics(
            tipo        = "archivo único" if total == 1 else "lote",
            progreso    = file_pct,
            archivo     = path.name,
            estado      = "completado",
            frames_arch = 1,
            tiempo_total= elapsed,
        )
        return [result], {
            "duration_sec":     None,
            "processing_sec":   file_processing_sec,
            "frames_processed": 1,
        }

    # ------------------------------------------------------------------
    # Emisión de métricas
    # ------------------------------------------------------------------

    def _emit_metrics(
        self,
        tipo:         str,
        progreso:     int,
        archivo:      str,
        estado:       str,
        frames_arch:  int,
        tiempo_total: float = 0.0,
    ) -> None:
        self.metrics_updated.emit({
            "tipo":             tipo,
            "progreso":         progreso,
            "archivo":          archivo,
            "estado":           estado,
            "tiempo_bioclip":   round(self._bioclip_time, 1),
            "tiempo_total":     round(tiempo_total, 1),
            "frames_archivo":   frames_arch,
            "frames_acumulados": self._total_frames_done,
        })
