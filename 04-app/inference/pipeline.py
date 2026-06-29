"""
Pipeline de inferencia para clasificación de fauna silvestre.

Módulo desacoplado de la GUI — testeable desde consola.

Clases:
    CatalogManager    — centroides, gallery KNN, nombres comunes
    BioCLIPEmbedder   — extracción de embeddings via open_clip
    SpeciesClassifier — clasificación por centroide + árbitro KNN
    VideoProcessor    — procesamiento de video con consenso temporal
"""

import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Rutas canónicas (relativas al archivo, no al cwd)
# ---------------------------------------------------------------------------
_INFERENCE_DIR = Path(__file__).resolve().parent   # 04-app/inference/
_APP_DIR       = _INFERENCE_DIR.parent             # 04-app/
_REPO_ROOT     = _APP_DIR.parent                   # raíz del repo

FEATURES_DIR  = _REPO_ROOT / "02-benchmarking" / "data" / "features" / "bioclip_v2"
DATASET_INDEX = _REPO_ROOT / "02-benchmarking" / "data" / "dataset_index.csv"
CATALOG_CACHE = _APP_DIR / "data" / "centroides_bioclip_v2.pkl"

# Umbral calibrado en 03-threshold-optimization (percentil 95 distribución intraclase)
CONFIDENCE_THRESHOLD = 0.1866
KNN_K = 5


# ===========================================================================
# CatalogManager
# ===========================================================================

class CatalogManager:
    """
    Gestiona el catálogo de especies: centroides, gallery y nombres comunes.

    Al instanciarse carga desde caché si existe; si no, recorre la jerarquía
    de embeddings en FEATURES_DIR, calcula centroides y persiste el resultado.

    Centroides: calculados con TODAS las imágenes (gallery + query, 4562 total).
    Gallery:    solo las 888 imágenes con split="gallery", para el árbitro KNN.
    """

    def __init__(self) -> None:
        self._centroids:     dict[str, np.ndarray]      = {}
        self._gallery_embs:  np.ndarray                 = np.empty((0, 768), dtype=np.float32)
        self._gallery_labels: np.ndarray                = np.empty((0,), dtype=object)
        self._common_names:  dict[str, dict[str, str]]  = {}
        self._load()

    # ------------------------------------------------------------------
    # Carga / construcción
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if CATALOG_CACHE.exists():
            self._load_from_cache()
        else:
            self._build_and_cache()

    def _load_from_cache(self) -> None:
        with open(CATALOG_CACHE, "rb") as f:
            data = pickle.load(f)
        self._centroids      = data["centroids"]
        self._gallery_embs   = data["gallery_embeddings"]
        self._gallery_labels = data["gallery_labels"]
        self._common_names   = data["common_names"]

    def _build_and_cache(self) -> None:
        print("[CatalogManager] Construyendo catálogo desde embeddings...")
        index_df = pd.read_csv(DATASET_INDEX)

        # Nombres comunes (una entrada por especie)
        self._common_names = {}
        for _, row in index_df.drop_duplicates("species").iterrows():
            self._common_names[row["species"]] = {
                "es_ar": str(row["nombre_comun_es_ar"]),
                "en":    str(row["nombre_comun_en"]),
            }

        all_embs:      list[np.ndarray] = []
        all_labels:    list[str]        = []
        gallery_embs:  list[np.ndarray] = []
        gallery_labels: list[str]       = []
        missing = 0

        for _, row in index_df.iterrows():
            species_folder = row["species"].replace(" ", "_")
            stem     = Path(row["filepath"]).stem
            npy_path = (FEATURES_DIR / row["family"] / row["genus"]
                        / species_folder / f"{stem}.npy")

            if not npy_path.exists():
                missing += 1
                continue

            emb = np.load(npy_path).astype(np.float32).ravel()
            all_embs.append(emb)
            all_labels.append(row["species"])

            if row["split"] == "gallery":
                gallery_embs.append(emb)
                gallery_labels.append(row["species"])

        if missing:
            print(f"[CatalogManager] Advertencia: {missing} archivos .npy no encontrados.")

        # Centroides: media de embeddings L2-normalizados, re-normalizada
        all_normed     = normalize(np.vstack(all_embs).astype(np.float64), norm="l2")
        all_labels_arr = np.array(all_labels)

        self._centroids = {}
        for sp in np.unique(all_labels_arr):
            mask     = all_labels_arr == sp
            mean_vec = all_normed[mask].mean(axis=0)
            norm_val = np.linalg.norm(mean_vec)
            self._centroids[sp] = (
                (mean_vec / norm_val).astype(np.float32)
                if norm_val > 0 else mean_vec.astype(np.float32)
            )

        # Gallery L2-normalizado para el árbitro KNN
        self._gallery_embs   = normalize(
            np.vstack(gallery_embs).astype(np.float64), norm="l2"
        ).astype(np.float32)
        self._gallery_labels = np.array(gallery_labels)

        CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CATALOG_CACHE, "wb") as f:
            pickle.dump({
                "centroids":          self._centroids,
                "gallery_embeddings": self._gallery_embs,
                "gallery_labels":     self._gallery_labels,
                "common_names":       self._common_names,
            }, f)
        print(f"[CatalogManager] Catálogo guardado en {CATALOG_CACHE}")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_species_list(self) -> list[str]:
        return sorted(self._centroids.keys())

    def get_centroids(self) -> dict[str, np.ndarray]:
        return self._centroids

    def get_common_names(self, species: str) -> dict[str, str]:
        return self._common_names.get(species, {"es_ar": "", "en": ""})

    def get_gallery_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (gallery_embs L2-norm, gallery_labels) para el árbitro KNN."""
        return self._gallery_embs, self._gallery_labels


# ===========================================================================
# BioCLIPEmbedder
# ===========================================================================

class BioCLIPEmbedder:
    """
    Carga BioCLIP v2 via open_clip y extrae embeddings L2-normalizados.

    Los pesos se descargan y cachean via Hugging Face Hub en el primer uso.
    Inferencia en CPU; no requiere CUDA.
    """

    _HF_HUB = "hf-hub:imageomics/bioclip-2"

    def __init__(self) -> None:
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(self._HF_HUB)
        self._model.eval()

    def embed_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Recibe una PIL.Image y devuelve el embedding L2-normalizado.

        Returns:
            np.ndarray shape (768,) float32.
        """
        img_t = self._preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_image(img_t)
        emb_np = emb.cpu().numpy().flatten().astype(np.float32)
        norm   = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm
        return emb_np

    def embed_batch(self, pil_images: list[Image.Image]) -> np.ndarray:
        """
        Recibe una lista de PIL.Image y devuelve embeddings L2-normalizados.

        Una única pasada forward del modelo procesa todas las imágenes del batch.

        Returns:
            np.ndarray shape (batch_size, 768) float32, cada fila normalizada.
        """
        batch = torch.stack([self._preprocess(img) for img in pil_images])
        with torch.no_grad():
            embs = self._model.encode_image(batch)
        embs_np = embs.cpu().numpy().astype(np.float32)
        norms   = np.linalg.norm(embs_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs_np / norms


# ===========================================================================
# SpeciesClassifier
# ===========================================================================

@dataclass
class ClassificationResult:
    species:            str
    nombre_comun_es_ar: str
    nombre_comun_en:    str
    confidence_level:   str         # "alta" | "baja"
    cosine_distance:    float       # distancia coseno al centroide de la especie predicha
    top5_candidates:    list[dict]  # [{"species": ..., "cosine_distance": ...}, ...]


class SpeciesClassifier:
    """
    Clasifica un embedding usando centroides (etapa principal) y árbitro KNN (zona gris).

    Si dmin_centroide <= CONFIDENCE_THRESHOLD → alta confianza, etiqueta directa.
    Si dmin_centroide >  CONFIDENCE_THRESHOLD → baja confianza, activa árbitro KNN
        (K=5 vecinos del gallery, voto ponderado por 1/distancia).
    """

    def __init__(self, catalog: CatalogManager) -> None:
        self._catalog = catalog

        species_list        = sorted(catalog.get_centroids().keys())
        self._species_index = species_list

        # Matriz de centroides (n_species × 768), L2-normalizada
        centroid_matrix = np.vstack(
            [catalog.get_centroids()[sp] for sp in species_list]
        ).astype(np.float64)
        self._centroid_matrix = normalize(centroid_matrix, norm="l2")

    def classify(self, embedding: np.ndarray) -> ClassificationResult:
        emb  = embedding.astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Distancias coseno a todos los centroides: 1 − similitud coseno
        dists      = 1.0 - (emb @ self._centroid_matrix.T)
        sorted_idx = np.argsort(dists)

        top5 = [
            {"species": self._species_index[i], "cosine_distance": float(dists[i])}
            for i in sorted_idx[:5]
        ]

        best_idx     = int(sorted_idx[0])
        best_species = self._species_index[best_idx]
        best_dist    = float(dists[best_idx])

        if best_dist <= CONFIDENCE_THRESHOLD:
            names = self._catalog.get_common_names(best_species)
            return ClassificationResult(
                species            = best_species,
                nombre_comun_es_ar = names["es_ar"],
                nombre_comun_en    = names["en"],
                confidence_level   = "alta",
                cosine_distance    = best_dist,
                top5_candidates    = top5,
            )

        # --- Árbitro KNN sobre el gallery completo ---
        gallery_embs, gallery_labels = self._catalog.get_gallery_embeddings()
        k_actual = min(KNN_K, len(gallery_embs))

        knn_dists    = 1.0 - (emb @ gallery_embs.astype(np.float64).T)
        top_k_idx    = np.argpartition(knn_dists, k_actual)[:k_actual]
        top_k_dists  = knn_dists[top_k_idx]
        top_k_labels = gallery_labels[top_k_idx]

        # Voto ponderado: peso = 1 / (distancia + ε)
        species_weights: dict[str, float] = {}
        for label, dist in zip(top_k_labels, top_k_dists):
            species_weights[label] = (
                species_weights.get(label, 0.0) + 1.0 / (float(dist) + 1e-6)
            )

        knn_winner = max(species_weights, key=species_weights.get)
        names      = self._catalog.get_common_names(knn_winner)

        # KNN winner puede no estar en top5 (que refleja distancias al centroide).
        # Insertarlo al frente para que aparezca en el dropdown de validación.
        if not any(c["species"] == knn_winner for c in top5):
            knn_winner_idx  = self._species_index.index(knn_winner)
            knn_winner_dist = float(dists[knn_winner_idx])
            top5 = [{"species": knn_winner, "cosine_distance": knn_winner_dist}] + top5

        return ClassificationResult(
            species            = knn_winner,
            nombre_comun_es_ar = names["es_ar"],
            nombre_comun_en    = names["en"],
            confidence_level   = "baja",
            cosine_distance    = best_dist,
            top5_candidates    = top5,
        )


# ===========================================================================
# VideoProcessor
# ===========================================================================

@dataclass
class BiologicalEvent:
    species:                  str
    nombre_comun_es_ar:       str
    nombre_comun_en:          str
    start_time:               float  # segundos desde el inicio del video
    end_time:                 float
    representative_frame_idx: int
    representative_timestamp: float
    cosine_distance:          float  # del frame representativo
    confidence_level:         str    # del frame representativo ("alta" | "baja")
    ambiguous:                bool
    top5_candidates:          list[dict] = field(default_factory=list)


class VideoProcessor:
    """
    Extrae frames de un video y aplica el pipeline con consenso temporal.

    Submuestreo: 1 frame cada N frames del video original.
    Consenso:    ventana no solapada de K frames; quórum M coincidencias de especie.
                 Si count >= M → evento confirmado.
                 Si count <  M → evento ambiguo (se registra la especie más frecuente).

    El frame representativo de un evento confirmado es el de menor distancia coseno
    entre los frames que votaron por la especie ganadora.
    El frame representativo de un evento ambiguo es el de menor distancia coseno global.
    """

    def __init__(
        self,
        embedder:   BioCLIPEmbedder,
        classifier: SpeciesClassifier,
        N: int = 30,
        K: int = 10,
        M: int = 6,
        batch_size: int = 8,
    ) -> None:
        self._embedder   = embedder
        self._classifier = classifier
        self._catalog    = classifier._catalog
        self.N          = N
        self.K          = K
        self.M          = M
        self.batch_size = batch_size

    def process(
        self,
        video_path:        str | Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[BiologicalEvent]:
        """
        Procesa el video y devuelve una lista de eventos biológicos.

        Args:
            video_path:        Ruta al archivo de video.
            progress_callback: callable(frames_procesados, total_estimado).
                               Firma compatible con GUI:
                               lambda done, tot: signal.emit(int(done / tot * 100))

        Returns:
            Lista de BiologicalEvent ordenada cronológicamente.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        est_sampled   = max(1, math.ceil(total_frames / self.N))

        frame_predictions: list[dict] = []
        batch_imgs:        list[Image.Image] = []
        batch_meta:        list[dict]        = []
        processed = 0

        def _flush() -> None:
            nonlocal processed
            if not batch_imgs:
                return
            embs = self._embedder.embed_batch(batch_imgs)
            for i, meta in enumerate(batch_meta):
                result = self._classifier.classify(embs[i])
                frame_predictions.append({
                    "frame_idx": meta["frame_idx"],
                    "timestamp": meta["timestamp"],
                    "result":    result,
                })
                processed += 1
                if progress_callback:
                    progress_callback(processed, est_sampled)
            batch_imgs.clear()
            batch_meta.clear()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.N == 0:
                timestamp = frame_idx / fps
                pil_img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch_imgs.append(pil_img)
                batch_meta.append({"frame_idx": frame_idx, "timestamp": timestamp})

                if len(batch_imgs) >= self.batch_size:
                    _flush()

            frame_idx += 1

        _flush()
        cap.release()
        return self._build_events(frame_predictions)

    # ------------------------------------------------------------------
    # Construcción de eventos con ventana de consenso
    # ------------------------------------------------------------------

    def _build_events(self, predictions: list[dict]) -> list[BiologicalEvent]:
        if not predictions:
            return []

        events: list[BiologicalEvent] = []
        i = 0

        while i < len(predictions):
            window    = predictions[i : i + self.K]
            counts    = Counter(w["result"].species for w in window)
            winner, count = counts.most_common(1)[0]
            ambiguous = count < self.M

            if not ambiguous:
                winner_frames = [w for w in window if w["result"].species == winner]
                best = min(winner_frames, key=lambda w: w["result"].cosine_distance)
            else:
                # Evento ambiguo: especie más frecuente, frame con menor distancia global
                best = min(window, key=lambda w: w["result"].cosine_distance)

            names = self._catalog.get_common_names(winner)

            top5 = best["result"].top5_candidates
            if not any(c.get("species") == winner for c in top5):
                # winner no está en el top5 del frame representativo (consenso ambiguo
                # o winner ganó por KNN). Buscar su distancia en los frames que votaron
                # por él; si no se encuentra, usar la distancia del frame representativo.
                winner_dist: "float | None" = None
                for w in window:
                    if w["result"].species != winner:
                        continue
                    for cand in w["result"].top5_candidates:
                        if cand.get("species") == winner:
                            d = cand.get("cosine_distance")
                            if winner_dist is None or d < winner_dist:
                                winner_dist = d
                            break
                top5 = [
                    {"species": winner,
                     "cosine_distance": winner_dist if winner_dist is not None else best["result"].cosine_distance}
                ] + top5

            events.append(BiologicalEvent(
                species                  = winner,
                nombre_comun_es_ar       = names["es_ar"],
                nombre_comun_en          = names["en"],
                start_time               = window[0]["timestamp"],
                end_time                 = window[-1]["timestamp"],
                representative_frame_idx = best["frame_idx"],
                representative_timestamp = best["timestamp"],
                cosine_distance          = best["result"].cosine_distance,
                confidence_level         = best["result"].confidence_level,
                ambiguous                = ambiguous,
                top5_candidates          = top5,
            ))

            i += self.K

        return events
