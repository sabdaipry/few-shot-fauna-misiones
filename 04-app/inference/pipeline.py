"""
Pipeline de inferencia para clasificación de fauna silvestre.

Módulo desacoplado de la GUI — testeable desde consola.

Clases:
    CatalogManager          — centroides, gallery KNN, nombres comunes
    BioCLIPEmbedder         — extracción de embeddings via open_clip
    SpeciesClassifier       — clasificación por centroide + árbitro KNN
    SlidingWindowConsensus  — consenso deslizante frame a frame
    VideoProcessor          — procesamiento de video con consenso temporal
"""

import json
import logging
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
_CONFIG_PATH  = _APP_DIR / "data" / "config.json"

def _load_config() -> dict:
    try:
        if _CONFIG_PATH.exists():
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

_CONFIG = _load_config()

logger = logging.getLogger("sareko.pipeline")

# Umbral calibrado en 03-threshold-optimization (percentil 95 distribución intraclase)
CONFIDENCE_THRESHOLD   = float(_CONFIG.get("confidence_threshold",   0.1866))
REJECTION_THRESHOLD    = float(_CONFIG.get("rejection_threshold",    0.25))
KNN_K                  = int(  _CONFIG.get("knn_k",                  5))
_SLIDING_P_DEFAULT     = int(  _CONFIG.get("sliding_close_quorum_P", 3))
_SLIDING_THRESHOLD     = float(_CONFIG.get("sliding_close_threshold", CONFIDENCE_THRESHOLD))


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
        logger.debug("[CatalogManager] Construyendo catálogo desde embeddings...")
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
            logger.warning("[CatalogManager] %d archivos .npy no encontrados.", missing)

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
        logger.debug("[CatalogManager] Catálogo guardado en %s", CATALOG_CACHE)

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

    def get_attention_map(self, pil_image: Image.Image) -> np.ndarray:
        """
        Calcula el mapa de atención via attention rollout sobre todos los bloques del ViT.

        Registra hooks temporales en cada bloque de atención, ejecuta una pasada
        forward, aplica attention rollout (producto acumulado de matrices de atención
        con residual connection) y redimensiona la máscara al tamaño de la imagen.

        Returns:
            np.ndarray shape (H, W) float32, valores normalizados 0–1.
        """
        W, H = pil_image.size
        resblocks = self._model.visual.transformer.resblocks
        captured: list[torch.Tensor] = []
        patched_mhas: list = []

        def _make_patched_fwd(cls_fwd, mha_instance, storage):
            def _patched(query, key, value, **kwargs):
                kwargs.pop("need_weights", None)
                kwargs.pop("average_attn_weights", None)
                attn_out, weights = cls_fwd(
                    mha_instance, query, key, value,
                    need_weights=True,
                    average_attn_weights=False,
                    **kwargs,
                )
                if weights is not None:
                    storage.append(weights.detach().cpu())
                return attn_out, None
            return _patched

        for block in resblocks:
            mha = block.attn
            mha.forward = _make_patched_fwd(type(mha).forward, mha, captured)
            patched_mhas.append(mha)

        try:
            img_t = self._preprocess(pil_image).unsqueeze(0)
            with torch.no_grad():
                self._model.encode_image(img_t)
        finally:
            for mha in patched_mhas:
                try:
                    del mha.forward
                except AttributeError:
                    pass

        if not captured:
            return np.zeros((H, W), dtype=np.float32)

        # Attention rollout: A_hat_l = (0.5·A_l + 0.5·I) @ A_hat_{l-1}
        seq_len = captured[0].shape[-1]
        rollout = torch.eye(seq_len, dtype=torch.float32)
        for attn in captured:
            a = attn[0].float().mean(dim=0)          # promedio sobre cabezas → (seq, seq)
            a = 0.5 * a + 0.5 * torch.eye(seq_len)
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a @ rollout

        # Fila del token [CLS] (índice 0), descartando la posición del propio CLS
        cls_attn  = rollout[0, 1:].numpy()
        grid_size = int(round(len(cls_attn) ** 0.5))
        mask_2d   = cls_attn.reshape(grid_size, grid_size)

        v_min, v_max = mask_2d.min(), mask_2d.max()
        if v_max > v_min:
            mask_2d = (mask_2d - v_min) / (v_max - v_min)

        mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8)).resize(
            (W, H), Image.Resampling.BILINEAR
        )
        return np.array(mask_img).astype(np.float32) / 255.0


# ===========================================================================
# SpeciesClassifier
# ===========================================================================

@dataclass
class ClassificationResult:
    species:            str
    nombre_comun_es_ar: str
    nombre_comun_en:    str
    confidence_level:   str         # "alta" | "baja" | "rechazado"
    cosine_distance:    float       # distancia coseno al centroide de la especie predicha
    top5_candidates:    list[dict]  # [{"species": ..., "cosine_distance": ...}, ...]
    decisor:            str = ""    # "BioCLIP" | "KNN" | "Rechazo"


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
                decisor            = "BioCLIP",
            )

        if best_dist >= REJECTION_THRESHOLD:
            names = self._catalog.get_common_names(best_species)
            logger.debug(
                "[SpeciesClassifier] Frame rechazado: %s d=%.4f > %.4f",
                best_species, best_dist, REJECTION_THRESHOLD,
            )
            return ClassificationResult(
                species            = best_species,
                nombre_comun_es_ar = names["es_ar"],
                nombre_comun_en    = names["en"],
                confidence_level   = "rechazado",
                cosine_distance    = best_dist,
                top5_candidates    = top5,
                decisor            = "Rechazo",
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
            decisor            = "KNN",
        )


# ===========================================================================
# SlidingWindowConsensus
# ===========================================================================

class SlidingWindowConsensus:
    """
    Consenso deslizante con estado persistente.

    - Buffer de los últimos K frames con predicciones y distancias.
    - Evento confirmado cuando M frames consecutivos coinciden en especie
      con d <= umbral.
    - Evento cerrado cuando P frames consecutivos tienen d > umbral.
    - Parámetros: K (buffer), M (quórum inicio), P (quórum cierre, default=3),
      umbral=CONFIDENCE_THRESHOLD.

    Uso:
        consensus = SlidingWindowConsensus(K=10, M=6, catalog=catalog)
        for pred in frame_predictions:
            ev = consensus.feed(pred)
            if ev: events.append(ev)
        final = consensus.flush()
        if final: events.append(final)
    """

    def __init__(
        self,
        K:         int,
        M:         int,
        P:         int   = _SLIDING_P_DEFAULT,
        threshold: float = _SLIDING_THRESHOLD,
        catalog:   "CatalogManager | None" = None,
    ) -> None:
        self.K         = K
        self.M         = M
        self.P         = P
        self.threshold = threshold
        self._catalog  = catalog
        self._reset()

    # ------------------------------------------------------------------
    # Estado interno

    def _reset(self) -> None:
        self._state: str = "idle"                   # "idle" | "active"
        self._candidate_frames:  list[dict] = []
        self._candidate_species: Optional[str] = None
        self._event_frames:  list[dict] = []
        self._close_counter: int = 0
        self._last_low_frame: Optional[dict] = None  # último frame con d <= threshold

    def _reset_event(self) -> None:
        self._state          = "idle"
        self._event_frames   = []
        self._close_counter  = 0
        self._last_low_frame = None
        self._candidate_frames   = []
        self._candidate_species  = None

    # ------------------------------------------------------------------
    # Alimentación frame a frame

    def feed(self, frame_data: dict) -> "Optional[BiologicalEvent]":
        """
        Procesa un frame. Devuelve un BiologicalEvent si el evento se cierra, None si no.

        frame_data keys: "frame_idx", "timestamp", "result" (ClassificationResult).
        """
        result  = frame_data["result"]
        species = result.species
        dist    = result.cosine_distance

        if self._state == "idle":
            if dist <= self.threshold:
                if (not self._candidate_frames
                        or species == self._candidate_species):
                    self._candidate_species = species
                    self._candidate_frames.append(frame_data)
                else:
                    # Especie distinta — reiniciar candidato desde este frame
                    self._candidate_frames  = [frame_data]
                    self._candidate_species = species

                if len(self._candidate_frames) >= self.M:
                    self._state          = "active"
                    self._event_frames   = list(self._candidate_frames)
                    self._last_low_frame = self._candidate_frames[-1]
                    self._close_counter  = 0
                    self._candidate_frames  = []
                    self._candidate_species = None
            else:
                # Alta distancia — reiniciar candidato
                self._candidate_frames  = []
                self._candidate_species = None

        elif self._state == "active":
            self._event_frames.append(frame_data)
            if dist <= self.threshold:
                self._close_counter  = 0
                self._last_low_frame = frame_data
            else:
                self._close_counter += 1
                if self._close_counter >= self.P:
                    event = self._build_event()
                    self._reset_event()
                    return event

        return None

    def flush(self) -> "Optional[BiologicalEvent]":
        """
        Cierra el evento pendiente al finalizar el video.

        Caso 1 — estado "active": el quórum M se alcanzó y el evento aún
        no se cerró (el animal seguía en escena al acabarse los frames).
        Se emite con ambiguous=False.

        Caso 2 — estado "idle" con candidatos: el quórum M nunca se
        completó, pero había frames con detección acumulándose. Se emite
        como evento ambiguo (ambiguous=True) para que el usuario pueda
        revisarlo en la pestaña Validación.
        """
        if self._state == "active" and self._event_frames:
            event = self._build_event()
            self._reset_event()
            return event

        if self._state == "idle" and self._candidate_frames:
            event = self._build_ambiguous_event_from_candidates()
            self._reset()
            return event

        return None

    # ------------------------------------------------------------------
    # Construcción del evento

    def _build_ambiguous_event_from_candidates(self) -> "BiologicalEvent":
        """
        Construye un evento ambiguo a partir de los candidatos pendientes en idle.

        Todos los frames en _candidate_frames tienen dist <= threshold y la misma
        especie (_candidate_species). El evento se marca ambiguous=True porque el
        quórum M no se alcanzó antes del fin del video.
        """
        frames = self._candidate_frames
        winner = self._candidate_species or frames[0]["result"].species
        best   = min(frames, key=lambda f: f["result"].cosine_distance)

        names = (
            self._catalog.get_common_names(winner)
            if self._catalog else {"es_ar": "", "en": ""}
        )

        top5 = best["result"].top5_candidates
        if not any(c.get("species") == winner for c in top5):
            top5 = [{"species": winner,
                     "cosine_distance": best["result"].cosine_distance}] + top5

        return BiologicalEvent(
            species                  = winner,
            nombre_comun_es_ar       = names["es_ar"],
            nombre_comun_en          = names["en"],
            start_time               = frames[0]["timestamp"],
            end_time                 = frames[-1]["timestamp"],
            representative_frame_idx = best["frame_idx"],
            representative_timestamp = best["timestamp"],
            cosine_distance          = best["result"].cosine_distance,
            confidence_level         = best["result"].confidence_level,
            ambiguous                = True,
            top5_candidates          = top5,
            consensus_mode           = "sliding",
            frame_distances          = [f["result"].cosine_distance for f in frames],
            frame_timestamps         = [f["timestamp"] for f in frames],
        )

    def _build_event(self) -> "BiologicalEvent":
        all_frames      = self._event_frames
        low_dist_frames = [
            f for f in all_frames
            if f["result"].cosine_distance <= self.threshold
        ] or all_frames

        # Especie ganadora: más votada entre frames de baja distancia
        counts          = Counter(f["result"].species for f in low_dist_frames)
        winner, _       = counts.most_common(1)[0]

        # Frame representativo: menor distancia entre frames del winner con baja dist
        winner_frames   = [f for f in low_dist_frames if f["result"].species == winner]
        best            = min(winner_frames, key=lambda f: f["result"].cosine_distance)

        # start_time: primer frame del evento
        # end_time:   último frame con d <= threshold (antes del cierre)
        start_time = all_frames[0]["timestamp"]
        end_time   = (
            self._last_low_frame["timestamp"]
            if self._last_low_frame else all_frames[-1]["timestamp"]
        )

        frame_distances  = [f["result"].cosine_distance for f in all_frames]
        frame_timestamps = [f["timestamp"]              for f in all_frames]

        names = (
            self._catalog.get_common_names(winner)
            if self._catalog else {"es_ar": "", "en": ""}
        )

        top5 = best["result"].top5_candidates
        if not any(c.get("species") == winner for c in top5):
            top5 = [{"species": winner,
                     "cosine_distance": best["result"].cosine_distance}] + top5

        return BiologicalEvent(
            species                  = winner,
            nombre_comun_es_ar       = names["es_ar"],
            nombre_comun_en          = names["en"],
            start_time               = start_time,
            end_time                 = end_time,
            representative_frame_idx = best["frame_idx"],
            representative_timestamp = best["timestamp"],
            cosine_distance          = best["result"].cosine_distance,
            confidence_level         = best["result"].confidence_level,
            ambiguous                = False,
            top5_candidates          = top5,
            consensus_mode           = "sliding",
            frame_distances          = frame_distances,
            frame_timestamps         = frame_timestamps,
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
    top5_candidates:          list[dict]  = field(default_factory=list)
    consensus_mode:           str         = "static"   # "static" | "sliding"
    frame_distances:          list[float] = field(default_factory=list)
    frame_timestamps:         list[float] = field(default_factory=list)
    rejected_frames:          int         = 0


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
        embedder:       BioCLIPEmbedder,
        classifier:     SpeciesClassifier,
        N:              int  = 30,
        K:              int  = 10,
        M:              int  = 6,
        P:              int  = _SLIDING_P_DEFAULT,
        batch_size:     int  = 8,
        consensus_mode: str  = "static",
    ) -> None:
        self._embedder      = embedder
        self._classifier    = classifier
        self._catalog       = classifier._catalog
        self.N              = N
        self.K              = K
        self.M              = M
        self.P              = P
        self.batch_size     = batch_size
        self.consensus_mode = consensus_mode

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

        frame_predictions:    list[dict]  = []
        rejected_timestamps:  list[float] = []
        batch_imgs:           list[Image.Image] = []
        batch_meta:           list[dict]        = []
        processed = 0

        def _flush() -> None:
            nonlocal processed
            if not batch_imgs:
                return
            embs = self._embedder.embed_batch(batch_imgs)
            for i, meta in enumerate(batch_meta):
                result = self._classifier.classify(embs[i])
                if result.confidence_level == "rechazado":
                    rejected_timestamps.append(meta["timestamp"])
                else:
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
        if self.consensus_mode == "sliding":
            events = self._build_events_sliding(frame_predictions)
        else:
            events = self._build_events(frame_predictions)

        for event in events:
            event.rejected_frames = sum(
                1 for ts in rejected_timestamps
                if event.start_time <= ts <= event.end_time
            )

        return events

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
                consensus_mode           = "static",
                frame_distances          = [w["result"].cosine_distance for w in window],
                frame_timestamps         = [w["timestamp"] for w in window],
            ))

            i += self.K

        return events

    # ------------------------------------------------------------------
    # Construcción de eventos con ventana deslizante
    # ------------------------------------------------------------------

    def _build_events_sliding(self, predictions: list[dict]) -> list[BiologicalEvent]:
        consensus = SlidingWindowConsensus(
            K=self.K, M=self.M, P=self.P,
            threshold=_SLIDING_THRESHOLD,
            catalog=self._catalog,
        )
        events: list[BiologicalEvent] = []
        for pred in predictions:
            ev = consensus.feed(pred)
            if ev is not None:
                events.append(ev)

        # Capturar estado antes de flush (flush llama a _reset y limpia el estado)
        _state_pre  = consensus._state
        _cands_pre  = len(consensus._candidate_frames)

        # Cerrar evento activo o candidatos pendientes al terminar el video
        final = consensus.flush()
        if final is not None:
            events.append(final)

        logger.debug(
            "[_build_events_sliding] frames=%d  state_pre_flush=%s  "
            "candidates_pendientes=%d  flush_emitio=%s  eventos_emitidos=%d",
            len(predictions), _state_pre, _cands_pre,
            "si" if final is not None else "no", len(events),
        )
        return events
