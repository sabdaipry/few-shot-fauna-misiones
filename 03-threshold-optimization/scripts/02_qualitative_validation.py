"""Validación cualitativa de umbrales calibrados del pipeline few-shot de fauna silvestre.

Aplica el pipeline en cascada de dos etapas sobre frames reales de cámaras trampa:

  Etapa 1 — Filtro open-set (DINOv2 Small + coseno, umbral p99 = 0.7594):
      Si d_min_dino > 0.7594 → frame rechazado (vacío o fauna desconocida).

  Etapa 2 — Confianza taxonómica (BioCLIP v2 + coseno centroide, umbral p95 = 0.1866):
      Si d_min_centroid > 0.1866 → predicción de baja confianza.

Carpetas de entrada:
    Carpeta A — mayormente vacíos y fauna fuera del dataset
    Carpeta B — mayormente con fauna, incluye humanos y casos borde

Salidas:
    03-threshold-optimization/data/resultados_validacion.csv
    03-threshold-optimization/data/validacion_visual/  (copia organizada de imágenes)
    03-threshold-optimization/logs/  (log con resumen)

Uso:
    python 03-threshold-optimization/scripts/02_qualitative_validation.py
"""

import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Ajuste de rutas — mismo patrón que 01_calibrate_thresholds.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR        = Path(__file__).resolve().parent   # 03-threshold-optimization/scripts/
_MODULE_ROOT       = _SCRIPT_DIR.parent                # 03-threshold-optimization/
_REPO_ROOT         = _MODULE_ROOT.parent               # raíz del repo
_BENCHMARKING_ROOT = _REPO_ROOT / "02-benchmarking"

sys.path.insert(0, str(_BENCHMARKING_ROOT))
from src.utils.logger import setup_logger
from src.backbones import create_extractor

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_LOG_DIR = _MODULE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger("qualitative_validation", log_dir=_LOG_DIR)

# ---------------------------------------------------------------------------
# Constantes y paths
# ---------------------------------------------------------------------------
DATASET_INDEX_PATH = _BENCHMARKING_ROOT / "data" / "dataset_index.csv"
FEATURES_DIR       = _BENCHMARKING_ROOT / "data" / "features"
OUTPUT_CSV         = _MODULE_ROOT / "data" / "resultados_validacion.csv"
VISUAL_DIR         = _MODULE_ROOT / "data" / "validacion_visual"

FOLDER_A = Path(r"C:\Users\Win 10\Desktop\CEIA\CEIA\TF\Imágenes Cámaras Trampa\Tarjeta 3\105EK113\105EK113")
FOLDER_B = Path(r"C:\Users\Win 10\Desktop\CEIA\CEIA\TF\100RECNX")

EXTENSIONS: tuple[str, ...] = (".JPG", ".jpg", ".PNG", ".png")

THRESHOLD_U1: float = 0.7594   # DINOv2 Small coseno p99
THRESHOLD_U2: float = 0.1866   # BioCLIP v2 coseno centroide p95

SEED: int = 29
MAX_IMAGES_PER_FOLDER: int = 100

FOLDER_A_FORCED_INCLUDES: list[str] = [
    "08040771.JPG", "08040775.JPG", "08040788.JPG", "08040871.JPG", "08040875.JPG",
    "08040879.JPG", "08040902.JPG", "08040947.JPG", "08040960.JPG", "08040981.JPG",
    "08040982.JPG",
]

# Columnas del CSV de salida (excluye _filepath, que es solo interno)
_CSV_COLUMNS: list[str] = [
    "carpeta", "filename", "d_min_dino", "resultado_u1",
    "especie_predicha", "d_min_centroid", "resultado_u2",
]


# ===========================================================================
# SECCIÓN 1 — Carga de embeddings del dataset (gallery + query combinados)
# Copiado directamente de 01_calibrate_thresholds.py — no se importa.
# ===========================================================================

def load_embeddings(
    backbone: str,
    index_df: pd.DataFrame,
    features_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga todos los embeddings .npy de un backbone (gallery + query).

    La ruta de cada archivo se construye como:
        features_dir/{backbone}/{family}/{genus}/{species_underscored}/{stem}.npy

    Args:
        backbone:     Nombre del backbone (subcarpeta en features_dir).
        index_df:     DataFrame del índice del dataset (todos los splits).
        features_dir: Ruta raíz de la carpeta de features.

    Returns:
        Tupla (embeddings float32 shape (N, D), labels array de strings).
    """
    backbone_dir = features_dir / backbone
    embs: list[np.ndarray] = []
    labels: list[str] = []
    missing = 0

    for _, row in index_df.iterrows():
        species_folder = row["species"].replace(" ", "_")
        stem = Path(row["filepath"]).stem
        npy_path = (
            backbone_dir / row["family"] / row["genus"]
            / species_folder / f"{stem}.npy"
        )
        if not npy_path.exists():
            logger.warning(f"No encontrado: {npy_path}")
            missing += 1
            continue

        embs.append(np.load(npy_path).astype(np.float32).ravel())
        labels.append(row["species"])

    if missing:
        logger.warning(f"[{backbone}] {missing} archivos .npy omitidos.")

    return np.vstack(embs), np.array(labels)


def compute_centroids(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula el centroide L2-normalizado de cada especie.

    Promedia los embeddings L2-normalizados por clase y re-normaliza el resultado.

    Args:
        gallery_embs:   Embeddings, shape (M, D).
        gallery_labels: Etiquetas de especie, shape (M,).

    Returns:
        Dict {species: centroid_vector shape (D,)} en float64.
    """
    gallery_normed = normalize(gallery_embs.astype(np.float64), norm="l2")
    centroids: dict[str, np.ndarray] = {}

    for sp in np.unique(gallery_labels):
        mask = gallery_labels == sp
        mean_vec = gallery_normed[mask].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids[sp] = mean_vec / norm if norm > 0 else mean_vec

    return centroids


# ===========================================================================
# SECCIÓN 2 — Cómputo de distancias para frames nuevos
# ===========================================================================

def cosine_distance_to_gallery(
    query_emb: np.ndarray,
    gallery_norm: np.ndarray,
) -> float:
    """Distancia coseno mínima de un embedding query al gallery completo.

    El gallery debe estar pre-normalizado (L2) para eficiencia.

    Args:
        query_emb:    Embedding query, shape (D,), sin normalizar.
        gallery_norm: Gallery L2-normalizado, shape (M, D).

    Returns:
        Distancia coseno mínima float en [0, 2].
    """
    q = query_emb.astype(np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = gallery_norm @ q   # (M,)
    return float(1.0 - sims.max())


def cosine_distance_to_centroids(
    query_emb: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> tuple[str, float]:
    """Distancia coseno a todos los centroides; devuelve el más cercano.

    Args:
        query_emb:  Embedding query, shape (D,), sin normalizar.
        centroids:  Dict {species: centroid_vector L2-normalizado, shape (D,)}.

    Returns:
        Tupla (especie_predicha, d_min_centroid).
    """
    q = query_emb.astype(np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)

    best_species = ""
    best_dist = np.inf

    for sp, c in centroids.items():
        dist = 1.0 - float(q @ c)
        if dist < best_dist:
            best_dist = dist
            best_species = sp

    return best_species, float(best_dist)


# ===========================================================================
# SECCIÓN 3 — Procesamiento por carpeta
# ===========================================================================

def collect_images(
    folder: Path,
    extensions: tuple[str, ...],
    forced_includes: list[str] = [],
) -> list[Path]:
    """Recopila recursivamente todas las imágenes con las extensiones dadas.

    Si el total supera MAX_IMAGES_PER_FOLDER, aplica muestreo aleatorio con
    SEED garantizando que los filenames en forced_includes estén en la muestra
    final (si existen en la carpeta). El resto se completa aleatoriamente.

    Args:
        folder:           Carpeta raíz de búsqueda.
        extensions:       Tupla de extensiones a incluir, ej. (".JPG", ".jpg").
        forced_includes:  Filenames que deben estar en la muestra si existen.

    Returns:
        Lista de Path ordenada por nombre de archivo.
    """
    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix in extensions
    ]
    images = sorted(images, key=lambda p: p.name)

    if len(images) > MAX_IMAGES_PER_FOLDER:
        images_total = len(images)
        rng = random.Random(SEED)
        forced_set = set(forced_includes)
        forced_paths = [p for p in images if p.name in forced_set]
        remaining = [p for p in images if p.name not in forced_set]
        n_fill = max(0, MAX_IMAGES_PER_FOLDER - len(forced_paths))
        sampled = rng.sample(remaining, min(n_fill, len(remaining)))
        images = forced_paths + sampled
        logger.info(
            f"  Muestreo aleatorio: {MAX_IMAGES_PER_FOLDER} de {images_total} imágenes (seed={SEED})"
        )

    return sorted(images, key=lambda p: p.name)


def process_folder(
    folder: Path,
    folder_label: str,
    extractor_dino,
    extractor_bio,
    gallery_dino_norm: np.ndarray,
    centroids_bio: dict[str, np.ndarray],
    threshold_u1: float,
    threshold_u2: float,
    extensions: tuple[str, ...],
    forced_includes: list[str] = [],
) -> list[dict]:
    """Aplica el pipeline en cascada sobre todas las imágenes de una carpeta.

    Por cada imagen:
      - Etapa 1: DINOv2 → d_min_dino. Si > threshold_u1: RECHAZADO.
      - Etapa 2 (solo si pasa U1): BioCLIP → especie más cercana por centroide.

    Cada fila incluye '_filepath' (campo interno, excluido del CSV final)
    para localizar la imagen original en la Sección 6.

    Args:
        folder:            Carpeta de imágenes a procesar.
        folder_label:      Etiqueta para el CSV ("A" o "B").
        extractor_dino:    Extractor DINOv2 Small ya cargado.
        extractor_bio:     Extractor BioCLIP v2 ya cargado.
        gallery_dino_norm: Gallery DINOv2 L2-normalizado, shape (M, D).
        centroids_bio:     Centroides BioCLIP v2 por especie (L2-normalizados).
        threshold_u1:      Umbral open-set (coseno p99 = 0.7594).
        threshold_u2:      Umbral confianza (coseno centroide p95 = 0.1866).
        extensions:        Extensiones de imagen a procesar.
        forced_includes:   Filenames garantizados en la muestra (ver collect_images).

    Returns:
        Lista de dicts con las columnas del CSV más '_filepath'.
    """
    images = collect_images(folder, extensions, forced_includes)
    logger.info(
        f"Carpeta {folder_label} ({folder.name}): {len(images)} imágenes encontradas."
    )

    rows: list[dict] = []

    for img_path in images:
        row: dict = {
            "carpeta":          folder_label,
            "filename":         img_path.name,
            "_filepath":        str(img_path),
            "d_min_dino":       np.nan,
            "resultado_u1":     "",
            "especie_predicha": "N/A",
            "d_min_centroid":   np.nan,
            "resultado_u2":     "N/A",
        }

        # --- Etapa 1: DINOv2 Small ---
        try:
            emb_dino = extractor_dino.get_embedding(str(img_path))
        except Exception as exc:
            logger.warning(f"[{img_path.name}] Error extrayendo DINOv2: {exc} — saltando.")
            rows.append(row)
            continue

        if emb_dino is None:
            logger.warning(f"[{img_path.name}] DINOv2 devolvió None — saltando.")
            rows.append(row)
            continue

        d_min_dino = cosine_distance_to_gallery(emb_dino, gallery_dino_norm)
        row["d_min_dino"] = round(d_min_dino, 6)

        if d_min_dino > threshold_u1:
            row["resultado_u1"] = "RECHAZADO"
            rows.append(row)
            continue

        row["resultado_u1"] = "PASA"

        # --- Etapa 2: BioCLIP v2 ---
        try:
            emb_bio = extractor_bio.get_embedding(str(img_path))
        except Exception as exc:
            logger.warning(
                f"[{img_path.name}] Error extrayendo BioCLIP: {exc} — etapa 2 omitida."
            )
            rows.append(row)
            continue

        if emb_bio is None:
            logger.warning(
                f"[{img_path.name}] BioCLIP devolvió None — etapa 2 omitida."
            )
            rows.append(row)
            continue

        especie_predicha, d_min_centroid = cosine_distance_to_centroids(
            emb_bio, centroids_bio
        )
        row["especie_predicha"] = especie_predicha
        row["d_min_centroid"]   = round(d_min_centroid, 6)
        row["resultado_u2"] = (
            "BAJA_CONFIANZA" if d_min_centroid > threshold_u2 else "ALTA_CONFIANZA"
        )

        rows.append(row)

    return rows


# ===========================================================================
# SECCIÓN 4 — Resumen en log
# ===========================================================================

def log_folder_summary(
    df_folder: pd.DataFrame,
    folder_label: str,
    top_n: int = 5,
) -> None:
    """Loguea el resumen de resultados para una carpeta.

    Args:
        df_folder:    Sub-DataFrame filtrado a una sola carpeta.
        folder_label: Etiqueta de carpeta ("A" o "B").
        top_n:        Número de especies a mostrar en el ranking.
    """
    total      = len(df_folder)
    rechazados = int((df_folder["resultado_u1"] == "RECHAZADO").sum())
    pasan_u1   = int((df_folder["resultado_u1"] == "PASA").sum())
    alta_conf  = int((df_folder["resultado_u2"] == "ALTA_CONFIANZA").sum())
    baja_conf  = int((df_folder["resultado_u2"] == "BAJA_CONFIANZA").sum())

    pct = lambda n: f"{n / total * 100:.1f}%" if total > 0 else "—"

    logger.info(f"--- Carpeta {folder_label} ---")
    logger.info(f"  Total frames procesados : {total}")
    logger.info(f"  Rechazados (U1)         : {rechazados}  ({pct(rechazados)})")
    logger.info(f"  Pasan U1                : {pasan_u1}  ({pct(pasan_u1)})")
    logger.info(f"  Alta confianza (U2)     : {alta_conf}  ({pct(alta_conf)})")
    logger.info(f"  Baja confianza (U2)     : {baja_conf}  ({pct(baja_conf)})")

    pasan_df = df_folder[df_folder["resultado_u1"] == "PASA"]
    if len(pasan_df) > 0:
        top_species = pasan_df["especie_predicha"].value_counts().head(top_n)
        logger.info(f"  Top {top_n} especies predichas (entre los que pasan U1):")
        for sp, cnt in top_species.items():
            logger.info(f"    {sp:<42} {cnt:>4} frames")
    else:
        logger.info("  (Sin frames que pasen U1)")


# ===========================================================================
# SECCIÓN 5 — Organización visual de resultados
# ===========================================================================

def _build_dest_filename(row: pd.Series) -> str:
    """Construye el nombre de archivo destino según el resultado del pipeline.

    Convención:
      - RECHAZADO:       dino{d_min_dino:.4f}_{filename}
      - ALTA_CONFIANZA:  {especie_predicha}_bio{d_min_centroid:.4f}_{filename}
      - BAJA_CONFIANZA:  {especie_predicha}_bio{d_min_centroid:.4f}_{filename}

    Los espacios en el nombre de especie se reemplazan por '_'.

    Args:
        row: Fila del DataFrame con todos los campos del pipeline.

    Returns:
        Nombre de archivo destino como string.
    """
    filename = row["filename"]
    if row["resultado_u1"] == "RECHAZADO":
        return f"dino{row['d_min_dino']:.4f}_{filename}"
    especie = str(row["especie_predicha"]).replace(" ", "_")
    return f"{especie}_bio{row['d_min_centroid']:.4f}_{filename}"


def organize_visual_results(
    df: pd.DataFrame,
    output_base: Path,
) -> None:
    """Copia las imágenes a subcarpetas organizadas por resultado del pipeline.

    Estructura de salida:
        output_base/
          carpeta_A/
            RECHAZADO/
            ALTA_CONFIANZA/
            BAJA_CONFIANZA/
          carpeta_B/
            RECHAZADO/
            ALTA_CONFIANZA/
            BAJA_CONFIANZA/

    Los frames con error de extracción (resultado_u1 vacío) se omiten.
    Usa shutil.copy2 para preservar metadatos de la imagen original.

    Args:
        df:          DataFrame completo, incluyendo la columna '_filepath'.
        output_base: Directorio raíz de salida para la organización visual.
    """
    folder_map: dict[str, str] = {"A": "carpeta_A", "B": "carpeta_B"}
    categories: list[str] = ["RECHAZADO", "ALTA_CONFIANZA", "BAJA_CONFIANZA"]

    # Crear estructura de directorios
    for folder_subdir in folder_map.values():
        for category in categories:
            (output_base / folder_subdir / category).mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    skip_count = 0

    for _, row in df.iterrows():
        resultado_u1 = row["resultado_u1"]
        resultado_u2 = row["resultado_u2"]

        if resultado_u1 == "RECHAZADO":
            category = "RECHAZADO"
        elif resultado_u2 in ("ALTA_CONFIANZA", "BAJA_CONFIANZA"):
            category = resultado_u2
        else:
            # Frame con error de extracción — resultado_u1 es cadena vacía
            skip_count += 1
            continue

        src_path  = Path(row["_filepath"])
        dest_name = _build_dest_filename(row)
        dest_dir  = output_base / folder_map[row["carpeta"]] / category
        dest_path = dest_dir / dest_name

        try:
            shutil.copy2(src_path, dest_path)
            key = f"{row['carpeta']}/{category}"
            counters[key] = counters.get(key, 0) + 1
        except Exception as exc:
            logger.warning(f"[{row['filename']}] Error copiando a {dest_path}: {exc}")

    logger.info("Imágenes copiadas por subcarpeta:")
    for folder_label in ("A", "B"):
        for category in categories:
            key   = f"{folder_label}/{category}"
            count = counters.get(key, 0)
            logger.info(f"  carpeta_{folder_label} / {category:<18}: {count:>4} imágenes")

    if skip_count > 0:
        logger.warning(f"  {skip_count} imágenes omitidas (error de extracción).")


# ===========================================================================
# SECCIÓN 6 — Main
# ===========================================================================

def main() -> None:
    """Ejecuta la validación cualitativa del pipeline few-shot sobre cámaras trampa."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("   VALIDACIÓN CUALITATIVA — PIPELINE FEW-SHOT FAUNA")
    logger.info("=" * 60)
    logger.info(f"Umbral U1 (DINOv2 Small, coseno p99)        : {THRESHOLD_U1}")
    logger.info(f"Umbral U2 (BioCLIP v2, centroide coseno p95): {THRESHOLD_U2}")

    # -----------------------------------------------------------------------
    # 1. Índice del dataset — todos los splits (gallery + query)
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("Cargando índice del dataset...")
    index_df = pd.read_csv(DATASET_INDEX_PATH)
    logger.info(f"  {len(index_df)} registros totales (gallery + query)")

    # -----------------------------------------------------------------------
    # 2. Embeddings DINOv2 Small — todos los splits
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("Cargando embeddings DINOv2 Small (gallery + query)...")
    embs_dino, labels_dino = load_embeddings("dinov2_small", index_df, FEATURES_DIR)
    logger.info(f"  Shape: {embs_dino.shape}")

    # Pre-normalizar para acelerar la búsqueda del vecino más cercano
    gallery_dino_norm = normalize(embs_dino.astype(np.float64), norm="l2")

    # -----------------------------------------------------------------------
    # 3. Embeddings BioCLIP v2 + centroides — todos los splits
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("Cargando embeddings BioCLIP v2 (gallery + query)...")
    embs_bio, labels_bio = load_embeddings("bioclip_v2", index_df, FEATURES_DIR)
    logger.info(f"  Shape: {embs_bio.shape}")

    logger.info("Computando centroides BioCLIP v2...")
    centroids_bio = compute_centroids(embs_bio, labels_bio)
    logger.info(f"  Centroides calculados para {len(centroids_bio)} especies.")

    # -----------------------------------------------------------------------
    # 4. Cargar modelos
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("Cargando modelos...")
    extractor_dino = create_extractor("dinov2_small", device="cpu")
    extractor_bio  = create_extractor("bioclip_v2",   device="cpu")

    # -----------------------------------------------------------------------
    # 5. Procesar carpetas
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("Procesando Carpeta A...")
    rows_a = process_folder(
        folder            = FOLDER_A,
        folder_label      = "A",
        extractor_dino    = extractor_dino,
        extractor_bio     = extractor_bio,
        gallery_dino_norm = gallery_dino_norm,
        centroids_bio     = centroids_bio,
        threshold_u1      = THRESHOLD_U1,
        threshold_u2      = THRESHOLD_U2,
        extensions        = EXTENSIONS,
        forced_includes   = FOLDER_A_FORCED_INCLUDES,
    )

    logger.info("")
    logger.info("Procesando Carpeta B...")
    rows_b = process_folder(
        folder            = FOLDER_B,
        folder_label      = "B",
        extractor_dino    = extractor_dino,
        extractor_bio     = extractor_bio,
        gallery_dino_norm = gallery_dino_norm,
        centroids_bio     = centroids_bio,
        threshold_u1      = THRESHOLD_U1,
        threshold_u2      = THRESHOLD_U2,
        extensions        = EXTENSIONS,
    )

    # DataFrame completo — incluye '_filepath' para la organización visual
    df = pd.DataFrame(rows_a + rows_b)

    # -----------------------------------------------------------------------
    # 6. Guardar CSV (sin '_filepath')
    # -----------------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[_CSV_COLUMNS].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"\nCSV guardado: {OUTPUT_CSV}  ({len(df)} filas)")

    # -----------------------------------------------------------------------
    # 7. Organización visual de resultados
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("Organizando imágenes en carpetas por resultado...")
    organize_visual_results(df, VISUAL_DIR)
    logger.info(f"Directorio de salida: {VISUAL_DIR}")

    # -----------------------------------------------------------------------
    # 8. Resumen final
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("   RESUMEN FINAL")
    logger.info("=" * 60)

    for folder_label in ("A", "B"):
        df_folder = df[df["carpeta"] == folder_label]
        log_folder_summary(df_folder, folder_label)
        logger.info("")

    logger.info("VALIDACIÓN COMPLETADA")


if __name__ == "__main__":
    main()
