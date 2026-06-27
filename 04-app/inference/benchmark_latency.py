"""
Benchmark de latencia de extracción de embeddings por batch en BioCLIP v2.

Prueba batch_size en {1, 2, 4, 8, 16} sobre un video real de cámara trampa.
Mide el tiempo de embedding (cuello de botella), con frames pre-extraídos en memoria.

Reporta por batch_size:
  - Tiempo total de embedding
  - ms/frame promedio
  - Factor respecto a duración real del video (< 1x = más rápido que tiempo real)
  - Speedup respecto a batch_size=1

Uso:
    python benchmark_latency.py [ruta_video] [N] [n_runs]

Argumentos opcionales:
    ruta_video  Ruta al video (default: hardcoded abajo).
    N           Submuestreo temporal (default: 30).
    n_runs      Repeticiones por batch_size para la mediana (default: 3).
"""

import statistics
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.pipeline import BioCLIPEmbedder

_DEFAULT_VIDEO = (
    r"C:\Users\Win 10\Desktop\CEIA\CEIA\TF\videos prueba\05\lv_0_20260209055215.mp4"
)
_BATCH_SIZES = [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, N: int) -> tuple[list[Image.Image], float]:
    """Extrae los frames muestreados del video en memoria. Devuelve (frames, duración_s)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    frames: list[Image.Image] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % N == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1

    cap.release()
    return frames, duration_s


def time_embed_batch(
    embedder: BioCLIPEmbedder,
    frames:   list[Image.Image],
    batch_size: int,
) -> float:
    """Tiempo total de una pasada de embedding sobre todos los frames (segundos)."""
    t0 = time.perf_counter()
    for i in range(0, len(frames), batch_size):
        embedder.embed_batch(frames[i : i + batch_size])
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_VIDEO
    N          = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    n_runs     = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print("=" * 72)
    print("  BENCHMARK — LATENCIA DE EMBEDDING POR BATCH  (BioCLIP v2 / CPU)")
    print("=" * 72)
    print(f"  Video  : {Path(video_path).name}")
    print(f"  N      : {N}  (1 frame muestreado cada {N} del video original)")
    print(f"  Runs   : {n_runs} mediciones por batch_size — se reporta la mediana")
    print()

    print("Extrayendo frames...", end=" ", flush=True)
    frames, duration_s = extract_frames(video_path, N)
    n_frames = len(frames)
    print(f"{n_frames} frames  ({duration_s:.1f} s de video)\n")

    print("Cargando BioCLIP v2...", end=" ", flush=True)
    embedder = BioCLIPEmbedder()
    print("listo\n")

    # Warm-up general antes de medir
    embedder.embed_batch(frames[:1])

    results: dict[int, float] = {}
    for bs in _BATCH_SIZES:
        times = [time_embed_batch(embedder, frames, bs) for _ in range(n_runs)]
        median_t = statistics.median(times)
        results[bs] = median_t
        runs_str = "  ".join(f"{t:.2f}s" for t in times)
        print(f"  batch_size={bs:>2}:  mediana={median_t:.2f}s  [{runs_str}]")

    # ------------------------------------------------------------------
    # Tabla de resultados
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(
        f"{'batch_size':>12}  {'T. total (s)':>13}  "
        f"{'ms/frame':>10}  {'Factor vs real':>15}  {'Speedup vs 1':>13}"
    )
    print("-" * 72)
    baseline = results[1]
    for bs in _BATCH_SIZES:
        t       = results[bs]
        ms_fr   = t * 1000 / n_frames
        factor  = t / duration_s
        speedup = baseline / t
        print(
            f"{bs:>12}  {t:>13.2f}  "
            f"{ms_fr:>10.0f}  {factor:>14.2f}x  {speedup:>12.2f}x"
        )
    print("=" * 72)
    print()
    print(f"  {n_frames} frames muestreados  |  N={N}  |  duración real: {duration_s:.1f} s")
    print("  Factor < 1x = procesa más rápido que tiempo real.")
    print()


if __name__ == "__main__":
    main()
