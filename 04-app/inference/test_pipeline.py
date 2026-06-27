"""
Script de prueba del pipeline de inferencia.

Uso:
    python test_pipeline.py <ruta_video> [N] [K] [M]

Argumentos:
    ruta_video  Ruta al archivo de video a procesar.
    N           Submuestreo: tomar 1 frame cada N (default: 30).
    K           Tamaño de ventana de consenso en frames (default: 10).
    M           Quórum mínimo de coincidencias (default: 6).
"""

import sys
import time
from pathlib import Path

# Agregar 04-app al path para resolver la importación
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.pipeline import (
    BioCLIPEmbedder,
    CatalogManager,
    SpeciesClassifier,
    VideoProcessor,
)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: no existe '{video_path}'")
        sys.exit(1)

    N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    M = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    print("=" * 62)
    print("  PIPELINE FEW-SHOT — PRUEBA DE INFERENCIA")
    print("=" * 62)
    print(f"  Video : {video_path.name}")
    print(f"  Params: N={N} (submuestreo)  K={K} (ventana)  M={M} (quórum)")
    print()

    # ------------------------------------------------------------------
    # 1. Catálogo
    # ------------------------------------------------------------------
    print("[1/3] Cargando catálogo de especies...")
    t0 = time.perf_counter()
    catalog = CatalogManager()
    t_catalog = time.perf_counter() - t0
    print(f"      {len(catalog.get_species_list())} especies  ({t_catalog:.1f}s)\n")

    # ------------------------------------------------------------------
    # 2. Modelo
    # ------------------------------------------------------------------
    print("[2/3] Cargando BioCLIP v2...")
    t0 = time.perf_counter()
    embedder = BioCLIPEmbedder()
    t_model = time.perf_counter() - t0
    print(f"      Modelo listo  ({t_model:.1f}s)\n")

    # ------------------------------------------------------------------
    # 3. Procesamiento de video
    # ------------------------------------------------------------------
    print("[3/3] Procesando video...")
    classifier = SpeciesClassifier(catalog)
    processor  = VideoProcessor(embedder, classifier, N=N, K=K, M=M)

    processed = [0]
    total_est = [1]

    def progress(done: int, tot: int) -> None:
        processed[0] = done
        total_est[0] = tot
        pct = int(done / max(tot, 1) * 100)
        print(f"\r      {pct:3d}%  ({done}/{tot} frames muestreados)", end="", flush=True)

    t0 = time.perf_counter()
    events = processor.process(video_path, progress_callback=progress)
    t_video = time.perf_counter() - t0
    n_frames = processed[0]
    print(f"\r      100%  ({n_frames} frames muestreados)          ")
    print(f"      Tiempo total: {t_video:.1f}s", end="")
    if n_frames > 0:
        print(f"  ({t_video / n_frames * 1000:.0f} ms/frame)", end="")
    print("\n")

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    print("=" * 62)
    print(f"  EVENTOS DETECTADOS: {len(events)}")
    print("=" * 62)

    for i, ev in enumerate(events, 1):
        estado = "AMBIGUO   " if ev.ambiguous else "CONFIRMADO"
        print(f"\nEvento #{i}  [{estado}]")
        print(f"  Especie      : {ev.species}")
        print(f"  Nombre ES-AR : {ev.nombre_comun_es_ar}")
        print(f"  Nombre EN    : {ev.nombre_comun_en}")
        print(f"  Intervalo    : {fmt_time(ev.start_time)} – {fmt_time(ev.end_time)}")
        print(f"  Frame repr.  : #{ev.representative_frame_idx}  "
              f"({fmt_time(ev.representative_timestamp)})")
        print(f"  Confianza    : {ev.confidence_level}  "
              f"(d_coseno = {ev.cosine_distance:.4f})")
        if ev.top5_candidates:
            print("  Top-5 candidatos:")
            for c in ev.top5_candidates:
                marker = " <--" if c["species"] == ev.species else ""
                print(f"    {c['species']:<40}  d={c['cosine_distance']:.4f}{marker}")

    # ------------------------------------------------------------------
    # Resumen de tiempos
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print("  TIEMPOS DE EJECUCIÓN")
    print(f"{'=' * 62}")
    print(f"  Catálogo  : {t_catalog:.2f}s")
    print(f"  Modelo    : {t_model:.2f}s")
    print(f"  Video     : {t_video:.2f}s  ({n_frames} frames)")
    if n_frames > 0:
        print(f"  Por frame : {t_video / n_frames * 1000:.0f} ms")
    print()


if __name__ == "__main__":
    main()
