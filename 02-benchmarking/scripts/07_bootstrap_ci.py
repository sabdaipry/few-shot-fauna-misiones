"""Calcula intervalos de confianza (IC) bootstrap al 95% sobre las predicciones ya
generadas por el pipeline de benchmarking (predictions_*.csv), sin reejecutar
extracción de embeddings ni clasificación.

Para cada combinación Backbone x Clasificador x Métrica (Accuracy, F1-macro) se
genera un IC bootstrap **estratificado por clase**: en cada iteración se resamplea
con reemplazo *dentro* de cada especie, preservando su n original, en lugar de
resamplear las 3674 filas del query set de forma global.

Limitaciones metodológicas (documentadas para el reporte académico):

1. Solo varianza de muestreo del query set: el IC responde "¿cuánto cambiaría la
   métrica con otra muestra de imágenes de consulta?", no captura la incertidumbre
   de re-elegir el support/gallery set few-shot (fijo en este benchmark) ni la del
   extractor de embeddings en sí (ambos son determinísticos aquí).
2. Comparaciones apareadas no explotadas: las 19 predictions_*.csv comparten
   exactamente el mismo orden de filas/idx (mismo query set), por lo que comparar
   sus ICs marginales para decidir "A es mejor que B" es conservador; un test
   riguroso requeriría bootstrapear la diferencia con los mismos índices de resample.
3. Clases singleton sin varianza real: 4 especies tienen una sola muestra en el
   query set; con bootstrap estratificado esa muestra se repite siempre, así que su
   contribución a F1-macro es constante entre iteraciones y angosta el IC reportado.
4. Clasificadores correlacionados: los 7 clasificadores de un mismo backbone
   comparten embeddings y gallery, por lo que sus errores no son independientes
   entre sí (su CI individual no captura esa correlación).

Uso:
    python scripts/07_bootstrap_ci.py [--iterations 1000] [--seed 29] [--confidence 0.95]
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Ajuste de rutas para imports relativos
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.config import BENCHMARK_RESULTS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("bootstrap_ci")


def build_class_index_map(y_true):
    """Mapea cada clase a las posiciones (índices) que ocupa en y_true."""
    indices = np.arange(len(y_true))
    return {cls: indices[y_true == cls] for cls in np.unique(y_true)}


def stratified_bootstrap_resample(class_index_map, n_total, rng):
    """Genera un array de índices resampleados con reemplazo, estratificado por
    clase: para cada clase se resamplea (con reemplazo) el mismo n de muestras
    que tiene originalmente, preservando la composición del query set."""
    resampled = np.empty(n_total, dtype=int)
    pos = 0
    for cls_idx in class_index_map.values():
        n = len(cls_idx)
        resampled[pos:pos + n] = rng.choice(cls_idx, size=n, replace=True)
        pos += n
    return resampled


def compute_bootstrap_ci(y_true, y_pred, labels, bootstrap_idx, confidence):
    """Calcula media e IC (método percentil) de Accuracy y F1-macro sobre las
    iteraciones bootstrap ya resampleadas en bootstrap_idx."""
    n_iterations = len(bootstrap_idx)
    accs = np.empty(n_iterations)
    f1s = np.empty(n_iterations)

    for i, idx in enumerate(bootstrap_idx):
        yt, yp = y_true[idx], y_pred[idx]
        accs[i] = accuracy_score(yt, yp)
        # labels fijo: mantiene el denominador de F1-macro constante entre
        # iteraciones aunque el resampleo altere qué filas de cada clase aparecen.
        f1s[i] = f1_score(yt, yp, labels=labels, average='macro', zero_division=0)

    alpha = 1 - confidence
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

    results = {}
    for metric_name, arr in [('Accuracy', accs), ('F1_Macro', f1s)]:
        mean = float(arr.mean())
        ci_lo, ci_hi = (float(v) for v in np.percentile(arr, [lo_pct, hi_pct]))
        results[metric_name] = (mean, ci_lo, ci_hi, ci_hi - ci_lo)
    return results


def main(args):
    logger.info("==============================================")
    logger.info("   FASE 7: BOOTSTRAP CI (95%) SOBRE PREDICCIONES")
    logger.info("==============================================")

    pred_files = sorted(BENCHMARK_RESULTS_DIR.glob("predictions_*.csv"))
    if not pred_files:
        logger.error(f"No se encontraron predictions_*.csv en {BENCHMARK_RESULTS_DIR}")
        return

    # Referencia: todas las predictions_*.csv comparten el mismo query set
    # (mismo orden de filas/idx/y_true), verificado en el diagnóstico previo.
    df_ref = pd.read_csv(pred_files[0])
    y_true = df_ref['y_true'].to_numpy()
    labels = np.unique(y_true)
    n_total = len(y_true)
    logger.info(f"Query set de referencia: {pred_files[0].name} ({n_total} filas, {len(labels)} clases)")

    rng = np.random.default_rng(args.seed)
    class_index_map = build_class_index_map(y_true)
    logger.info(f"Generando {args.iterations} resamples bootstrap estratificados (seed={args.seed})...")
    bootstrap_idx = [
        stratified_bootstrap_resample(class_index_map, n_total, rng)
        for _ in range(args.iterations)
    ]

    # Recolectar combinaciones Backbone x Clasificador (sin Faiss, ver diagnóstico).
    combos = []
    for pred_path in pred_files:
        model_name = pred_path.stem.replace("predictions_", "")
        df_pred = pd.read_csv(pred_path)

        if not np.array_equal(df_pred['y_true'].to_numpy(), y_true):
            logger.warning(f"{pred_path.name}: y_true difiere del query set de referencia, se omite.")
            continue

        pred_cols = [c for c in df_pred.columns if c.startswith('pred_') and 'Faiss' not in c]
        for col in pred_cols:
            clf_name = col.replace('pred_', '')
            combos.append((model_name, clf_name, df_pred[col].to_numpy()))

    logger.info(f"{len(combos)} combinaciones Backbone x Clasificador a evaluar.")

    records = []
    for model_name, clf_name, y_pred in tqdm(combos, desc="Bootstrap CI"):
        ci_results = compute_bootstrap_ci(y_true, y_pred, labels, bootstrap_idx, args.confidence)
        for metric_name, (mean, ci_lo, ci_hi, ci_width) in ci_results.items():
            records.append({
                'Embedding Model': model_name,
                'Classifier': clf_name,
                'Metric': metric_name,
                'Mean': mean,
                'CI_lower': ci_lo,
                'CI_upper': ci_hi,
                'CI_width': ci_width,
            })

    df_out = pd.DataFrame(records)
    out_path = BENCHMARK_RESULTS_DIR / "bootstrap_ci.csv"
    df_out.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path} ({len(df_out)} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CI (95%) sobre predictions_*.csv existentes.")
    parser.add_argument("--iterations", type=int, default=1000, help="Número de resamples bootstrap (default: 1000)")
    parser.add_argument("--seed", type=int, default=29, help="Semilla aleatoria (default: 29)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Nivel de confianza del IC (default: 0.95)")
    main(parser.parse_args())
