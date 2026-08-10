"""Bootstrap CI (95%) desagregado por categoría de Índice de Valor de Conservación
(IVC), para todos los backbones estándar (15, sin variantes _gap), sobre las
predicciones ya guardadas (predictions_<backbone>.csv en query_dev). No
recalcula embeddings ni reentrena nada.

Replica exactamente la metodología de la figura 07_ivc_ranking (src/analysis.py
::analyze_ivc_performance + summarize_ivc_performance_by_backbone, invocada desde
06_generate_full_report.py): el "desempeño predictivo normalizado" por categoría
IVC es el promedio NO ponderado de la accuracy de los 7 clasificadores clásicos
(Nearest Centroid, KNN k=1/3/5, Linear SVM, RBF SVM, Random Forest) dentro de esa
categoría — no un único clasificador ni una accuracy pooleada sobre las filas.

Las 4 variantes _gap (dinov2_*_gap, dinov3_*_gap) quedan excluidas: sus
predictions_*.csv tienen 3674 filas (query set completo, pre-separación
dev/test) en vez de las 2572 de query_dev — no comparten índice/orden con el
resto. Es la misma exclusión que ya aplica el resto del pipeline de
benchmarking.

Reutiliza la implementación de bootstrap estratificado por especie de
07_bootstrap_ci.py (build_class_index_map, stratified_bootstrap_resample) sin
reimplementarla — mismo esquema de resampleo, misma semilla, mismas
iteraciones. Todos los backbones comparten el mismo query_dev fila a fila, por
lo que se reutiliza el mismo conjunto de resamples para todos (habilita
bootstrap pareado entre cualquier par de modelos, si se lo necesita después).

También reporta, sobre los datos crudos (sin bootstrap), la dispersión entre
los 7 clasificadores por combinación modelo x categoría IVC (mínimo, máximo,
rango, desvío estándar, y qué clasificador alcanza cada extremo).

Salidas: data/benchmark_results/ivc_bootstrap_ci.csv y .md.

Uso:
    python scripts/09_ivc_bootstrap_ci.py [--iterations 1000] [--seed 29] [--confidence 0.95]
"""
import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Ajuste de rutas para imports relativos
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.config import BENCHMARK_RESULTS_DIR, MODELS_TO_TEST
from src.utils.logger import setup_logger

logger = setup_logger("ivc_bootstrap_ci")

# Mismo colapso de categorías que src/analysis.py::analyze_ivc_performance
# (normalize_cat), para que las categorías reportadas acá coincidan exactamente
# con las de la figura 07_ivc_ranking.
EXOTIC_CATEGORIES = {'Doméstica', 'Exótica', 'Invasora', 'Exótica/Invasora', 'Introducida'}
CATEGORY_ORDER = ["Crítico", "Alto", "Medio", "Bajo", "Nulo"]
COMBINED_LABEL = "Medio+Bajo (combinado)"


def _import_bootstrap_module():
    """Carga 07_bootstrap_ci.py como módulo (el nombre empieza con dígito, no se
    puede usar `import` normal) para reutilizar sus funciones de resampleo."""
    module_path = current_script_path.parent / "07_bootstrap_ci.py"
    spec = importlib.util.spec_from_file_location("bootstrap_ci_07", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_category(raw_cat):
    """Colapsa categorías de especies exóticas/introducidas a 'Nulo' (igual que
    src/analysis.py::analyze_ivc_performance.normalize_cat)."""
    c = str(raw_cat).strip()
    return 'Nulo' if c in EXOTIC_CATEGORIES else c


def load_model_predictions(model_name):
    """Carga predictions_<model_name>.csv y arma arrays y_true / categoría
    normalizada / predicciones por clasificador."""
    path = BENCHMARK_RESULTS_DIR / f"predictions_{model_name}.csv"
    df = pd.read_csv(path)

    clf_cols = [c for c in df.columns if c.startswith('pred_') and 'Faiss' not in c]
    classifiers = [c.replace('pred_', '') for c in clf_cols]

    y_true = df['y_true'].to_numpy()
    cat_norm = df['ivc_category'].apply(normalize_category).to_numpy()
    preds = {clf: df[col].to_numpy() for clf, col in zip(classifiers, clf_cols)}

    return {
        'y_true': y_true,
        'cat_norm': cat_norm,
        'preds': preds,
        'classifiers': classifiers,
    }


def category_mask(cat_norm_arr, category):
    """Máscara booleana para una categoría, soportando la pseudo-categoría
    'Medio+Bajo (combinado)' (unión de filas Medio y Bajo)."""
    if category == COMBINED_LABEL:
        return np.isin(cat_norm_arr, ['Medio', 'Bajo'])
    return cat_norm_arr == category


def mean_of_classifiers_accuracy(y_true, preds, classifiers, mask):
    """Accuracy por clasificador dentro de `mask`, y su media no ponderada
    (== metodología de la figura 07_ivc_ranking). Retorna (mean, [acc_por_clf])."""
    accs = []
    for clf in classifiers:
        yt, yp = y_true[mask], preds[clf][mask]
        accs.append(float((yt == yp).mean()) * 100.0 if mask.any() else np.nan)
    return float(np.mean(accs)), accs


def compute_dispersion(classifiers, accs):
    """Dispersión cruda (sin bootstrap) entre clasificadores: min, max, rango,
    desvío estándar (muestral), y qué clasificador toca cada extremo."""
    arr = np.array(accs)
    i_min, i_max = int(np.argmin(arr)), int(np.argmax(arr))
    return {
        'disp_min': float(arr[i_min]),
        'disp_max': float(arr[i_max]),
        'disp_range': float(arr[i_max] - arr[i_min]),
        'disp_std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'disp_min_classifier': classifiers[i_min],
        'disp_max_classifier': classifiers[i_max],
    }


def main(args):
    logger.info("==============================================")
    logger.info("   BOOTSTRAP CI (95%) POR CATEGORÍA IVC")
    logger.info("==============================================")

    bootstrap_mod = _import_bootstrap_module()
    build_class_index_map = bootstrap_mod.build_class_index_map
    stratified_bootstrap_resample = bootstrap_mod.stratified_bootstrap_resample

    # Descubrir todos los backbones estándar disponibles (excluye _gap: split
    # distinto, ver docstring). Orden canónico tomado de src.config.MODELS_TO_TEST.
    candidate_names = [m for m in MODELS_TO_TEST if '_gap' not in m]

    models = {}
    ref = None
    for name in candidate_names:
        pred_path = BENCHMARK_RESULTS_DIR / f"predictions_{name}.csv"
        if not pred_path.exists():
            logger.warning(f"{name}: no se encontró {pred_path.name}, se omite.")
            continue
        m = load_model_predictions(name)
        if ref is None:
            ref = m
            models[name] = m
            continue
        if not np.array_equal(m['y_true'], ref['y_true']) or not np.array_equal(m['cat_norm'], ref['cat_norm']):
            logger.warning(f"{name}: y_true/ivc_category no coincide con el query_dev de referencia — se omite.")
            continue
        models[name] = m

    n_total = len(ref['y_true'])
    logger.info(f"query_dev: {n_total} filas, {len(models)} backbones alineados: {list(models.keys())}")

    # Resampleo estratificado por especie — un único set de 1000 resamples,
    # reutilizado para todos los modelos.
    rng = np.random.default_rng(args.seed)
    class_index_map = build_class_index_map(ref['y_true'])
    logger.info(f"Generando {args.iterations} resamples bootstrap estratificados (seed={args.seed})...")
    bootstrap_idx = [
        stratified_bootstrap_resample(class_index_map, n_total, rng)
        for _ in range(args.iterations)
    ]

    alpha = 1 - args.confidence
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

    categories = CATEGORY_ORDER + [COMBINED_LABEL]

    # Por modelo x categoría: n, puntual (media de 7 clasificadores), IC 95%
    # bootstrap, y dispersión cruda entre clasificadores.
    table_rows = []

    for model_name, m in models.items():
        y_true, cat_norm, preds, classifiers = m['y_true'], m['cat_norm'], m['preds'], m['classifiers']

        for category in categories:
            mask_raw = category_mask(cat_norm, category)
            n_images = int(mask_raw.sum())
            n_species = int(pd.unique(y_true[mask_raw]).shape[0]) if n_images > 0 else 0

            point_mean, raw_accs = mean_of_classifiers_accuracy(y_true, preds, classifiers, mask_raw)
            disp = compute_dispersion(classifiers, raw_accs)

            # Distribución bootstrap de la media-de-7 dentro de la categoría.
            # El tamaño de la categoría es invariante entre iteraciones porque el
            # resampleo es estratificado POR ESPECIE y cada especie pertenece a una
            # única categoría IVC (así que el conteo por categoría no cambia, solo
            # qué imágenes de cada especie se repiten).
            iter_vals = np.empty(args.iterations)
            for i, idx in enumerate(bootstrap_idx):
                yt_r = y_true[idx]
                mask_r = category_mask(cat_norm[idx], category)
                accs_i = [
                    (yt_r[mask_r] == preds[clf][idx][mask_r]).mean() * 100.0
                    for clf in classifiers
                ]
                iter_vals[i] = np.mean(accs_i)

            ci_lo, ci_hi = np.percentile(iter_vals, [lo_pct, hi_pct])

            table_rows.append({
                'Embedding Model': model_name,
                'Category': category,
                'n_images': n_images,
                'n_species': n_species,
                'Accuracy_mean7_pct': round(point_mean, 2),
                'CI_lower': round(float(ci_lo), 2),
                'CI_upper': round(float(ci_hi), 2),
                'Disp_min_pct': round(disp['disp_min'], 2),
                'Disp_max_pct': round(disp['disp_max'], 2),
                'Disp_range_pp': round(disp['disp_range'], 2),
                'Disp_std_pp': round(disp['disp_std'], 2),
                'Disp_min_classifier': disp['disp_min_classifier'],
                'Disp_max_classifier': disp['disp_max_classifier'],
            })
            logger.info(
                f"{model_name} | {category}: n={n_images} ({n_species} spp) "
                f"mean7={point_mean:.2f}% CI[{ci_lo:.2f}, {ci_hi:.2f}] "
                f"disp_range={disp['disp_range']:.2f}pp (min={disp['disp_min_classifier']}, max={disp['disp_max_classifier']})"
            )

    df_table = pd.DataFrame(table_rows)
    csv_path = BENCHMARK_RESULTS_DIR / "ivc_bootstrap_ci.csv"
    df_table.to_csv(csv_path, index=False)
    logger.info(f"Guardado: {csv_path} ({len(df_table)} filas)")

    md_cols = [
        'Embedding Model', 'Category', 'n_images', 'n_species',
        'Accuracy_mean7_pct', 'CI_lower', 'CI_upper',
        'Disp_min_pct', 'Disp_max_pct', 'Disp_range_pp', 'Disp_std_pp',
        'Disp_min_classifier', 'Disp_max_classifier',
    ]
    header = "| " + " | ".join(md_cols) + " |"
    sep = "|" + "|".join(["---"] * len(md_cols)) + "|"
    lines = [header, sep]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[c]) for c in md_cols) + " |")
    md_table = "\n".join(lines)

    md_path = BENCHMARK_RESULTS_DIR / "ivc_bootstrap_ci.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_table + "\n")
    logger.info(f"Guardado: {md_path}")

    print("\n" + md_table + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CI (95%) por categoría IVC sobre predictions_*.csv existentes.")
    parser.add_argument("--iterations", type=int, default=1000, help="Número de resamples bootstrap (default: 1000)")
    parser.add_argument("--seed", type=int, default=29, help="Semilla aleatoria (default: 29)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Nivel de confianza del IC (default: 0.95)")
    main(parser.parse_args())
