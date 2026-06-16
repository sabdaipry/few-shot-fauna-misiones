"""Genera el reporte HTML interactivo final consolidando métricas, gráficos y UMAPs
de todos los experimentos del pipeline de benchmarking."""
import sys
import os
import gc
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import yaml

# Ajuste de rutas para imports relativos
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.benchmarking import ModelEvaluator
from src.utils.logger import setup_logger
from src.config import (
    DATASET_INDEX_PATH, FEATURES_DIR, BENCHMARK_RESULTS_DIR, REPORTS_DIR,
    BACKBONES_TIMES_PATH, SCALABILITY_RESULTS_PATH,
    INCREMENTAL_RESULTS_PATH, OUTLIER_RESULTS_PATH,
)
import src.visualization as viz
import src.reporting as rep
import src.analysis as analysis

logger = setup_logger("report_orchestrator")

def clean_filename(name):
    """Convierte nombres sucios a slug seguro para archivos."""
    # Ej: "KNN (k=1)" -> "knn_k1", "Linear SVM" -> "linear_svm"
    slug = name.lower()
    slug = slug.replace(' (', '_').replace(')', '').replace('=', '')
    slug = slug.replace(' ', '_')
    return slug

def main():
    """Orquesta la generación de todos los gráficos y del reporte HTML interactivo final."""
    logger.info("==============================================")
    logger.info("   FASE 6: GENERACIÓN DE REPORTE FINAL")
    logger.info("==============================================")
    
    # 1. Configurar Directorios
    REPORT_DIR = REPORTS_DIR
    ASSETS_DIR = REPORT_DIR / "assets"
    FIG_DIR = REPORT_DIR / "figures"
    UMAP_DIR = FIG_DIR / "umaps"
    ERR_DIR = FIG_DIR / "errors"
    IVC_DIR = FIG_DIR / "ivc"

    for d in [REPORT_DIR, ASSETS_DIR, FIG_DIR, UMAP_DIR, ERR_DIR, IVC_DIR]: d.mkdir(parents=True, exist_ok=True)

    # 2. Cargar Datos y Estadísticas
    logger.info("Cargando datos...")
    
    data = {}
    try:
        data['summary'] = pd.read_csv(BENCHMARK_RESULTS_DIR / "benchmark_summary.csv")
        data['times'] = pd.read_csv(BACKBONES_TIMES_PATH)
        data['scalability'] = pd.read_csv(SCALABILITY_RESULTS_PATH)
        data['incremental'] = pd.read_csv(INCREMENTAL_RESULTS_PATH)
        data['outliers'] = pd.read_csv(OUTLIER_RESULTS_PATH)

        # Bootstrap CI (Fase 7) — opcional: el script 07_bootstrap_ci.py se corre
        # por separado (no forma parte de este pipeline) y puede no haberse
        # ejecutado todavía, por lo que se carga con guard.
        bootstrap_ci_path = BENCHMARK_RESULTS_DIR / "bootstrap_ci.csv"
        if bootstrap_ci_path.exists():
            data['bootstrap_ci'] = pd.read_csv(bootstrap_ci_path)
        else:
            logger.warning(
                f"No se encontró {bootstrap_ci_path.name}; se omite la figura de "
                "bootstrap CI. Corré `python scripts/07_bootstrap_ci.py` antes para incluirla."
            )
            data['bootstrap_ci'] = None

        # Mapa taxonómico familia → clase (para análisis de errores cross-clase)
        tax_yaml_path = project_root / "data" / "taxonomic_class_mapping.yaml"
        family_to_class = {}
        if tax_yaml_path.exists():
            with open(tax_yaml_path, 'r', encoding='utf-8') as f:
                class_to_families = yaml.safe_load(f)
            for cls, families in class_to_families.items():
                for fam in (families or []):
                    family_to_class[fam] = cls

        # Mapa de Familias
        df_index = pd.read_csv(DATASET_INDEX_PATH)

        # Cargar estadisticas del dataset
        stats = {}
        # A. Desde statistics.json (si existe)
        json_path = Path("data/fauna_seleccionada_bosque_atlantico/statistics.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                summary = json_data.get('summary', {})
                
                stats['total_species'] = summary.get('total_species', 0)
                stats['total_families'] = summary.get('total_families', 0)
                stats['total_images'] = len(df_index)
        
        # B. Desde dataset_index.csv (Split Query/Gallery)
        if 'split' in df_index.columns:
            counts = df_index['split'].value_counts()
            stats['query_count'] = counts.get('query', 0)
            stats['gallery_count'] = counts.get('gallery', 0)
            # Fallback si statistics.json falló
            if 'total_images' not in stats: stats['total_images'] = len(df_index)
            if 'total_species' not in stats: 
                col = 'species' if 'species' in df_index.columns else 'class_name'
                stats['total_species'] = df_index[col].nunique()


    except Exception as e:
        logger.error(f"Error cargando CSVs: {e}")
        return
    
    # 3. Generar Gráficos Generales (Descartando GAPs donde corresponda)
    # i. Gráficos de Rendimiento (Ranking, Heatmap)
    logger.info("Generando gráficos de rendimiento...")
    viz.plot_leaderboard(data['summary'], "F1_Macro", FIG_DIR / "01_leaderboard_f1.png")
    viz.plot_leaderboard(data['summary'], "Accuracy", FIG_DIR / "01_leaderboard_acc.png")
    viz.plot_heatmap(data['summary'], "F1_Macro", FIG_DIR / "04_heatmap_f1.png")
    viz.plot_heatmap(data['summary'], "Accuracy", FIG_DIR / "04_heatmap_acc.png")
    
    # ii. Eficiencia (Pareto, Latencia Barras)
    logger.info("Generando gráficos de eficiencia...")
    viz.plot_pareto(data['summary'], data['times'], "Accuracy", FIG_DIR / "02_pareto_acc.png")
    viz.plot_pareto(data['summary'], data['times'], "F1_Macro", FIG_DIR / "02_pareto_f1.png")
    viz.plot_backbone_latency(data['times'], FIG_DIR / "latency_backbone.png")
    viz.plot_classifier_latency(data['summary'], FIG_DIR / "latency_classifier.png")
    
    # iii. Métricas Intrínsecas (Calidad Embeddings)
    logger.info("Generando métricas de clustering...")
    viz.plot_embedding_metrics(data['summary'], 
                               metrics=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
                               output_path_prefix=str(FIG_DIR / "emb_metric"))
    
    # iv. Comparativas Específicas
    logger.info("Generando DINO y Escalabilidad...")
    viz.plot_dino_comparison(data['summary'], "Accuracy", FIG_DIR / "05_dino_comparison_acc.png")
    viz.plot_scalability_curves(data['scalability'], FIG_DIR / "03_scalability.png")
    
    # 3. Análisis Detallado (Matriz Completa Backbone x Clasificador)
    logger.info("Generando gráficos por combinación Backbone-Clasificador...")
    
    df_summ = data['summary']
    # Filtro Backbones (Sin GAP)
    valid_backbones = []
    if 'Embedding Model' in df_summ.columns:
        valid_backbones = df_summ[~df_summ['Embedding Model'].str.contains('_gap', na=False)]['Embedding Model'].unique()
    else:
        valid_backbones = []

    # Lista de Clasificadores (Sin Faiss)
    # Nos aseguramos de obtener la lista completa y limpia para el dropdown
    if 'Classifier' in df_summ.columns:
        all_classifiers = df_summ['Classifier'].unique()
        # Filtramos Faiss
        valid_classifiers = [c for c in all_classifiers if 'Faiss' not in str(c)]
    else:
        valid_classifiers = []
        logger.warning("No se encontró columna 'Classifier' en summary.")

    logger.info(f"Clasificadores encontrados para el menú: {valid_classifiers}")
    
    umap_files = {}
    tax_records = []  # Conteos crudos por backbone×clasificador, para 06_error_ranking
    ivc_records = []  # DataFrames Category/Correct/Incorrect por backbone×clasificador, para 07_ivc_ranking

    evaluator = ModelEvaluator(DATASET_INDEX_PATH, FEATURES_DIR)
    logging.getLogger("backbones").setLevel(logging.ERROR)
    
    pbar = tqdm(valid_backbones, desc="Procesando")
    for model_name in pbar:
        # A. UMAP (Uno por Backbone, independiente del clasificador)
        out_umap = UMAP_DIR / f"umap_{model_name}.png"
        umap_files[model_name] = out_umap
        
        if not out_umap.exists() or not out_umap.with_suffix('.svg').exists():
            embed_data = evaluator.load_embeddings(model_name)
            if embed_data[0] is not None:
                # Mapa de familias
                fam_col = 'family' if 'family' in df_index.columns else 'Family'
                sp_col = 'species' if 'species' in df_index.columns else 'class_name'
                if fam_col in df_index.columns and sp_col in df_index.columns:
                    fam_map = dict(zip(df_index[sp_col], df_index[fam_col]))
                else:
                    fam_map = None
                viz.plot_umap(embed_data[3], embed_data[4], model_name, out_umap, fam_map)

        # B. Gráficos Específicos por Clasificador (Errores e IVC)
        pred_path = BENCHMARK_RESULTS_DIR / f"predictions_{model_name}.csv"
        if pred_path.exists():
            df_pred = pd.read_csv(pred_path)
            
            # Detectar columna True
            col_true = None
            for c in ['y_true', 'True Label', 'true_label']:
                if c in df_pred.columns: col_true = c; break
            
            if not col_true: continue
            y_true = df_pred[col_true]

            # Detectar columnas Pred
            pred_cols = [c for c in df_pred.columns if c.startswith('pred_') and 'Faiss' not in c]
            
            for col_pred in pred_cols:
                clf_name = col_pred.replace('pred_', '')
                clf_slug = clean_filename(clf_name)
                y_p = df_pred[col_pred]

                out_tax = ERR_DIR / f"tax_error_{model_name}_{clf_slug}.png"
                out_ivc = IVC_DIR / f"ivc_perf_{model_name}_{clf_slug}.png"
                tax_svg = out_tax.with_suffix('.svg')
                ivc_svg = out_ivc.with_suffix('.svg')

                # El cálculo (analyze_*, sobre pandas ya en memoria) es barato y se
                # hace siempre para alimentar las figuras de resumen 06/07. El guard
                # de caché solo evita el _save_figure costoso (600 DPI) cuando la
                # figura individual de esta combinación ya existe.
                err_counts = analysis.analyze_taxonomic_errors(y_true, y_p, df_index, family_to_class)
                tax_records.append({**err_counts, 'Embedding Model': model_name, 'Classifier': clf_name})

                if not out_tax.exists() or not tax_svg.exists():
                    viz.plot_taxonomic_errors(err_counts, f"{model_name} + {clf_name}", out_tax)

                df_ivc = analysis.analyze_ivc_performance(y_true, y_p, df_index)
                if df_ivc is not None:
                    df_ivc_tagged = df_ivc.copy()
                    df_ivc_tagged['Embedding Model'] = model_name
                    df_ivc_tagged['Classifier'] = clf_name
                    ivc_records.append(df_ivc_tagged)

                    if not out_ivc.exists() or not ivc_svg.exists():
                        viz.plot_ivc_performance(df_ivc, f"{model_name} + {clf_name}", out_ivc)

        gc.collect()

    # v. Figuras de Resumen Comparativo (Ranking de Errores Taxonómicos e IVC, por Backbone)
    logger.info("Generando figuras de resumen comparativo (ranking de errores e IVC)...")
    df_err_rank = analysis.summarize_taxonomic_errors_by_backbone(tax_records)
    viz.plot_error_ranking(df_err_rank, FIG_DIR / "06_error_ranking.png")

    df_ivc_rank = analysis.summarize_ivc_performance_by_backbone(ivc_records)
    viz.plot_ivc_ranking(
        df_ivc_rank, FIG_DIR / "07_ivc_ranking.png",
        order=df_err_rank['Embedding Model'].tolist()
    )

    # vi. Análisis por Clase Taxonómica (Top-5 backbones por Accuracy, Linear SVM)
    logger.info("Generando análisis de desempeño por clase taxonómica (top-5 backbones)...")
    TOP5_BACKBONES = ['bioclip_v2', 'dinov2_base', 'dinov3_base', 'siglip2_so400m', 'dinov2_small']
    taxclass_results = {}
    confusion_results = {}
    for model_name in TOP5_BACKBONES:
        pred_path = BENCHMARK_RESULTS_DIR / f"predictions_{model_name}.csv"
        if not pred_path.exists():
            logger.warning(f"No se encontró {pred_path.name}, se omite del análisis por clase taxonómica.")
            continue
        df_pred_tc = pd.read_csv(pred_path)
        if 'pred_Linear SVM' not in df_pred_tc.columns:
            logger.warning(f"{pred_path.name} no tiene columna 'pred_Linear SVM', se omite.")
            continue
        taxclass_results[model_name] = analysis.analyze_performance_by_taxclass(
            df_pred_tc, 'pred_Linear SVM', df_index, family_to_class
        )
        confusion_results[model_name] = analysis.compute_taxclass_confusion(
            df_pred_tc, 'pred_Linear SVM', df_index, family_to_class
        )

    if taxclass_results:
        present_top5 = [b for b in TOP5_BACKBONES if b in taxclass_results]
        viz.plot_taxclass_heatmap(
            taxclass_results, present_top5, FIG_DIR / "08_taxclass_analysis.png"
        )
        viz.plot_taxclass_confusion(
            confusion_results, present_top5, FIG_DIR / "09_confusion_matrix_taxclass.png"
        )

    # vii. Bootstrap CI (Fase 7, opcional)
    if data.get('bootstrap_ci') is not None:
        logger.info("Generando figura de bootstrap CI...")
        viz.plot_bootstrap_forest(
            data['bootstrap_ci'], metric="Accuracy", output_path=FIG_DIR / "10_bootstrap_forest.png"
        )

    # viii. Generar HTML
    # 4. HTML Generación
    logger.info("Maquetando HTML Interactivo...")
    
    rep.generate_html_report(
        data, 
        REPORT_DIR / "benchmark_report.html", 
        umap_files, 
        sorted(list(valid_backbones)), 
        sorted(list(valid_classifiers)),
        stats
    )
    logger.info("Reporte Final Generado")

if __name__ == "__main__":
    main()