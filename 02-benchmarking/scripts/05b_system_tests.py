"""Tests de sistema: actualización incremental (agregar nuevas especies sin reentrenar)
y detección de outliers (Open Set Recognition) para los modelos top del benchmark."""
import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import Normalizer
import warnings
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.benchmarking import ModelEvaluator
from src.config import (
    DATASET_INDEX_PATH, FEATURES_DIR, BENCHMARK_RESULTS_DIR,
    BACKBONES_TIMES_PATH, INCREMENTAL_RESULTS_PATH, OUTLIER_RESULTS_PATH,
)

# Configuración del logger
logger = setup_logger("system-tests")
warnings.filterwarnings("ignore", category=UserWarning)  # Silenciar warnings de sklearn

def get_target_models(summary_path, times_path):
    """
    Selecciona: Top 5 F1-Score + Top 5 Más Rápidos.
    """
    selected_models = set()

    # 1. Top 5 mejores (F1-Macro)
    if os.path.exists(summary_path):
        try:
            df_sum = pd.read_csv(summary_path)
            best_f1 = df_sum.groupby('Embedding Model')['F1-Macro'].max().sort_values(ascending=False).head(5)
            selected_models.update(best_f1.index.tolist())
            logger.info(f"Top 5 F1-Score: {best_f1.index.tolist()}")
        except Exception as e:
            logger.error(f"Error leyendo benchmark_summary.csv: {e}")

    # 2. Top 5 más rápidos
    if os.path.exists(times_path):
        try:
            df_time = pd.read_csv(times_path)
            fastest = df_time.sort_values('Backbone Time (ms)', ascending=True).head(5)
            selected_models.update(fastest['Embedding Model'].tolist())
            logger.info(f"Top 5 Más Rápidos: {fastest['Embedding Model'].tolist()}")
        except Exception as e:
            logger.error(f"Error leyendo backbones_times.csv: {e}")

    if not selected_models:
        logger.warning("No se pudieron seleccionar los modelos automáticamente. Usando default.")
        return ["dinov2_small", "dinov2_base", 
                "dinov3_small", "dinov3_base", 
                "bioclip_v1", "bioclip_v2", 
                "convnextv2_tiny", 
                "clip_base", 
                "siglip_base", "siglip2_base"
                ] 
    
    return list(selected_models)

def test_incremental_update(X_train, y_train, model_name, backbone_time_ms):
    """
    Mide el tiempo promedio de agregar UNA especie nueva.
    Se repite el proceso con 5 especies distintas y se promedia.
    """
    logger.info(f"--- Test Incremental: {model_name} ---")
    
    classes = np.unique(y_train)
    if len(classes) < 15:
        logger.warning("Pocas clases para test incremental. Saltando.")
        return None

    # Fase 1: Entrenar sistema base (10 clases)
    base_classes = classes[:10]
    mask_base = np.isin(y_train, base_classes)
    # Clasificador simple basado en centroides (NearestCentroid) para velocidad y simplicidad
    clf = NearestCentroid(metric='euclidean')
    clf.fit(X_train[mask_base], y_train[mask_base])
    
    # Fase 2: Bucle de actualización (Agregar 5 especies nuevas)
    new_classes = classes[10:15]
    
    times_ingest = []
    times_math = []
    times_total = []
    samples_per_update = []

    t_backbones_sec = backbone_time_ms / 1000  # Convertir ms a segundos
    for new_cls in new_classes:
        # a. Simular Ingesta (tomar 5 fotos)
        mask_cls = (y_train == new_cls)
        X_cls = X_train[mask_cls]

        # Usamos 5 fotos (o menos si no hay suficientes)
        n_shots = min(5, len(X_cls))
        if n_shots == 0: continue

        X_shots = X_cls[:n_shots]

        # Tiempo Ingesta (Simulado: N * Backbone Time)
        t_ingest = n_shots * t_backbones_sec

        # b. Simular Cálculo de Centroides (Matemáticas)
        t0 = time.perf_counter()

        # Calcular nuevo centroide para la clase nueva
        new_centroid = np.mean(X_shots, axis=0)

        # Actualizar sistema (Concatenar centroides y clases)
        # (Simulamos la actualización en memoria, no reentrenamos todo el modelo)
        clf.centroids_ = np.vstack([clf.centroids_, new_centroid])
        clf.classes_ = np.concatenate([clf.classes_, [new_cls]])
        t_math = time.perf_counter() - t0

        # Guardar métricas de esta iteración
        times_ingest.append(t_ingest)
        times_math.append(t_math)
        times_total.append(t_ingest + t_math)
        samples_per_update.append(n_shots)

    t_total = sum(times_total)
    avg_time_per_class = np.mean(times_total) if times_total else 0.0
    
    logger.info(f"Tiempo Total (5 clases): {t_total:.4f} s")
    logger.info(f"Tiempo por clase: {avg_time_per_class:.4f} s")
    
    return {
        'Embedding Model': model_name,
        'Initial Classes': len(base_classes),
        'Added Classes': len(new_classes),
        'Total Time (s)': t_total,
        'Avg Time per Class (s)': avg_time_per_class,
        'Avg Time Ingest (s)': np.mean(times_ingest) if times_ingest else 0.0,
        'Avg Time Math (s)': np.mean(times_math) if times_math else 0.0
    }

def test_outlier_detection(X_train, y_train, X_test, y_test, model_name):
    """
    Open Set Recognition:
    Entrena con 80% de clases (Inliers).
    Prueba con mezcla de Inliers y 20% clases nunca vistas (Outliers).
    """
    logger.info(f"--- Test Outliers (Open Set Recognition): {model_name} ---")
    
    classes = np.unique(y_train)
    if len(classes) < 20: return None
    
    # División: 80% Conocidas (Inliers), 20% Desconocidas (Outliers)
    n_known = int(len(classes) * 0.8)
    known_classes = classes[:n_known]
    unknown_classes = classes[n_known:]
    
    # Preparar datos
    # Train: Solo con clases conocidas
    mask_train = np.isin(y_train, known_classes)
    X_tr_known = X_train[mask_train]
    y_tr_known = y_train[mask_train]
    
    # Test: Mezcla de Conocidas y Desconocidas
    # Etiquetamos: 0 = Inlier (Conocido), 1 = Outlier (Desconocido)
    y_test_binary = np.isin(y_test, unknown_classes).astype(int)
    
    # Entrenar solo con conocidas
    clf = NearestCentroid(metric='euclidean')
    clf.fit(X_tr_known, y_tr_known)
    
    # Calcular Umbral (Percentil 95 de distancias en Train)
    # Distancias de cada muestra de train a su centroide de clase correspondiente
    train_dists = []
    for i, x in enumerate(X_tr_known):
        centroid = clf.centroids_[np.where(clf.classes_ == y_tr_known[i])[0][0]]
        train_dists.append(np.linalg.norm(x - centroid))
    threshold = np.percentile(train_dists, 95)
    
    # Predecir en Test
    preds_cls = clf.predict(X_test)
    pred_idxs = [np.where(clf.classes_ == y)[0][0] for y in preds_cls]
    test_dists = np.linalg.norm(X_test - clf.centroids_[pred_idxs], axis=1)
    
    # Clasificar como Outlier si supera umbral
    y_pred_binary = (test_dists > threshold).astype(int)
    
    # Métricas
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_binary, y_pred_binary, average='binary', pos_label=1)
    
    # Tasa de Falsas Alarmas (False Positive Rate): Inliers clasificados como Outliers
    # FPR = FP / (FP + TN)
    # Aquí "Positive" es Outlier. Entonces un FP es un Inlier que dijimos que era Outlier.
    # TN son Inliers detectados como Inliers.
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    logger.info(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | False Alarm: {false_alarm_rate:.4f}")
    
    return {
        'Embedding Model': model_name,
        'Precision Outliers': prec,
        'Recall Outliers': rec,
        'F1 Score Outliers': f1,
        'False Alarm Rate': false_alarm_rate,
        'Threshold Distance': threshold
    }

def main():
    """Ejecuta los tests de actualización incremental y detección de outliers."""
    logger.info("==============================================")
    logger.info("   FASE 5b: TESTS DE SISTEMA (Incr. + Outliers)")
    logger.info("   (Selección: Top 5 F1 Score y Top 5 Velocidad)")
    logger.info("==============================================")


    backbone_times = {}
    if os.path.exists(BACKBONES_TIMES_PATH):
        df_bb = pd.read_csv(BACKBONES_TIMES_PATH)
        backbone_times = dict(zip(df_bb['Embedding Model'], df_bb['Backbone Time (ms)']))
        logger.info(f"Tiempos de backbone cargados para {len(backbone_times)} modelos.")

    # Selección Automática
    # - Mejores F1-Score (Top 5)
    # - Más Rápidos (Top 5)
    MODELS = get_target_models(
        BENCHMARK_RESULTS_DIR / "benchmark_summary.csv",
        BACKBONES_TIMES_PATH,
    )
    logger.info(f"- Modelos a evaluar: {len(MODELS)}")

    evaluator = ModelEvaluator(DATASET_INDEX_PATH, FEATURES_DIR)

    res_incremental = []
    res_outliers = []

    pbar = tqdm(MODELS, desc="Evaluando Modelos", unit="model")
    for model_name in pbar:
        pbar.set_description(f"Test Sistema: {model_name}")
        
        t_bb = backbone_times.get(model_name, 0.0)
        # Cargar embeddings
        data = evaluator.load_embeddings(model_name)
        if data[0] is None: continue
        
        X_train, y_train, _, X_test, y_test, _ = data
        scaler = Normalizer(norm='l2')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 1. Test Incremental
        inc_data = test_incremental_update(X_train, y_train, model_name, t_bb)
        if inc_data: res_incremental.append(inc_data)

        # 2. Test Outliers
        out_data = test_outlier_detection(X_train, y_train, X_test, y_test, model_name)
        if out_data: res_outliers.append(out_data)

    # Guardar
    if res_incremental:
        pd.DataFrame(res_incremental).to_csv(INCREMENTAL_RESULTS_PATH, index=False)
        logger.info("\n--- ACTUALIZACIÓN INCREMENTAL (Simulación 5 especies) ---" \
        f"\n{pd.DataFrame(res_incremental)[['Embedding Model', 'Total Time (s)', 'Avg Time per Class (s)']].sort_values('Avg Time per Class (s)').to_string(index=False)}")

        logger.info(f"Resultados de Test Incremental guardados en: {INCREMENTAL_RESULTS_PATH}")
    if res_outliers:
        df_out = pd.DataFrame(res_outliers)
        df_out.to_csv(OUTLIER_RESULTS_PATH, index=False)
        logger.info("\n--- DETECCIÓN DE OUTLIERS (Trade-off) ---" \
        f"\n{df_out[['Embedding Model', 'Precision Outliers', 'Recall Outliers', 'False Alarm Rate']].sort_values('Recall Outliers', ascending=False).to_string(index=False)}")

        logger.info(f"Resultados de Test Outliers guardados en: {OUTLIER_RESULTS_PATH}")

        logger.info("Tests de Sistema finalizados.")

if __name__ == "__main__":
    main()