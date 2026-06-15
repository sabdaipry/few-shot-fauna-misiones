import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import Normalizer
import logging
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.benchmarking import ModelEvaluator
# Configuración del logger
logger = setup_logger("scalability-test")
# Silenciar warnings de sklearn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    logger.info("==============================================")
    logger.info("   FASE 5: PRUEBA DE ESCALABILIDAD")
    logger.info("   (Curvas de Degradación)")
    logger.info("==============================================")

    # --- CONFIGURACIÓN ---
    INDEX_PATH = "data/dataset_index.csv"
    FEATURES_DIR = "data/features"
    OUTPUT_DIR = "data/scalability_results.csv"
    BACKBONES_TIMES_FILE = "data/backbones_times.csv"

    # Escalones de cantidad de especies a probar
    # (Ajustado al dataset de ~91 especies)
    STEPS = [5, 10, 20, 30, 50, 70, 91]

    # Cantidad de veces que repetimos el experimento para promediar (random seeds)
    ITERATIONS = 5

    # Modelos a evaluar:
    MODELS_TO_TEST = [
        "resnet50",
        "convnextv2_tiny", "convnextv2_base",
        "dinov2_small", "dinov2_base",
        "dinov2_small_gap", "dinov2_base_gap",
        "dinov3_small", "dinov3_base",
        "dinov3_small_gap", "dinov3_base_gap",
        "siglip_base", "siglip_so400m",
        "siglip2_base", "siglip2_so400m",
        "bioclip_v1", "bioclip_v2",
        "clip_base", "clip_large"
    ]

    # Cargar tiempos de backbone (para sumar latencia real)
    backbone_times = {}
    if os.path.exists(BACKBONES_TIMES_FILE):
        df_bb = pd.read_csv(BACKBONES_TIMES_FILE)
        # Crear diccionario {modelo: tiempo_ms}
        backbone_times = dict(zip(df_bb['Embedding Model'], df_bb['Backbone Time (ms)']))
        logger.info(f"Tiempos de backbone cargados para {len(backbone_times)} modelos.")
    else:
        logger.warning(f"No se encontró el archivo de tiempos de backbones en {BACKBONES_TIMES_FILE}. Los tiempos de backbone no serán considerados.")
    
    # Instanciar el evaluador
    evaluator = ModelEvaluator(INDEX_PATH, FEATURES_DIR)
    results = []

    # Barra Maestra (Modelos)
    pbar_models = tqdm(MODELS_TO_TEST, desc="Modelos", unit="model")

    for model_name in pbar_models:
        pbar_models.set_description(f"Procesando: {model_name}")

        # Recuperar tiempo de backbone (si no existe, 0)
        t_backbone = backbone_times.get(model_name, 0.0)
        
        # 1. Cargar embeddings (Usamos el evaluador existente)
        data = evaluator.load_embeddings(model_name)

        # Desempaquetar (load embeddings devuelve 6 valores)
        if data[0] is None:
            logger.warning(f"-> No se encontraron embeddings para {model_name}. Saltando...")
            continue

        X_train_full, y_train_full, _, X_test_full, y_test_full, _ = data

        # 2. Normalización L2 (Crucial para NearestCentroid Euclidiano == Coseno)
        scaler = Normalizer(norm='l2')
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)

        # Lista total de clases disponibles en este modelo
        all_classes = np.unique(y_train_full)
        max_classes_available = len(all_classes)

        # Pre-entrenar para calcular umbral de outliers base
        # (Usamos el percentil 95 de distancias intra-clase como umbral)
        clf_base = NearestCentroid(metric='euclidean')
        clf_base.fit(X_train_full, y_train_full)
        centroids = clf_base.centroids_
        # Distancias de train a sus propios centroides
        y_idxs = [np.where(clf_base.classes_ == y)[0][0] for y in y_train_full]
        train_dists = np.linalg.norm(X_train_full - centroids[y_idxs], axis=1)
        # Umbral heurístico: Distancia máxima esperada
        OUTLIER_THRESHOLD = np.percentile(train_dists, 95)

        # Warmup del clasificador
        # Ejecutamos una vez en falso para cargar librerías y evitar pico inicial
        try:
            clf_warm = NearestCentroid(metric='euclidean')
            clf_warm.fit(X_train_full[:50], y_train_full[:50])
            clf_warm.predict(X_test_full[:10])
        except:
            pass

        # Loop por escalones
        for n_classes in STEPS:
            # Si el paso pide más clases de las que existen, lo ajustamos al máximo real
            actual_n = min(n_classes, max_classes_available)
            
            # Repetir N veces para reducir varianza aleatoria
            metrics = {
                'f1': [],
                'acc': [],
                'prec': [],
                'rec': [],
                'lat': [],
                'samples': [],
                'out_rate': []
            }

            for i in range(ITERATIONS):
                # a. Selección aleatoria de clases
                # Si pedimos todas las clases, no es necesario randomizar
                if actual_n == max_classes_available:
                    selected_classes = all_classes
                else:
                    selected_classes = np.random.choice(all_classes, size=actual_n, replace=False)
                
                # b. Filtrado de datos para las clases seleccionadas
                mask_train = np.isin(y_train_full, selected_classes)
                mask_test = np.isin(y_test_full, selected_classes)

                X_tr_sub = X_train_full[mask_train]
                y_tr_sub = y_train_full[mask_train]
                X_te_sub = X_test_full[mask_test]
                y_te_sub = y_test_full[mask_test]

                if len(X_te_sub) == 0 or len(X_tr_sub) == 0:
                    logger.warning(f"-> No hay datos para las clases seleccionadas en la iteración {i+1} con {actual_n} clases. Saltando esta iteración.")
                    continue

                # c. Entrenamiento y evaluación del clasificador NearestCentroid
                # Usamos metric='euclidean' porque ya normalizamos L2 (equivale a coseno)
                clf = NearestCentroid(metric='euclidean')
                clf.fit(X_tr_sub, y_tr_sub)

                # d. Inferencia y Latencia
                t0 = time.perf_counter()
                y_pred = clf.predict(X_te_sub)
                t1 = time.perf_counter()

                # Tiempo del clasificador por imagen (ms)
                t_clf_per_img = ((t1 - t0) / len(X_te_sub)) * 1000  # ms

                # LATENCIA TOTAL = Backbone + Clasificador
                t_total_per_img = t_backbone + t_clf_per_img

                # e. Detección de outliers basada en distancia al centroide predicho
                pred_idxs = [np.where(clf.classes_ == y)[0][0] for y in y_pred]
                dists = np.linalg.norm(X_te_sub - clf.centroids_[pred_idxs], axis=1)

                # Si distancia > umbral, lo consideramos "dudoso" o "outlier"
                outliers = np.sum(dists > OUTLIER_THRESHOLD)
                outlier_rate = outliers / len(X_te_sub) if len(X_te_sub) > 0 else 0.0

                # F. Calcular métricas
                metrics['f1'].append(f1_score(y_te_sub, y_pred, average='macro', zero_division=0))
                metrics['acc'].append(accuracy_score(y_te_sub, y_pred))
                metrics['prec'].append(precision_score(y_te_sub, y_pred, average='macro', zero_division=0))
                metrics['rec'].append(recall_score(y_te_sub, y_pred, average='macro', zero_division=0))
                metrics['lat'].append(t_total_per_img)
                metrics['samples'].append(len(X_te_sub))
                metrics['out_rate'].append(outlier_rate)

            # Promediar resultados de las iteraciones
            results.append({
                'Embedding Model': model_name,
                'Num Classes': actual_n,
                'Samples': int(np.mean(metrics['samples'])),
                'Accuracy': np.mean(metrics['acc']),
                'F1 Score Macro': np.mean(metrics['f1']),
                'Precision Macro': np.mean(metrics['prec']),
                'Recall Macro': np.mean(metrics['rec']),
                'Latency per image (ms)': np.mean(metrics['lat']),
                'Outlier Rate': np.mean(metrics['out_rate'])
            })

            if actual_n == max_classes_available: break  # No tiene sentido seguir aumentando si ya usamos todas las clases
    
    # Guardar resultados finales
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_DIR, index=False)
        logger.info("="*50)
        logger.info("PRUEBA DE ESCALABILIDAD FINALIZADA")
        logger.info(f"Resultados guardados en: {OUTPUT_DIR}")

        # Preview rápido
        logger.info("Vista previa de resultados (todas las clases):")
        final_step = df_results[df_results['Num Classes'] == df_results['Num Classes'].max()]
        print(final_step[['Embedding Model', 'Accuracy', 'F1 Score Macro']].sort_values('F1 Score Macro', ascending=False).to_string(index=False))
    else:
        logger.error("No se obtuvieron resultados de la prueba de escalabilidad.")


if __name__ == "__main__":
    main()
