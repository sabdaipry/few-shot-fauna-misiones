"""
Clasificadores personalizados y orquestador de benchmark para embeddings de imágenes.

Contiene:
- FaissKNNClassifier: KNN acelerado con FAISS (IndexFlatIP, similitud coseno).
- FaissNearestCentroid: clasificador por centroide más cercano con FAISS.
- ModelEvaluator: orquesta la carga de embeddings, evaluación de clasificadores y
  persistencia de resultados en CSV/Excel.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
# Ignorar advertencias de división por cero o métricas indefinidas para limpiar la consola
warnings.filterwarnings("ignore", category=UserWarning)

# Scikit-learn
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, 
                             top_k_accuracy_score, silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score)
from sklearn.preprocessing import Normalizer
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# FAISS
import faiss

# Ajuste de rutas para imports relativos
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

# Importamos nuestro logger
from src.utils.logger import setup_logger

logger = setup_logger("benchmarking-module")

# --- WRAPPER PARA FAISS KNN (Para que funcione como un clasificador de sklearn) ---
class FaissKNNClassifier:
    """KNN acelerado con FAISS usando similitud coseno (IndexFlatIP sobre vectores L2-normalizados)."""

    def __init__(self, k=5):
        self.k = k
        self.index = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """Indexa X en FAISS y guarda las etiquetas de entrenamiento."""
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        d = X.shape[1]
        # Usamos IndexFlatIP (Inner Product) que es equivalente a Coseno si los vectores están normalizados
        self.index = faiss.IndexFlatIP(d)
        self.index.add(X.astype(np.float32))

    def predict(self, X):
        """Retorna la etiqueta del vecino más cercano para cada muestra en X."""
        distances, indices = self.index.search(X.astype(np.float32), 1)
        return self.y_train[indices.flatten()]

    def predict_top_k(self, X, k=5):
        """Retorna una matriz [n_samples, k] con las etiquetas de los k vecinos más cercanos."""
        search_k = min(self.k, k)
        distances, indices = self.index.search(X.astype(np.float32), search_k)
        top_k_preds = []
        for i in range(len(X)):
            neighbor_labels = self.y_train[indices[i]]
            # FAISS ya devuelve ordenado por distancia; tomamos clases únicas en ese orden
            unique_labels = pd.unique(neighbor_labels)

            # Rellenamos si faltan candidatos (poco probable con k suficiente)
            if len(unique_labels) < k:
                unique_labels = np.pad(unique_labels, (0, k - len(unique_labels)),
                                       mode='constant', constant_values=unique_labels[-1])
            top_k_preds.append(unique_labels[:k])
        return np.array(top_k_preds)

class FaissNearestCentroid:
    """Clasificador por centroide más cercano acelerado con FAISS (similitud coseno)."""

    def __init__(self):
        self.index = None
        self.centroid_labels = None
        self.classes_ = None

    def fit(self, X, y):
        """Calcula el centroide L2-normalizado de cada clase e indexa en FAISS."""
        self.classes_ = np.unique(y)
        self.centroid_labels = self.classes_
        centroids = []
        for cls in self.classes_:
            cls_vectors = X[y == cls]
            centroids.append(cls_vectors.mean(axis=0))

        centroids_vectors = np.array(centroids, dtype=np.float32)

        # Normalización L2: vital para que IndexFlatIP funcione como similitud coseno
        faiss.normalize_L2(centroids_vectors)

        d = X.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(centroids_vectors)

    def predict(self, X):
        """Retorna la etiqueta del centroide más cercano para cada muestra en X."""
        if self.index is None:
            return np.array([])
        distances, indices = self.index.search(X.astype(np.float32), 1)
        return self.centroid_labels[indices.flatten()]

    def predict_top_k(self, X, k=5):
        """Retorna una matriz [n_samples, k] con las etiquetas de los k centroides más cercanos."""
        if self.index is None:
            return np.array([])
        distances, indices = self.index.search(X.astype(np.float32), k)
        return self.centroid_labels[indices]

# ---  WRAPPER DE EVALUACIÓN DE MODELOS ---

class ModelEvaluator:
    """
    Orquesta el benchmark de modelos de embeddings contra una batería de clasificadores.

    Carga embeddings pre-computados, normaliza, corre clasificadores y persiste resultados
    de forma incremental (caché en CSV) para evitar re-cómputo ante interrupciones.
    """

    def __init__(self, index_path, features_root_dir,
                 output_dir=None):
        self.index_df = pd.read_csv(index_path)
        self.features_root = Path(features_root_dir)
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "data" / "results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo maestro de resultados
        self.summary_file = self.output_dir / "benchmark_summary.csv"
        
        # Cargar caché si existe
        if self.summary_file.exists():
            self.results_df = pd.read_csv(self.summary_file)
            self.evaluated_models = set(self.results_df['Embedding Model'].unique())
        else:
            self.results_df = pd.DataFrame()
            self.evaluated_models = set()

    def check_if_processed(self, model_name, classifier_name=None):
        """
        Verifica si ya existen resultados en el caché.
        - Si classifier_name es None: verifica si hay ALGO del modelo.
        - Si classifier_name existe: verifica esa combinación específica.
        """
        if self.results_df.empty:
            return False
        
        # Filtramos por modelo
        model_mask = self.results_df['Embedding Model'] == model_name
        
        if classifier_name:
            # Verificamos si este clasificador específico ya corrió para este modelo
            clf_mask = self.results_df['Classifier'] == classifier_name
            exists = not self.results_df[model_mask & clf_mask].empty
            return exists
        else:
            # Verificamos si hay algún registro del modelo
            return not self.results_df[model_mask].empty

    def load_embeddings(self, model_folder):
        """
        Carga los embeddings pre-computados de un modelo.

        Retorna una tupla de 6 elementos:
        (X_train, y_train, idx_train, X_test, y_test, idx_test).
        En caso de error retorna (None, None, None, None, None, None).
        """
        model_dir = self.features_root / model_folder
        if not model_dir.exists():
            logger.warning(f"Carpeta no encontrada: {model_dir}")
            return None, None, None, None, None, None

        X_train, y_train, idx_train = [], [], []
        X_test, y_test, idx_test = [], [], []

        # Usamos un diccionario para búsqueda rápida de archivos
        # Esto optimiza la carga vs iterar el dataframe 1 a 1 buscando archivos
        available_files = {f.name: f for f in model_dir.rglob("*.npy")}

        missing = 0

        for idx, row in self.index_df.iterrows():
            orig_path = Path(row['filepath'])
            
            npy_name = orig_path.with_suffix('.npy').name
            
            if npy_name in available_files:
                emb = np.load(available_files[npy_name])
                label = row['species']
                
                if row['split'] == 'gallery':
                    X_train.append(emb)
                    y_train.append(label)
                    idx_train.append(idx)
                else:
                    X_test.append(emb)
                    y_test.append(label)
                    idx_test.append(idx)
            else:
                missing += 1

        if len(X_train) == 0:
            return None, None, None, None, None, None

        return (np.array(X_train), np.array(y_train), np.array(idx_train),
                np.array(X_test), np.array(y_test), np.array(idx_test))
    
    def calculate_embedding_metrics(self, X, y):
        """
        Calcula métricas intrínsecas del espacio de embeddings: Silhouette (coseno),
        Davies-Bouldin y Calinski-Harabasz. Usa muestreo si N > 5000 (Silhouette es O(N²)).

        Retorna (silhouette, davies_bouldin, calinski_harabasz); en caso de error (-1, -1, -1).
        """
        try:
            if len(X) > 5000: # Sampling si es gigante
                indices = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample, y_sample = X, y

            sil = silhouette_score(X_sample, y_sample, metric='cosine')
            db = davies_bouldin_score(X_sample, y_sample)
            cal = calinski_harabasz_score(X_sample, y_sample)
            return sil, db, cal
        except Exception as e:
            logger.warning(f"No se pudo calcular métricas de embedding: {e}")
            return -1, -1, -1

    def _compute_top_k_acc(self, clf, X_test, y_test, k=5):
        """
        Calcula Top-K Accuracy intentando diferentes métodos según el clasificador.
        """
        try:
            # CASO 1: Modelos con método personalizado predict_top_k (Nuestros FAISS)
            if hasattr(clf, 'predict_top_k'):
                top_preds = clf.predict_top_k(X_test, k=k)
                # top_preds es una matriz de [n_samples, k] con las etiquetas
                hits = [1 if y_true in row else 0 for y_true, row in zip(y_test, top_preds)]
                return np.mean(hits)
            
            # CASO 2: Modelos con predict_proba (Random Forest, KNN Sklearn)
            elif hasattr(clf, 'predict_proba'):
                probs = clf.predict_proba(X_test)
                classes = clf.classes_
                # Obtenemos los índices de las k probabilidades más altas
                # argsort ordena ascendente, tomamos los últimos k y revertimos
                top_k_idxs = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
                top_preds = classes[top_k_idxs]
                hits = [1 if y_true in row else 0 for y_true, row in zip(y_test, top_preds)]
                return np.mean(hits)
            
            # CASO 3: Modelos con decision_function (Linear SVM)
            elif hasattr(clf, 'decision_function'):
                scores = clf.decision_function(X_test)
                classes = clf.classes_
                # Si es binario, decision_function devuelve 1 columna. Top-5 no tiene sentido (o es 100%)
                if scores.ndim == 1:
                    return 1.0 if k >= 2 else accuracy_score(y_test, clf.predict(X_test))
                
                top_k_idxs = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
                top_preds = classes[top_k_idxs]
                hits = [1 if y_true in row else 0 for y_true, row in zip(y_test, top_preds)]
                return np.mean(hits)
            
            else:
                return np.nan # No soportado (ej. NearestCentroid de sklearn sin logica extra)
        
        except Exception as e:
            logger.warning(f"Error calculando Top-{k} Accuracy: {e}")
            return np.nan
    
    def evaluate_model(self, model_name, force_rerun=False):
        """
        Ejecuta la batería de clasificadores para un modelo de embeddings dado.
        """
        # 1. Definir todos los clasificadores candidatos
        candidate_classifiers = {
            'Nearest Centroid': NearestCentroid(metric='euclidean'), # Baseline simple y rápida
            'KNN (k=1)': KNeighborsClassifier(n_neighbors=1, metric='cosine'), # Simula búsqueda vectorial pura
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3, metric='cosine'), # Más robusto que k=1
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine'), # Más robusto que k=1
            'Linear SVM': LinearSVC(C=1.0, dual="auto", max_iter=2000), # El estándar de oro para embeddings
            'RBF SVM': SVC(C=1.0, kernel='rbf', gamma='scale', max_iter=2000), # Captura relaciones no lineales
            'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=29), # Clasificador basado en árboles
            'Faiss KNN (k=1)': FaissKNNClassifier(k=1), # KNN rápido con FAISS
            'Faiss KNN (k=3)': FaissKNNClassifier(k=3), # KNN rápido con FAISS
            'Faiss KNN (k=5)': FaissKNNClassifier(k=5), # KNN rápido con FAISS
            'Faiss Nearest Centroid': FaissNearestCentroid() # Centroides con FAISS
        }

        # 2. Filtrado inteligente: Solo correr clasificadores no evaluados aún
        classifiers_to_run = {}
        for clf_name, clf in candidate_classifiers.items():
            if force_rerun or not self.check_if_processed(model_name, clf_name):
                classifiers_to_run[clf_name] = clf

        # 3. Si no falta ninguno, nos vamos temprano y ahorramos tiempo/memoria
        if not classifiers_to_run:
            tqdm.write(f"-> Todos los clasificadores ya evaluados para {model_name}, saltando...")
            return
        
        # ==========================================
        # Solo cargamos datos si realmente hace falta
        # ==========================================
        tqdm.write(f"-> Cargando datos para: {model_name}...")
        data = self.load_embeddings(model_name)
        if data[0] is None:
            tqdm.write(f"No se encontraron datos para {model_name}")
            return

        X_train, y_train, idx_train, X_test, y_test, idx_test = data


        # Normalización (Importante para SVM y KNN)
        # --- CAMBIO CLAVE: L2 NORMALIZATION ---
        # Esto hace que todos los vectores vivan en una esfera unitaria.
        # Con esto, la distancia Euclidiana es equivalente a la Similitud Coseno.
        scaler = Normalizer(norm='l2')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 1. Métricas intrínsecas del espacio de embeddings
        tqdm.write(f"-> Calculando métricas de embedding para: {model_name}...")
        # Unimos train y test para ver la calidad global del espacio
        X_full = np.vstack((X_train, X_test))
        y_full = np.hstack((y_train, y_test))
        sil_score, db_index, cal_index = self.calculate_embedding_metrics(X_full, y_full)

        # Dimensión de los embeddings
        emb_dim = X_train.shape[1]

        preds_file = self.output_dir / f"predictions_{model_name}.csv"
        if preds_file.exists():
            all_preds_df = pd.read_csv(preds_file)
            # Asegurar que idx es índice si no lo es
            if 'idx' in all_preds_df.columns:
                all_preds_df.set_index('idx', inplace=True)
        else:
            all_preds_df = pd.DataFrame({'idx': idx_test, 'y_true': y_test})
            all_preds_df.set_index('idx', inplace=True)
        # all_preds_df es un contenedor para guardar TODAS las predicciones de este modelo
        # Esto sirve para luego calcular Taxonomía, IVC, Matrices de confusión, etc.
        
        results_buffer = []

        # 4. Correr solo los faltantes
        pbar = tqdm(classifiers_to_run.items(), desc=f"   Evaluando", leave=False, unit="clf")

        for clf_name, clf in pbar:
            try:
                # Entrenar
                clf.fit(X_train, y_train)

                # Inference Time
                t0_inf = time.time()
                y_pred = clf.predict(X_test) # PREDICCIÓN
                inf_time = (time.time() - t0_inf) / len(X_test) * 1000 # ms por imagen
                
                # Métricas básicas
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
                top5_acc = self._compute_top_k_acc(clf, X_test, y_test, k=5)
        

                # Guardar métricas resumen
                results_buffer.append({
                    'Embedding Model': model_name,
                    'Classifier': clf_name,
                    'Accuracy': acc,
                    'Top-5 Accuracy': top5_acc,
                    'F1-Macro': f1,
                    'Precision': precision,
                    'Recall': recall,
                    'Inference Time Classifier (ms/img)': inf_time,
                    'Embedding Dim': emb_dim,
                    'Silhouette Score': sil_score,
                    'Davies-Bouldin Index': db_index,
                    'Calinski-Harabasz Index': cal_index
                })

                # Guardar predicciones detalladas en el DF auxiliar
                all_preds_df[f'pred_{clf_name}'] = y_pred

                pbar.set_postfix({"Acc": f"{acc:.2f}"})

            except Exception as e:
                tqdm.write(f"Error evaluando {clf_name} en {model_name}: {e}")

        # Guardar predicciopnes actualizadas
        # solo si es nuevo o hubo cambios
        # Agregar metadata taxonómica si existe en el index original
        if 'genus' not in all_preds_df.columns:
            cols_to_merge = ['genus', 'family', 'ivc_score', 'ivc_category'] # Asegurate que estas columnas existan
            idx_cols = [c for c in cols_to_merge if c in self.index_df.columns]
        
            if idx_cols:
                meta_df = self.index_df.loc[idx_test, idx_cols]
                all_preds_df = all_preds_df.join(meta_df)
        
        all_preds_df.to_csv(preds_file)
        
        # Actualizar archivo maestro de resultados
        new_results_df = pd.DataFrame(results_buffer)
        
        if self.summary_file.exists():
            new_results_df.to_csv(self.summary_file, mode='a', header=False, index=False)
        else:
            new_results_df.to_csv(self.summary_file, index=False)
        
        # Actualizar caché en memoria
        # (Nota: aquí podríamos agregar lógica para 'partial', pero por ahora asumimos procesado)
        # self.evaluated_models.add(model_name)

    def save_results(self, output_file="benchmark_results.csv"):
        """
        Exporta self.results_df ordenado por F1-Macro a CSV y, si openpyxl está disponible,
        también a Excel. Retorna el DataFrame resultante.
        """
        df = self.results_df.copy()
        df.sort_values(by=['F1-Macro'], ascending=False, inplace=True)
        df.to_csv(output_file, index=False)

        xlsx_file = str(output_file).replace(".csv", ".xlsx")
        try:
            df.to_excel(xlsx_file, index=False)
        except Exception as e:
            logger.warning(f"No se pudo guardar Excel en {xlsx_file}: {e}")

        logger.info(f"-> Resultados guardados en {output_file}")
        return df