"""Funciones de visualización para el benchmark de fauna de la Selva Paranaense."""
import pandas as pd
import numpy as np
import matplotlib
# Forzar backend headless antes de importar pyplot: evita OOM con Qt en loops largos de figuras a 600 DPI.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger("visualization")

# Configuración de Estilo Global
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 20

# Paleta "Selva" (Verdes)
FOREST_PALETTE = "Greens_r"


def _save_figure(output_path, fig=None):
    """Guarda la figura en PNG y SVG derivando la ruta SVG de output_path.

    Depende de rcParams ya configurados: savefig.bbox='tight' y figure.dpi=600.
    El formato SVG es vectorial; matplotlib ignora dpi para ese formato.
    """
    path = Path(output_path)
    svg_path = path.with_suffix('.svg')
    save_fn = fig.savefig if fig is not None else plt.savefig
    save_fn(path)
    save_fn(svg_path)
    plt.close(fig) if fig is not None else plt.close()

def standardize_columns(df):
    """
    Normaliza nombres de columnas a un estándar canónico único.
    """
    # Mapeo de Variaciones -> Nombre Canónico
    rename_map = {
        # F1 Score
        "F1-Macro": "F1_Macro",
        "F1 Score Macro": "F1_Macro",
        "F1 Score (Macro)": "F1_Macro",
        "F1 Score": "F1_Macro",

        # Accuracy
        "Accuracy": "Accuracy",

        # Latencia
        "Latency per image (ms)": "Backbone Time (ms)",
        "Latency (ms)": "Backbone Time (ms)",
        "Backbone Time (ms)": "Backbone Time (ms)",
        "Inference Time Classifier (ms/img)": "Classifier Time (ms)",

        # Otros
        "Num Classes": "N_Classes",
        "Embedding Model": "Embedding Model",
        "Classifier": "Classifier",
        "Embedding Dim": "Dim",
        "Embedding Dimension": "Dim",
        "Silhouette Score": "Silhouette Score",
        "Davies-Bouldin Score": "Davies-Bouldin Index",
        "Calinski-Harabasz Score": "Calinski-Harabasz Index",
        "family": "Family"
    }

    # Renombrar columnas existentes
    df = df.rename(columns=rename_map)
    return df

def filter_data(df, exclude_gap=True, exclude_faiss=True):
    """Filtra modelos GAP y clasificadores FAISS."""
    df_filtered = df.copy()
    if exclude_gap and 'Embedding Model' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['Embedding Model'].str.contains('_gap', case=False, na=False)]
    if exclude_faiss and 'Classifier' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['Classifier'].str.contains('FAISS', case=False, na=False)]
    return df_filtered

def get_model_palette(models):
    """Genera una paleta consistente para los modelos."""
    unique_models = sorted(list(set(models)))
    palette = sns.color_palette("tab20", n_colors=len(unique_models))
    return dict(zip(unique_models, palette))

def plot_leaderboard(df, metric="Accuracy", output_path="leaderboard.png"):
    """Ranking de modelos (Muestra solo el MEJOR clasificador por backbone)."""
    df = standardize_columns(df)
    df = filter_data(df)

    # Validar que la métrica exista
    if metric not in df.columns:
        print(f"Warning: Métrica '{metric}' no encontrada. Columnas disponibles: {df.columns.tolist()}")
        return

    # Agrupamos por backbone y nos quedamos con el valor máximo de la métrica
    # Esto resuelve el problema de que sea "ilegible" por tener 7 barras por modelo
    df_best = df.loc[df.groupby("Embedding Model")[metric].idxmax()].sort_values(metric, ascending=False)

    plt.figure(figsize=(10, 10))
    # Paleta de verdes según ranking (más oscuro = mejor)
    palette = sns.color_palette(FOREST_PALETTE, n_colors=len(df_best))

    ax = sns.barplot(
        data=df_best,
        y="Embedding Model",
        x=metric,
        palette=palette,
        hue="Embedding Model",
        legend=False,
        edgecolor="black",
        linewidth=0.5
    )

    plt.title(f"Ranking de Backbones (Mejor {metric} obtenido)", fontweight='bold')
    plt.xlabel(metric.replace('_', ' '))
    plt.ylabel("")
    plt.xlim(0, 1.08)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Etiquetas de valor
    for i, v in enumerate(df_best[metric]):
        ax.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=11, fontweight='bold')

    _save_figure(output_path)

def plot_heatmap(df, metric="Accuracy", output_path="heatmap.png"):
    """Matriz: Backbone (Eje Y) vs Clasificador (Eje X)."""
    df = standardize_columns(df)
    df = filter_data(df)

    if metric not in df.columns:
        return
    pivot_df = df.pivot(index="Embedding Model", columns="Classifier", values=metric)

    # Ordenar por promedio para que los mejores queden arriba
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('mean', ascending=False).drop('mean', axis=1)

    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="Greens", cbar_kws={'label': metric}, annot_kws={'fontsize': 16}, linewidths=.5, linecolor='white')
    ax.tick_params(axis='both', labelsize=14)

    plt.title(f"Performance: Backbone vs Clasificador ({metric})", fontweight='bold')
    plt.ylabel("Backbone")
    plt.xlabel("Clasificador")
    plt.xticks(rotation=45, ha='right')

    _save_figure(output_path)

def plot_pareto(df_sum, df_time, metric="Accuracy", output_path="pareto.png"):
    """
    Gráfico de Burbujas: Latencia vs Performance.
    Tamaño = Dimensión del Embedding (Dato dinámico del CSV).
    """
    df_sum = standardize_columns(df_sum)
    df_time = standardize_columns(df_time)

    df_sum = filter_data(df_sum) # Sin Gaps ni Faiss
    # df_time también necesita filtro de GAPs
    df_time = filter_data(df_time)

    if metric not in df_sum.columns: return

    # 1. Obtener mejor resultado por backbone
    df_best = df_sum.loc[df_sum.groupby("Embedding Model")[metric].idxmax()]

    # 2. Merge con Tiempos
    df = pd.merge(df_best, df_time, on="Embedding Model", how="inner")

    # 3. Determinar tamaño de burbuja (Dimensión)
    # Si la columna 'Dim' o 'Embedding Dimension' existe, la usamos. Si no, tamaño fijo.
    size_col = 'Dim' if 'Dim' in df.columns else None

    plt.figure(figsize=(14, 9))
    palette = get_model_palette(df["Embedding Model"])

    sns.scatterplot(
        data=df,
        x="Backbone Time (ms)",
        y=metric,
        hue="Embedding Model",
        palette=palette,
        size=size_col,
        sizes=(100, 1000) if size_col else None,
        alpha=0.7,
        edgecolor="black",
        linewidth=1
    )

    for i, row in df.iterrows():
        plt.text(
            row['Backbone Time (ms)'],
            row[metric] + 0.003,
            row['Embedding Model'],
            fontsize=11,
            ha='center',
            fontweight='bold',
            alpha=0.8
        )

    plt.title(f"Frontera de Pareto: Latencia vs {metric}", fontweight='bold')
    plt.xlabel("Latencia por Imagen (ms) [Log Scale]")
    plt.ylabel(metric)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Modelo / Dimensión")

    _save_figure(output_path)

def plot_scalability_curves(df_scale, output_path):
    """Curvas de degradación (TODOS los modelos)."""
    df_scale = standardize_columns(df_scale)
    df_scale = filter_data(df_scale) # Importante: sacar GAPs

    plt.figure(figsize=(14, 8))
    models = df_scale['Embedding Model'].unique()
    palette = get_model_palette(models)

    sns.lineplot(
        data=df_scale,
        x="N_Classes",
        y="Accuracy",
        hue="Embedding Model",
        palette=palette,
        style="Embedding Model",
        markers=True,
        dashes=False,
        linewidth=2,
        markersize=8
    )

    plt.title("Escalabilidad: Degradación de Accuracy al aumentar especies", fontweight='bold')
    plt.xlabel("Cantidad de Especies")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title="Backbone")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)

    _save_figure(output_path)

def plot_dino_comparison(df, metric="Accuracy", output_path="dino_comp.png"):
    """Comparativa DINO: CLS vs GAP."""
    df = standardize_columns(df)

    dinos = df[df['Embedding Model'].str.contains('dino')].copy()
    if dinos.empty:
        return
    dinos = dinos.loc[dinos.groupby("Embedding Model")[metric].idxmax()]

    dinos['Arquitectura'] = dinos['Embedding Model'].str.replace('_gap', '').str.replace('_cls', '')
    dinos['Pooling'] = dinos['Embedding Model'].apply(lambda x: 'GAP' if 'gap' in x else 'CLS')
    palette = sns.color_palette(FOREST_PALETTE, n_colors=2)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=dinos, x="Arquitectura", y=metric, hue="Pooling", palette=palette)

    plt.title(f"Impacto del Pooling en DINO ({metric})", fontweight='bold')
    plt.ylim(0.2, 1.0)
    plt.ylabel(metric)
    plt.xlabel("Modelo Base")

    _save_figure(output_path)

def plot_embedding_metrics(df, metrics=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'], output_path_prefix="emb_metric"):
    """Barplots para métricas de clustering."""
    df = standardize_columns(df)
    df = filter_data(df)

    df_unique = df.drop_duplicates(subset=["Embedding Model"]).copy()

    for met in metrics:
        if met not in df.columns: continue

        plt.figure(figsize=(12, 7))
        # Ordenar (Silhouette/Calinski: Mayor mejor, Davies: Menor mejor)
        ascending = True if "Davies" in met else False
        df_sorted = df_unique.sort_values(met, ascending=ascending)

        # Paleta de verdes
        palette = sns.color_palette(FOREST_PALETTE, n_colors=len(df_sorted))

        # Etiqueta eje X (Silhouette/Calinski: Mayor mejor, Davies: Menor mejor)
        x_label = f"{met} (Menor mejor)" if "Davies" in met else f"{met} (Mayor mejor)"

        sns.barplot(
            data=df_sorted,
            y="Embedding Model",
            x=met,
            palette=palette,
            hue="Embedding Model",
            legend=False,
            edgecolor="black",
        )

        plt.title(f"Calidad del Espacio Latente: {met}", fontweight='bold')
        plt.xlabel(x_label)
        plt.ylabel("")

        safe_name = met.replace(' ', '_').lower()
        _save_figure(f"{output_path_prefix}_{safe_name}.png")

def plot_backbone_latency(df_time, output_path="latency_backbone.png"):
    """Nueva gráfica: Latencia por Backbone."""
    df_time = standardize_columns(df_time)
    df_time = filter_data(df_time) # Sin GAPs

    df_sorted = df_time.sort_values("Backbone Time (ms)", ascending=True) # Más rápido arriba

    plt.figure(figsize=(10, 8))
    # Paleta verdes (más rápido = más oscuro/mejor o al revés? Usemos un solo color o degradado)
    palette = sns.color_palette(FOREST_PALETTE, n_colors=len(df_sorted))
    # Invertimos paleta para que los rápidos (arriba) sean oscuros/fuertes o claros?
    # Mejor: Barras más cortas = Mejor. Usemos un color fijo.

    sns.barplot(
        data=df_sorted,
        y="Embedding Model",
        x="Backbone Time (ms)",
        palette=palette,
        hue="Embedding Model",
        edgecolor="black",
        linewidth=0.5
    )

    plt.title("Latencia de Extracción (Backbone)", fontweight='bold', color='#1b4f25')
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("")

    # Anotar valores
    for i, v in enumerate(df_sorted["Backbone Time (ms)"]):
        plt.text(v + 5, i, f"{v:.1f} ms", va='center', fontsize=11)

    _save_figure(output_path)

def plot_classifier_latency(df_sum, output_path="latency_classifier.png"):
    """Nueva gráfica: Latencia Promedio por Clasificador."""
    df_sum = standardize_columns(df_sum)
    df_sum = filter_data(df_sum) # Sin FAISS

    if "Classifier Time (ms)" not in df_sum.columns: return

    # Promedio por clasificador
    df_clf = df_sum.groupby("Classifier")["Classifier Time (ms)"].mean().reset_index()
    df_sorted = df_clf.sort_values("Classifier Time (ms)", ascending=True)
    palette = sns.color_palette(FOREST_PALETTE, n_colors=len(df_sorted))
    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=df_sorted,
        y="Classifier",
        x="Classifier Time (ms)",
        palette=palette,
        hue="Classifier",
        edgecolor="black",
        linewidth=0.5
    )

    plt.title("Latencia de Inferencia (Clasificador)", fontweight='bold', color='#1b4f25')
    plt.xlabel("Tiempo Promedio (ms)")
    plt.ylabel("")
    plt.xscale('log') # Logarítmico porque NearestCentroid es ms y SVM puede ser más

    for i, v in enumerate(df_sorted["Classifier Time (ms)"]):
        plt.text(v * 1.1, i, f"{v:.4f} ms", va='center', fontsize=11)

    _save_figure(output_path)

def plot_umap(X, y, model_name, output_path, family_map=None):
    """UMAP por Familia."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(12, 10))

    if family_map is not None:
        try:
            # Normalizar nombres de especies (reemplazar _ por espacio por si acaso)
            families = []
            for label in y:
                fam = family_map.get(label)
                if not fam: fam = family_map.get(label.replace('_', ' ')) # Intento con espacio
                families.append(fam if fam else "Unknown")

            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=families, palette="tab20", s=15, alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Familia", ncol=2, fontsize=13)
        except Exception as e:
            logger.warning("UMAP family coloring failed (%s), falling back to scatter.", e)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(y)[0], cmap='Spectral', s=10, alpha=0.6)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(y)[0], cmap='Spectral', s=10, alpha=0.6)

    plt.title(f"Proyección UMAP: {model_name}", fontweight='bold')
    plt.xlabel("UMAP Dimensión 1")
    plt.ylabel("UMAP Dimensión 2")
    plt.xticks([])
    plt.yticks([])

    _save_figure(output_path)


def plot_taxonomic_errors(error_counts, model_name, output_path):
    """
    Gráfico de barras apiladas: Correcto vs Tipos de Error.
    error_counts: dict {'Correct': int, 'Mild': int, 'Medium': int, 'Severe': int}
    """
    labels = ['Correcto', 'Error Leve\n(Mismo Género)', 'Error Medio\n(Misma Familia)', 'Error Severo\n(Familia Distinta)']
    values = [
        error_counts.get('Correct', 0),
        error_counts.get('Mild', 0),
        error_counts.get('Medium', 0),
        error_counts.get('Severe', 0)
    ]
    total = sum(values)
    percentages = [v/total*100 for v in values]

    # Colores semánticos: Verde -> Amarillo -> Naranja -> Rojo
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, percentages, color=colors, edgecolor='black', alpha=0.8)

    plt.title(f"Desglose Taxonómico de Errores: {model_name}", fontweight='bold', color='#1b4f25')
    plt.ylabel("Porcentaje de Predicciones (%)")
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Etiquetas de valor
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    _save_figure(output_path)

def plot_ivc_performance(df_ivc, model_name, output_path):
    """
    Gráfico de barras agrupadas: Aciertos vs Fallos por categoría IVC.
    df_ivc columns: ['Category', 'Correct', 'Incorrect']
    """
    # Orden Específico
    ivc_order = ["Crítico", "Alto", "Medio", "Bajo", "Nulo"]

    # 1. Melt para tener formato largo (Correct/Incorrect)
    df_melt = df_ivc.melt(id_vars="Category", value_vars=["Correct", "Incorrect"], var_name="Status", value_name="Count")

    # 2. Calcular totales por categoría para normalizar
    # Agrupamos por categoría y sumamos los Counts (Correct + Incorrect)
    totals = df_melt.groupby('Category')['Count'].transform('sum')

    # 3. Calcular Porcentaje (evitando división por cero)
    df_melt['Percentage'] = np.where(totals > 0, (df_melt['Count'] / totals) * 100, 0)

    # 4. Ordenar visualización
    present_cats = [c for c in ivc_order if c in df_melt['Category'].unique()]
    others = [c for c in df_melt['Category'].unique() if c not in ivc_order]
    final_order = present_cats + others

    plt.figure(figsize=(12, 7))
    palette = {"Correct": "#2ecc71", "Incorrect": "#e74c3c"}

    # 5. Graficar Porcentaje
    ax = sns.barplot(
        data=df_melt,
        x="Category",
        y="Percentage",
        hue="Status",
        order=final_order,
        palette=palette,
        edgecolor="black",
        linewidth=0.8
    )

    plt.title(f"Desempeño Normalizado por IVC: {model_name}", fontweight='bold', color='#1b4f25')
    plt.xlabel("Categoría de Conservación")
    plt.ylabel("Porcentaje de Muestras (%)")
    plt.legend(title="Predicción", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0, 105) # Escala fija 0-100%
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 6. Etiquetas de valor (opcional, para claridad)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', fontsize=11, padding=3)

    _save_figure(output_path)
