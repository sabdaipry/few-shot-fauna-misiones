"""
Funciones de análisis post-benchmarking.

- analyze_taxonomic_errors: desglosa errores por nivel taxonómico
  (género, familia, severo).
- analyze_ivc_performance: agrupa aciertos/fallos por categoría del
  Índice de Valor de Conservación (IVC).
"""
import numpy as np
import pandas as pd


def _get_species_column(df_index):
    return 'species' if 'species' in df_index.columns else 'class_name'


def analyze_taxonomic_errors(y_true, y_pred, df_index, family_to_class=None):
    """
    Calcula el desglose de errores taxonómicos.
    Retorna un diccionario con conteos: Correct, Mild (Género), Medium (Familia),
    Severe (misma clase, familia distinta), Critical (clase distinta).

    family_to_class: dict {familia: clase} para distinguir Severe de Critical.
                     Si es None, todo lo que no es Mild/Medium cae en Severe.
    """
    species_col = _get_species_column(df_index)

    if species_col not in df_index.columns or 'genus' not in df_index.columns or 'family' not in df_index.columns:
        return {'Correct': 0, 'Mild': 0, 'Medium': 0, 'Severe': len(y_true), 'Critical': 0}

    s2g = dict(zip(df_index[species_col], df_index['genus']))
    s2f = dict(zip(df_index[species_col], df_index['family']))

    s2c = {}
    if family_to_class:
        for species, family in s2f.items():
            cls = family_to_class.get(family)
            if cls is not None:
                s2c[species] = cls

    counts = {'Correct': 0, 'Mild': 0, 'Medium': 0, 'Severe': 0, 'Critical': 0}

    for t, p in zip(y_true, y_pred):
        if t == p:
            counts['Correct'] += 1
        elif s2g.get(t) == s2g.get(p) and s2g.get(t) is not None:
            counts['Mild'] += 1
        elif s2f.get(t) == s2f.get(p) and s2f.get(t) is not None:
            counts['Medium'] += 1
        elif s2c.get(t) == s2c.get(p) and s2c.get(t) is not None:
            counts['Severe'] += 1
        else:
            counts['Critical'] += 1

    return counts


def analyze_ivc_performance(y_true, y_pred, df_index):
    """
    Calcula aciertos vs fallos agrupados por categoría de IVC.
    Retorna un DataFrame listo para graficar.
    """
    sp_col = _get_species_column(df_index)

    # 1. Crear mapeo Especie -> Categoría
    s2ivc = {}
    if 'ivc_category' in df_index.columns:
        s2ivc = dict(zip(df_index[sp_col], df_index['ivc_category']))
    elif 'ivc_score' in df_index.columns:
        # Fallback numérico
        def score_to_cat(s):
            """Mapea puntaje IVC numérico a categoría textual."""
            if s == 0:
                return "Nulo"  # 0 suele ser exótica
            if s <= 2:
                return "Bajo"
            if s < 4:
                return "Medio"
            if s < 5:
                return "Alto"
            return "Crítico"

        s2ivc = dict(zip(df_index[sp_col], df_index['ivc_score'].apply(score_to_cat)))
    else:
        return None  # No hay datos de IVC

    # 2. Función de normalización (Agrupamiento)
    def normalize_cat(c):
        """Colapsa categorías de especies exóticas/introducidas a 'Nulo'."""
        c_str = str(c).strip()
        if c_str in ['Doméstica', 'Exótica', 'Invasora', 'Exótica/Invasora', 'Introducida']:
            return 'Nulo'
        return c_str

    # 3. Procesar predicciones
    results = []
    for t, p in zip(y_true, y_pred):
        raw_cat = s2ivc.get(t, "Desconocido")
        cat = normalize_cat(raw_cat)
        status = "Correct" if t == p else "Incorrect"
        results.append({"Category": cat, "Status": status})

    if not results:
        return None

    df_res = pd.DataFrame(results)

    # 4. Agrupar y contar
    df_counts = df_res.groupby(["Category", "Status"]).size().reset_index(name="Count")

    # 5. Pivotar para asegurar formato (Category, Correct, Incorrect)
    df_pivot = df_counts.pivot(index="Category", columns="Status", values="Count").fillna(0).reset_index()

    # Asegurar que existan ambas columnas
    if 'Correct' not in df_pivot.columns:
        df_pivot['Correct'] = 0
    if 'Incorrect' not in df_pivot.columns:
        df_pivot['Incorrect'] = 0

    return df_pivot


def summarize_taxonomic_errors_by_backbone(records):
    """
    Agrega el desglose de errores taxonómicos por backbone, promediando sobre
    los clasificadores evaluados.

    records: list[dict] con claves 'Embedding Model', 'Classifier' y los
             conteos crudos de analyze_taxonomic_errors
             ('Correct', 'Mild', 'Medium', 'Severe', 'Critical').
    Retorna un DataFrame con columnas
    ['Embedding Model', 'Correct', 'Mild', 'Medium', 'Severe', 'Critical']
    (valores en %), ordenado descendente por % Correct.
    """
    cat_cols = ['Correct', 'Mild', 'Medium', 'Severe', 'Critical']
    if not records:
        return pd.DataFrame(columns=['Embedding Model'] + cat_cols)

    df = pd.DataFrame(records)
    totals = df[cat_cols].sum(axis=1)
    for c in cat_cols:
        df[c] = (df[c] / totals * 100).where(totals > 0, 0)

    df_agg = df.groupby('Embedding Model', as_index=False)[cat_cols].mean()
    df_agg = df_agg.sort_values('Correct', ascending=False).reset_index(drop=True)
    return df_agg


def summarize_ivc_performance_by_backbone(records):
    """
    Agrega el desempeño por categoría IVC por backbone, promediando sobre
    los clasificadores evaluados.

    records: list[pd.DataFrame], cada uno con columnas
             ['Embedding Model', 'Classifier', 'Category', 'Correct', 'Incorrect']
             (salida de analyze_ivc_performance con las columnas de
             identificación de backbone/clasificador agregadas).
    Retorna un DataFrame pivot: índice = Embedding Model, columnas =
    categorías IVC presentes en los datos, valores = % Correct promedio.
    """
    if not records:
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)
    total = df_all['Correct'] + df_all['Incorrect']
    df_all['Pct_Correct'] = np.where(total > 0, df_all['Correct'] / total * 100, np.nan)

    df_mean = (
        df_all.groupby(['Embedding Model', 'Category'])['Pct_Correct']
        .mean()
        .reset_index()
    )
    df_pivot = df_mean.pivot(index='Embedding Model', columns='Category', values='Pct_Correct')
    return df_pivot
