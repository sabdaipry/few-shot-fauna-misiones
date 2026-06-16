# 02-benchmarking — Benchmark de backbones y clasificadores few-shot

## Qué hace esta fase

Este módulo evalúa de forma sistemática qué combinación de **backbone de extracción de embeddings** y **clasificador clásico** ofrece el mejor desempeño para identificar especies de fauna de la Selva Paranaense a partir de pocas imágenes por especie (few-shot learning).

El pipeline parte de un dataset de imágenes organizado taxonómicamente, extrae embeddings con 19 backbones distintos (CNNs, ViTs autosupervisados, modelos visión-lenguaje y modelos de dominio biológico), entrena y evalúa 7 clasificadores por backbone sobre esos embeddings, mide latencia y escalabilidad, y consolida todo en un **reporte HTML interactivo** con tablas, gráficos y matrices de confusión.

**Qué produce:**

- Un índice del dataset (`dataset_index.csv`) con split galería/query por especie e IVC integrado.
- Embeddings precalculados por backbone (`data/features/`).
- Una tabla comparativa de métricas para las 133 combinaciones backbone × clasificador (`benchmark_summary.csv`).
- Intervalos de confianza bootstrap al 95% estratificados por clase (`bootstrap_ci.csv`).
- Mediciones de latencia, escalabilidad y comportamiento ante datos nuevos (incremental / outliers).
- Un reporte HTML final con leaderboards, heatmaps, UMAPs, matrices de confusión y análisis por clase taxonómica.

---

## Requisitos e instalación

- Python 3.12
- CPU es suficiente (todo el pipeline está pensado para ejecutarse sin GPU)

Se recomienda instalar `torch` **antes** del resto de las dependencias, apuntando al índice de PyTorch para CPU, y luego instalar el resto sin que `pip` intente reinstalarlo con otra build:

```bash
cd 02-benchmarking

# 1. Crear entorno virtual (o usar el venv de la raíz del repo)
python -m venv ../.venv

# 2. Instalar PyTorch (build CPU) desde su índice propio
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Instalar el resto de dependencias sin reinstalar torch
pip install -r requirements.txt --ignore-installed torch
```

> ⚠️ Si se omite el paso 2 y se instala `torch` directamente desde `requirements.txt`, `pip` puede traer la build con soporte CUDA (mucho más pesada e innecesaria si no hay GPU disponible).

---

## Estructura de carpetas

```
02-benchmarking/
├── src/                  # Módulos reutilizables (lógica de negocio)
├── scripts/              # Pipeline ejecutable, paso a paso (01 a 07)
│   └── helpers/          # Scripts auxiliares de mantenimiento
├── data/                 # Datos de entrada, intermedios y resultados
│   ├── benchmark_results/    # CSVs de salida del benchmark (summary, predicciones, bootstrap CI)
│   ├── features/             # Embeddings por backbone (.npy) — no versionados, ver nota abajo
│   ├── fauna_seleccionada_bosque_atlantico/  # Imágenes del dataset y metadatos — no versionado
│   ├── reports/               # Reporte HTML final + figuras generadas
│   │   └── figures/           # PNG/SVG de cada gráfico del reporte
│   └── results/               # Resultados intermedios de pruebas puntuales
├── logs/                 # Logs de ejecución de cada script
└── requirements.txt
```

| Carpeta | Contenido |
|---|---|
| `src/` | Configuración, extractores de embeddings, clasificadores, análisis, visualización y generación de reportes |
| `scripts/` | Los 7 pasos del pipeline, numerados en orden de ejecución |
| `scripts/helpers/` | Utilidades puntuales de mantenimiento de datos (no forman parte del pipeline principal) |
| `data/` | Todos los datos del proyecto: índice, embeddings, resultados y reportes |
| `logs/` | Logs estructurados generados por `src/utils/logger.py` en cada corrida |

---

## Pipeline completo

Los scripts se ejecutan en orden, ya que cada uno consume el output del anterior.

### `01_generate_index.py`

- **Qué hace:** escanea recursivamente las imágenes del dataset, integra los puntajes del Índice de Valor de Conservación (IVC) y asigna un split galería/query por especie.
- **Input:** imágenes en `data/fauna_seleccionada_bosque_atlantico/images/`, `data/Indice_Valor_Conservacion_Misiones.csv`, `data/manual_fixes.yaml`.
- **Output:** `data/dataset_index.csv` — índice maestro con una fila por imagen (especie, familia, IVC, split galería/query).

### `02_extract_features.py`

- **Qué hace:** extrae embeddings de cada imagen del índice usando el backbone indicado, y los guarda replicando la estructura de carpetas del dataset.
- **Input:** `data/dataset_index.csv` + el nombre del modelo (`--model <nombre>`, ver `MODELS_TO_TEST` en `src/config.py`).
- **Output:** archivos `.npy` por imagen en `data/features/<modelo>/`.

```bash
python scripts/02_extract_features.py --model bioclip_v2
```

> Se ejecuta una vez por cada backbone a evaluar.

### `03_run_benchmark.py`

- **Qué hace:** corre el benchmark de clasificación few-shot: para cada backbone con embeddings ya extraídos, entrena los 7 clasificadores sobre el split galería y evalúa sobre el split query.
- **Input:** `data/dataset_index.csv` + `data/features/<modelo>/`.
- **Output:** `data/benchmark_results/benchmark_summary.csv` (tabla general de métricas) y un `predictions_<modelo>.csv` por backbone con las predicciones fila a fila (insumo del bootstrap CI).

### `04_profile_backbones.py`

- **Qué hace:** perfila la latencia de inferencia pura de cada backbone (forward pass sobre una imagen de muestra, sin contar la carga del modelo).
- **Input:** `data/dataset_index.csv` (una imagen de muestra) + los backbones configurados.
- **Output:** `data/backbones_times.csv`.

### `05_scalability_test.py`

- **Qué hace:** mide cómo se degradan las métricas y la latencia del clasificador Nearest Centroid a medida que aumenta el número de especies del dataset (escalones: 5, 10, 20, 30, 50, 70, 91).
- **Input:** `data/dataset_index.csv` + `data/features/`.
- **Output:** `data/scalability_results.csv`.

### `05b_system_tests.py`

- **Qué hace:** corre dos pruebas de sistema sobre los modelos top del benchmark (Top 5 por F1-macro + Top 5 más rápidos): actualización incremental (agregar especies nuevas sin reentrenar) y detección de outliers / Open Set Recognition.
- **Input:** `data/benchmark_results/benchmark_summary.csv`, `data/backbones_times.csv`, `data/features/`.
- **Output:** `data/incremental_results.csv` y `data/outlier_results.csv`.

### `06_generate_full_report.py`

- **Qué hace:** orquesta la generación de todos los gráficos (`src/visualization.py`), los análisis post-benchmarking (`src/analysis.py`) y el dashboard HTML final (`src/reporting.py`), consolidando todos los resultados previos.
- **Input:** todos los CSVs generados por los pasos 01–05b.
- **Output:** `data/reports/benchmark_report.html` + figuras en `data/reports/figures/`.

### `07_bootstrap_ci.py`

- **Qué hace:** calcula intervalos de confianza bootstrap al 95% **estratificados por clase** para Accuracy y F1-macro, por cada combinación backbone × clasificador, reutilizando las predicciones ya generadas (no re-ejecuta extracción ni clasificación).
- **Input:** `data/benchmark_results/predictions_*.csv`.
- **Output:** `data/benchmark_results/bootstrap_ci.csv` + figura `10_bootstrap_forest.png/svg` (forest plot).

```bash
python scripts/07_bootstrap_ci.py --iterations 1000 --seed 29 --confidence 0.95
```

> ⚠️ Este script documenta explícitamente sus limitaciones metodológicas en su docstring (varianza solo del query set, comparaciones no apareadas, clases singleton, clasificadores correlacionados); vale la pena leerlas antes de citar los IC en el reporte académico.

---

## Módulos reutilizables (`src/`)

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Configuración centralizada: raíz del proyecto, lista de modelos a evaluar (`MODELS_TO_TEST`) y todos los paths derivados de `data/`. |
| `backbones.py` | Extractores de embeddings para los 19 backbones (CNNs, ViTs autosupervisados, multimodales visión-lenguaje y de dominio biológico). Todos heredan de `BaseModel` y exponen `load_model()` y `get_embedding(image_path)`. |
| `benchmarking.py` | Clasificadores personalizados (`FaissKNNClassifier`, `FaissNearestCentroid`) y `ModelEvaluator`, que orquesta la carga de embeddings, el entrenamiento/evaluación de clasificadores y la persistencia de resultados. |
| `analysis.py` | Análisis post-benchmarking: desglose de errores por nivel taxonómico, desempeño por categoría IVC y accuracy/error crítico por clase taxonómica (Mammalia/Aves/Reptilia). |
| `visualization.py` | Generación de todas las figuras del reporte (leaderboards, heatmaps, Pareto, UMAPs, matrices de confusión, forest plot) con Matplotlib/Seaborn en backend headless. |
| `reporting.py` | Construcción del dashboard HTML final a partir de las métricas y figuras generadas. |
| `utils/logger.py` | Logging estructurado (consola + archivo) usado por todos los scripts del pipeline. |

---

## Configuración

| Archivo | Rol |
|---|---|
| `src/config.py` | Fuente central de verdad de paths (`DATA_DIR`, `FEATURES_DIR`, `BENCHMARK_RESULTS_DIR`, `REPORTS_DIR`, etc.) y de la lista canónica de modelos a evaluar (`MODELS_TO_TEST`, derivada de `MODEL_REGISTRY` en `src/backbones.py`). Todos los scripts importan estas constantes en lugar de duplicar rutas. |
| `data/manual_fixes.yaml` | Correcciones manuales de nombres taxonómicos: `aliases` (subespecies de las carpetas de imágenes que deben mapearse a la especie principal del CSV de IVC) y `hardcoded` (especies sin entrada en el CSV de IVC, a las que se les asigna `ivc_score`/`ivc_category` directamente). |
| `data/taxonomic_class_mapping.yaml` | Mapeo de familia biológica → clase taxonómica (Mammalia / Aves / Reptilia), usado por `src/analysis.py` para el análisis de errores cross-clase. |

---

## Resultados

| Qué buscás | Dónde está |
|---|---|
| Reporte interactivo completo (leaderboards, heatmaps, UMAPs, matrices de confusión, análisis por clase taxonómica) | [`data/reports/benchmark_report.html`](data/reports/benchmark_report.html) |
| Figuras individuales (PNG + SVG) usadas en el reporte | `data/reports/figures/` |
| Tablas de métricas en CSV (resumen general, predicciones por backbone, intervalos de confianza bootstrap) | `data/benchmark_results/` |

---

## Nota sobre datos no incluidos en el repositorio

Por su tamaño, los siguientes directorios **se excluyen del control de versiones** (ver `.gitignore`):

- `data/features/` — embeddings extraídos por backbone (`.npy`).
- `data/fauna_seleccionada_bosque_atlantico/` — imágenes originales del dataset y sus metadatos.

Ambos son completamente **reproducibles** ejecutando el pipeline desde cero (`01_generate_index.py` → `02_extract_features.py`) sobre el dataset fuente de iNaturalist descripto en el README de la raíz del repositorio.
