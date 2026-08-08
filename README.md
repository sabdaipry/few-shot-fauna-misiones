# Automatización del análisis de cámaras trampa en la Selva Paranaense mediante few-shot learning

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Descripción del proyecto

El monitoreo de fauna silvestre mediante cámaras trampa es una herramienta fundamental para la conservación de ecosistemas, pero genera un **cuello de botella crítico**: el volumen de videos e imágenes capturados crece mucho más rápido que la capacidad humana para analizarlos. El análisis manual es lento, tedioso y propenso a errores (especialmente con especies pequeñas, nocturnas o parcialmente ocultas), lo que retrasa la disponibilidad de información para la toma de decisiones de conservación.

Este proyecto busca automatizar dicho análisis en el contexto específico de la **Selva Paranaense**, en la provincia de Misiones, Argentina, uno de los ecosistemas con mayor biodiversidad y mayor presión de conservación de Sudamérica, parte del Bosque Atlántico del Alto Paraná.

A diferencia de los enfoques clásicos de visión artificial, que requieren grandes datasets etiquetados por especie para entrenar un clasificador desde cero, aquí se adopta un enfoque de **few-shot learning basado en modelos fundacionales** (*foundation models*): se aprovechan embeddings ya entrenados sobre grandes corpus de imágenes (incluyendo modelos especializados en biología) y se entrena únicamente un clasificador liviano sobre esos embeddings. Esto permite obtener buen desempeño de clasificación **con relativamente pocas imágenes por especie**, sin necesidad de infraestructura de entrenamiento pesada ni de re-entrenar redes profundas completas.

---

## 🗂️ Estructura del repositorio

El proyecto está organizado en cuatro módulos secuenciales:

| Carpeta | Descripción | Estado |
|---|---|---|
| [`01-data-curation/`](01-data-curation/) | Herramienta de limpieza visual de datasets de imágenes (repositorio separado: [wildlife-image-dataset-curator](https://github.com/sabdaipry/wildlife-image-dataset-curator)) | ✅ **Completo** |
| [`02-benchmarking/`](02-benchmarking/) | Benchmark comparativo de 19 backbones × 7 clasificadores para identificar la mejor combinación embedding + clasificador | ✅ **Completo** |
| [`03-threshold-optimization/`](03-threshold-optimization/) | Calibración de umbrales del pipeline de clasificación en cascada | ✅ **Completo** |
| [`04-app/`](04-app/) | Aplicación de escritorio (SAREKO) para investigadores | 🔧 **En desarrollo** (pipeline de inferencia y GUI de las tres pestañas ya funcionales; pendiente una herramienta de evaluación contra ground truth y pulido de UI/UX) |

---

## 🏆 Resultados principales del benchmark

Se evaluaron **19 backbones** de extracción de embeddings combinados con **11 clasificadores** (209 combinaciones totales; la comparación principal reportada es de **15 backbones × 7 clasificadores clásicos = 105 combinaciones**, excluyendo 4 variantes diagnósticas DINO `_gap` y 4 variantes Faiss). El sistema completo opera **de forma local, sin GPU y sin servidores externos**, lo que lo hace viable para investigadores en campo con conectividad limitada.

### Mejor combinación encontrada

| Métrica | Valor |
|---|---|
| **Backbone** | BioCLIP v2 |
| **Clasificador** | Linear SVM |
| **Accuracy Top-1** | **89.33 %** (IC 95%: 88.0 % – 90.5 %) |
| **Accuracy Top-5** | **98.34 %** |
| **F1-macro** | **80.30 %** |

> Los intervalos de confianza se calcularon mediante *bootstrap* estratificado al 95 % (ver `02-benchmarking/scripts/07_bootstrap_ci.py`).

> **Nota**: esta tabla es el benchmark exploratorio original, usado para *elegir* arquitectura y
> clasificador, no es el número final a citar. El pipeline de producción usa BioCLIP v2 + Nearest
> Centroid (no Linear SVM, por extensibilidad del catálogo sin reentrenar); su número final,
> evaluado una sola vez sobre un conjunto nunca antes visto, está en la sección siguiente.

---

## 🧪 Metodología dev/test/holdout

Este proyecto separa el *query set* en dos partes para evitar sesgo de selección (*data snooping*):
usar el mismo dato tanto para elegir un modelo como para reportar su accuracy final infla el número
reportado, porque ese dato ya influyó en la elección.

| Split | Tamaño | Uso |
|---|---|---|
| `gallery` | 888 imágenes | Catálogo de referencia (sin cambios) |
| `query_dev` | 2572 imágenes (70 %) | Selección de arquitectura, clasificador y umbrales, de uso repetido |
| `query_test` | 1102 imágenes (30 %) | Evaluación final **congelada**, se evalúa **una sola vez** |

**Regla del proyecto de acá en adelante**: `query_test` no se vuelve a tocar salvo que se rehaga todo
el proceso de selección desde cero.

### Número final (evaluación held-out, una sola corrida)

| Métrica | BioCLIP v2 + Nearest Centroid (pipeline de producción) |
|---|---|
| **Accuracy Top-1** | **89.20 %** (IC95: 87.66 % – 90.83 %) |
| **F1-macro** | **85.38 %** (IC95: 82.55 % – 86.53 %) |

Sobre 1102 imágenes / 87 especies de `query_test`, nunca antes usadas para ninguna decisión de diseño.

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| **Fuente** | [iNaturalist](https://www.inaturalist.org/) |
| **Región de origen** | Bosque Atlántico del Alto Paraná |
| **Imágenes totales** | 4562 |
| **Especies** | 91 |
| **Familias** | 45 |

### Distribución por clase taxonómica

| Clase taxonómica | Familias |
|---|---|
| Mammalia | 19 |
| Aves | 24 |
| Reptilia | 2 |

---

## 🔬 Modelos evaluados

Se compararon backbones de cuatro familias arquitectónicas distintas, para evaluar si los modelos especializados en dominio biológico ofrecen una ventaja real frente a backbones genéricos:

| Familia | Modelos |
|---|---|
| **CNNs** | ResNet50, ConvNeXtV2 |
| **ViTs autosupervisados** | DINOv2, DINOv3 |
| **Multimodales visión-lenguaje** | CLIP, SigLIP, SigLIP2 |
| **Dominio biológico** | BioCLIP v1, BioCLIP v2 |

---

## ⚙️ Cómo reproducir el benchmark

Los siguientes pasos reproducen el pipeline completo del módulo `02-benchmarking/`, desde la indexación del dataset hasta el reporte final con intervalos de confianza.

```bash
cd 02-benchmarking

# Crear entorno virtual (o usar el venv de la raíz del repo)
python -m venv ../.venv

# Instalar PyTorch (versión CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias sin reinstalar torch
pip install -r requirements.txt --ignore-installed torch

# 1. Generar el índice del dataset
python scripts/01_generate_index.py

# 2. Extraer embeddings para cada backbone (repetir por cada modelo)
python scripts/02_extract_features.py --model <nombre_del_modelo>

# 3. Ejecutar el benchmark de clasificadores sobre los embeddings extraídos
python scripts/03_run_benchmark.py

# 4. Perfilar tiempos y recursos de cada backbone
python scripts/04_profile_backbones.py

# 5. Evaluar escalabilidad y comportamiento del sistema
python scripts/05_scalability_test.py
python scripts/05b_system_tests.py

# 6. Generar el reporte completo (HTML + figuras)
python scripts/06_generate_full_report.py

# 7. Calcular intervalos de confianza por bootstrap estratificado
python scripts/07_bootstrap_ci.py
```

---

## 👥 Créditos

| Rol | Nombre |
|---|---|
| **Autora** | Ing. Sabrina Daiana Pryszczuk |
| **Director** | Ing. Axel Alfredo Skrauba (DIEC-FI-UNaM) |
| **Institución** | Carrera de Especialización en Inteligencia Artificial, Facultad de Ingeniería, Universidad de Buenos Aires (FI-UBA) |

📍 Ciudad de Oberá, Misiones, Argentina, 2026

---

## 📄 Licencia

Este proyecto se distribuye bajo licencia [MIT](LICENSE).
