# Automatización del análisis de cámaras trampa en la Selva Paranaense mediante few-shot learning

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Descripción del proyecto

El monitoreo de fauna silvestre mediante cámaras trampa es una herramienta fundamental para la conservación de ecosistemas, pero genera un **cuello de botella crítico**: el volumen de videos e imágenes capturados crece mucho más rápido que la capacidad humana para analizarlos. El análisis manual es lento, tedioso y propenso a errores —especialmente con especies pequeñas, nocturnas o parcialmente ocultas—, lo que retrasa la disponibilidad de información para la toma de decisiones de conservación.

Este proyecto busca automatizar dicho análisis en el contexto específico de la **Selva Paranaense**, en la provincia de Misiones, Argentina —uno de los ecosistemas con mayor biodiversidad y mayor presión de conservación de Sudamérica, parte del Bosque Atlántico del Alto Paraná—.

A diferencia de los enfoques clásicos de visión artificial, que requieren grandes datasets etiquetados por especie para entrenar un clasificador desde cero, aquí se adopta un enfoque de **few-shot learning basado en modelos fundacionales** (*foundation models*): se aprovechan embeddings ya entrenados sobre grandes corpus de imágenes (incluyendo modelos especializados en biología) y se entrena únicamente un clasificador liviano sobre esos embeddings. Esto permite obtener buen desempeño de clasificación **con relativamente pocas imágenes por especie**, sin necesidad de infraestructura de entrenamiento pesada ni de re-entrenar redes profundas completas.

---

## 🗂️ Estructura del repositorio

El proyecto está organizado en cuatro módulos secuenciales:

| Carpeta | Descripción | Estado |
|---|---|---|
| [`01-data-curation/`](01-data-curation/) | Herramienta de limpieza visual de datasets de imágenes (ver repositorio separado: [wildlife-image-dataset-curator](https://github.com/sabdaipry/wildlife-image-dataset-curator)) | — |
| [`02-benchmarking/`](02-benchmarking/) | Benchmark comparativo de 19 backbones × 7 clasificadores para identificar la mejor combinación embedding + clasificador | ✅ **Completo** |
| [`03-threshold-optimization/`](03-threshold-optimization/) | Calibración de umbrales del pipeline de clasificación en cascada | 🔧 **En desarrollo** |
| [`04-app/`](04-app/) | Aplicación de escritorio final para investigadores | 🔜 **Próximamente** |

---

## 🏆 Resultados principales del benchmark

Se evaluaron **19 backbones** de extracción de embeddings combinados con **7 clasificadores** clásicos, totalizando **133 combinaciones**. El sistema completo opera **de forma local, sin GPU y sin servidores externos**, lo que lo hace viable para investigadores en campo con conectividad limitada.

### Mejor combinación encontrada

| Métrica | Valor |
|---|---|
| **Backbone** | BioCLIP v2 |
| **Clasificador** | Linear SVM |
| **Accuracy Top-1** | **89.33 %** (IC 95%: 88.0 % – 90.5 %) |
| **Accuracy Top-5** | **98.34 %** |
| **F1-macro** | **80.30 %** |

> Los intervalos de confianza se calcularon mediante *bootstrap* estratificado al 95 % (ver `02-benchmarking/scripts/07_bootstrap_ci.py`).

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
| **Institución** | Carrera de Especialización en Inteligencia Artificial — Facultad de Ingeniería, Universidad de Buenos Aires (FI-UBA) |

📍 Ciudad de Oberá, Misiones, Argentina — 2026

---

## 📄 Licencia

Este proyecto se distribuye bajo licencia [MIT](LICENSE).
