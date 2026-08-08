# 03-threshold-optimization: Calibración de umbrales

## Qué hace esta fase

Calibra el umbral de confianza que decide, en el pipeline en cascada, cuándo una predicción de BioCLIP v2 se acepta directamente y cuándo pasa al árbitro KNN por baja confianza. Módulo completo: el umbral está calibrado, validado sobre `query_test` y validado cualitativamente sobre fotogramas reales de cámara trampa.

**Qué produce:**

- El umbral de confianza calibrado (`thresholds.json` / `data/dev_test_split/thresholds_dev.json`), a partir de la distribución intraclase de distancias coseno al centroide.
- Validación cualitativa sobre fotogramas reales de cámara trampa (no imágenes curadas de iNaturalist), para confirmar que el umbral se comporta razonablemente fuera del dataset de benchmark.

---

## Umbral calibrado (valor final del pipeline de producción)

| Parámetro | Valor |
|---|---|
| Modelo | BioCLIP v2 |
| Métrica | Distancia coseno al centroide de la especie predicha |
| Cómputo del centroide | Media de embeddings L2-normalizados del gallery, re-normalizada |
| Umbral seleccionado | **0.1869** (percentil 95 de la distribución intraclase, calibrado sobre `query_dev`) |
| Cobertura a ese umbral | 94.97 % (query_dev) · 95.24 % (validado sobre `query_test` congelado) |
| Contaminación interclase | 57.21 % (query_dev) · 57.91 % (validado sobre `query_test` congelado) |

Se calibró también un umbral para DINOv2 Small (etapa de rechazo de fotogramas vacíos), pero **se eliminó del pipeline definitivo**: en validación cualitativa con fotogramas reales, la tasa de rechazo fue 0 % (distancias en rango 0.41–0.56, muy por debajo del umbral calibrado de 0.7594), no hay separabilidad real entre fauna y ausencia de fauna con ese modelo.

9 especies quedan excluidas del cómputo de percentiles por tener menos de 3 imágenes en el gallery (soporte insuficiente).

---

## Pipeline de scripts

### `01_calibrate_thresholds.py`

- **Qué hace:** para cada imagen del query set, calcula `dmin` (distancia coseno mínima al gallery de la misma especie), evalúa los percentiles 90/95/97/99 como candidatos a umbral, y elige el punto de equilibrio entre cobertura y contaminación interclase.
- **Input:** `02-benchmarking/data/dataset_index.csv` + embeddings en `02-benchmarking/data/features/`.
- **Output:** `data/thresholds.json`, figuras en `data/reports/figures/`.

### `02_qualitative_validation.py`

- **Qué hace:** clasifica dos carpetas de fotogramas reales de cámara trampa (Carpeta A: mayormente vacíos y fauna fuera de catálogo; Carpeta B: fauna conocida + casos borde como humanos) usando el pipeline en cascada, y organiza copias de las imágenes por resultado para inspección visual.
- **Input:** fotogramas reales (no versionados) + umbrales calibrados.
- **Output:** `data/resultados_validacion.csv`, `data/validacion_visual/` (no versionado), logs con el resumen.

### `dev_test_split/01_calibrate_thresholds_dev.py`

- **Qué hace:** misma calibración que `01_calibrate_thresholds.py`, pero restringida a `query_dev`. Es la versión que efectivamente se usa desde que el proyecto separó el query set (ver metodología dev/test/holdout en el README raíz). El umbral resultante (0.1869) es prácticamente igual al calibrado originalmente sobre el query set completo (0.18663, diferencia de 0.0003, dentro del ruido); `thresholds.json` (nombre canónico) se regeneró con este valor.
- **Output:** `data/dev_test_split/thresholds_dev.json`, figuras en `data/reports_dev/figures/`.

La validación congelada del umbral sobre `query_test` no tiene script propio en este módulo, se calculó junto con la evaluación final del clasificador en `02-benchmarking/scripts/dev_test_split/05_final_eval_test.py` (ver `data/final_holdout_evaluation/README.md`).

---

## Estructura de carpetas

```
03-threshold-optimization/
├── scripts/
│   ├── 01_calibrate_thresholds.py
│   ├── 02_qualitative_validation.py
│   └── dev_test_split/
│       └── 01_calibrate_thresholds_dev.py
├── data/
│   ├── thresholds.json                  # Umbral vigente (nombre canónico, calculado sobre query_dev)
│   ├── dev_test_split/thresholds_dev.json  # Misma calibración, generada por el script de dev_test_split
│   ├── resultados_validacion.csv        # Salida de la validación cualitativa
│   ├── reports/ , reports_dev/          # Figuras (distribuciones, cobertura por clase, boxplots)
│   └── final_holdout_evaluation/        # Puntero a la validación congelada (vive en 02-benchmarking)
└── logs/                                # Logs de ejecución (no versionados)
```

---

## Cómo reproducir

```bash
cd 03-threshold-optimization

# Calibración vigente (sobre query_dev)
python scripts/dev_test_split/01_calibrate_thresholds_dev.py

# Validación cualitativa sobre fotogramas reales propios (requiere carpetas locales, no versionadas)
python scripts/02_qualitative_validation.py
```

Requiere los embeddings de `02-benchmarking/data/features/` ya extraídos (ver `02-benchmarking/README.md`).
