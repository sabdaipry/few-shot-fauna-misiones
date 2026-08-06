# Informe — Re-validación metodológica (split dev/test del query set)

**Fecha**: 2026-07-15
**Motivación**: el query set completo (3674 imágenes) se usaba tanto para tomar todas las
decisiones de diseño (backbone, clasificador, métrica de distancia, umbrales) como para
reportar el accuracy final de esas mismas decisiones — sesgo de selección (data snooping).
Este trabajo separa "query" en `query_dev` (70%, todas las decisiones de diseño) y
`query_test` (30%, congelado, tocado una sola vez para el número final).

---

## 1. Qué cambió y qué no

### No se tocó
- `dataset_index.csv` original (se generó un archivo nuevo, no se sobrescribió).
- `benchmark_summary.csv`, `predictions_*.csv`, `bootstrap_ci.csv`, `distance_benchmark.csv`,
  `thresholds.json` originales — intactos.
- El split de **gallery** (888 imágenes, 91 especies) — sin cambios, sigue siendo el catálogo
  de referencia completo.
- `04-app` — no se tocó nada, como se pidió.

### Se agregó (todo nuevo, nada sobrescrito)
- `02-benchmarking/data/dev_test_split/dataset_index_dev_test_split.csv` — mismo esquema,
  `split` ahora toma `gallery` / `query_dev` / `query_test`.
- `02-benchmarking/scripts/dev_test_split/01_generate_dev_test_index.py` … `05_final_eval_test.py`
- `03-threshold-optimization/scripts/dev_test_split/01_calibrate_thresholds_dev.py`
- Parametrización mínima y no invasiva de `src/benchmarking.py` (`ModelEvaluator`):
  parámetro opcional `test_split_value` (default `None` = comportamiento original intacto) y
  parámetro opcional `classifier_names` en `evaluate_model` (default `None` = los 11
  clasificadores de siempre). `03_run_benchmark.py` original sigue funcionando exactamente
  igual, sin flags, sin cambios de comportamiento.
- Resultados nuevos en `data/benchmark_results_dev/`, `data/benchmark_results_test/`,
  `data/dev_test_split/` (03-threshold-optimization), `data/reports_dev/`.

---

## 2. Split del query set

- Estratificado por especie, semilla **29** (consistente con el resto del repo).
- Regla para especies con muy pocas imágenes: si `query_count == 1`, la especie queda entera
  en `query_dev` y excluida de `query_test` (no se puede garantizar ≥1 en cada lado).
  Para el resto (`n ≥ 2`): `n_test = clamp(round(n·0.3), 1, n-1)`, garantizando al menos 1
  imagen en cada lado.

**Especies excluidas de `query_test` (4, todas con `query_count == 1`)**:

| Especie | query_count |
|---|---|
| *Conopophaga lineata vulgaris* | 1 |
| *Leopardus pardalis mitis* | 1 |
| *Strix virgata borelliana* | 1 |
| *Tapirus terrestris terrestris* | 1 |

Nota: esta lista es distinta de las "9 especies con soporte insuficiente" que ya se excluían
en `03-threshold-optimization` por tener `gallery_count < 3` (esas 9 incluyen a estas 4, más
5 adicionales: Axis axis, Cabassous tatouay, Cryptopezus nattereri, Puma concolor concolor,
Sus scrofa domesticus). Son dos criterios de exclusión conceptualmente distintos y no deben
confundirse.

**Resultado del split**: gallery 888 (sin cambios) · query_dev 2572 (70.0%) · query_test 1102
(30.0%) · 87 de 91 especies representadas en `query_test`.

---

## 3. Benchmark backbone × clasificador sobre `query_dev`

15 backbones estándar (excluidas las 4 variantes DINO `_gap`, experimento diagnóstico aparte)
× 7 clasificadores (excluidas las 4 variantes Faiss, prueba de latencia aparte).

**Ganador confirmado: BioCLIP v2 — sin cambio de backbone respecto al pipeline actual.**

| | Top-1 (query completo, original) | Top-1 (query_dev) | Top-1 (query_test, congelado) |
|---|---|---|---|
| **BioCLIP v2 + Linear SVM** | 89.33% | 89.50% | 88.93% |
| **BioCLIP v2 + Nearest Centroid** | 89.25% | 89.27% | **89.20%** |

Sobre `query_test` (el número que realmente importa), **Nearest Centroid termina levemente
por delante de Linear SVM** (89.20% vs 88.93%) — se invierte el orden que tenían en dev, lo
cual confirma que la diferencia entre ambos siempre estuvo dentro del ruido de muestreo, no es
una ventaja real y estable de Linear SVM. Nearest Centroid además tiene mejor F1-macro en las
tres evaluaciones (83.26% / 83.21% / **85.38%** vs 80.30% / 80.08% / 83.98% de Linear SVM).

### Bootstrap CI (95%, estratificado, 1000 iteraciones, seed=29)

| | query completo (n=3674) | query_dev (n=2572) |
|---|---|---|
| Nearest Centroid | 89.27% [88.35–90.12%] | 89.28% [88.22–90.28%] |
| Linear SVM | ~89.34% (IC no recalculado en este informe) | 89.49% [88.57–90.44%] |

Los intervalos de Nearest Centroid y Linear SVM se solapan ampliamente en ambos esquemas —
no hay evidencia estadística de que uno sea mejor que el otro. Se aplicaron las mismas 4
limitaciones metodológicas documentadas en el docstring de `07_bootstrap_ci.py` (varianza solo
del query set, comparaciones apareadas no explotadas, clases singleton, clasificadores
correlacionados) también a la corrida sobre `query_dev`.

---

## 4. Selección de métrica de distancia

**Ubicación real**: `02-benchmarking/scripts/08_distance_benchmark.py` (no en
`03-threshold-optimization`, como se asumía inicialmente — corregido en el Paso 0).

| | query completo (original) | query_dev |
|---|---|---|
| BioCLIP v2 + coseno | 87.18% (accuracy 1-NN) | 86.86% |
| BioCLIP v2 + euclidiana (cruda) | **87.26%** (ganador nominal) | **87.01%** (ganador nominal) |
| Diferencia | +0.08 pp | +0.15 pp |

En ambos esquemas la euclidiana cruda queda técnicamente primera, por un margen mínimo (dentro
del ruido, <1 pp) — esto **ya estaba así en el benchmark original**, no es un cambio introducido
por el re-split. La arquitectura completa (centroides L2-normalizados, umbrales, árbitro KNN
ponderado por 1/distancia) está construida sobre geometría coseno de punta a punta; cambiar de
métrica ahora sería un cambio de arquitectura no solicitado. Se mantiene coseno. Recomiendo
documentar explícitamente esta salvedad en la tesis en vez de decir sin matices "se seleccionó
coseno por benchmark", ya que el número crudo dice lo contrario por un margen despreciable.

---

## 5. Calibración de umbrales sobre `query_dev`

| Parámetro | Original (query completo) | Recalibrado (query_dev) |
|---|---|---|
| Umbral 2, Centroide, p95 | 0.18663 | **0.18693** |
| Coverage p95 | 94.997% | 94.97% |
| Contaminación inter-clase p95 | 56.65% | 57.21% |
| Separation gap p95 | -0.01132 | -0.0118 |

El umbral es **muy estable** (diferencia de 0.0003, ruido). Confirmado también sobre
`query_test` congelado (ver sección 6): coverage 95.24%, contaminación 57.91%, gap -0.0122 —
mismo orden de magnitud en las tres evaluaciones.

### Hallazgo de documentación (independiente del re-split)

`03-threshold-optimization/CLAUDE.md` afirma: *"Se seleccionó p95 para BioCLIP v2 ... con gap
aún positivo (+0.011)"*. Verificado contra `thresholds.json` real: **el gap en p95 (variante
Centroide) es -0.01132, negativo**, no positivo. El valor "+0.011" corresponde a la fila de
**p90** (variante 1-NN: +0.0107), no a p95. Mismo patrón de error de transcripción de fila que
el hallazgo del Paso 5 (87.18%). Recomiendo corregir esa frase en la documentación.

---

## 6. Evaluación final única sobre `query_test` (congelada)

Corrida **una sola vez**, después de fijar arquitectura/clasificador/métrica/umbral solo con
`query_dev`. No se volvió a tocar `query_test` después de esta corrida.

| Métrica | Nearest Centroid | Linear SVM |
|---|---|---|
| Top-1 | **89.20%** | 88.93% |
| Top-5 | 98.64% | 98.37% |
| F1-macro | **85.38%** | 83.98% |
| Correcto | 89.20% (983/1102) | 88.93% (980/1102) |
| Error leve (género) | 5.90% | 5.26% |
| Error medio (familia) | 1.81% | 2.09% |
| Error severo (clase) | 2.81% | 3.45% |
| Error crítico | 0.27% | 0.27% |

**Validación del umbral de confianza** (p95 centroide = 0.18693, calibrado en `query_dev`,
aplicado congelado sobre `query_test`): coverage 95.24%, contaminación inter-clase 57.91%,
separation gap -0.0122. Consistente con dev y con el original — el umbral generaliza bien.

### Bootstrap CI (95%, estratificado, 1000 iteraciones, seed=29) sobre `query_test`

Mismo script y misma semilla que en las corridas sobre query completo y `query_dev` (sección
3), aplicado ahora sobre las predicciones congeladas de `query_test`.

**Aclaración metodológica importante (detectada por revisión posterior):** la columna `Mean`
de `bootstrap_ci_test.csv` (y de `bootstrap_ci.csv` / `bootstrap_ci_dev.csv` originales — es un
comportamiento heredado de `compute_bootstrap_ci()` en `07_bootstrap_ci.py`, no algo introducido
en este informe) es el **promedio de la métrica sobre las 1000 remuestras bootstrap**, no la
métrica calculada una sola vez sobre las 1102 imágenes reales del test set. Para **Accuracy**
(estadístico lineal) esta diferencia es despreciable (89.2015% puntual vs 89.2028% media
bootstrap, sesgo de 0.001pp). Para **F1-macro** (promedio armónico no lineal sobre 87 clases,
varias con muy pocas muestras) el sesgo es real y no despreciable:

| Clasificador | F1-macro puntual (1102 imgs) | F1-macro media bootstrap | Sesgo (bootstrap − puntual) |
|---|---|---|---|
| Nearest Centroid | 85.38% | 86.28% | **+0.89 pp** |
| Linear SVM | 83.98% | 83.75% | −0.22 pp |

El intervalo percentil reportado en `bootstrap_ci_test.csv` (`CI_lower`/`CI_upper`) sigue siendo
un IC95 válido en el sentido "percentile bootstrap" — son los percentiles 2.5/97.5 de la
distribución de remuestreo, útiles para cuantificar cuánto variaría la métrica con otra muestra
comparable — pero están centrados alrededor de la **media bootstrap**, no del punto observado.
Para reportar un intervalo centrado explícitamente en el valor puntual de la sección 6 (el que
realmente se citaría como resultado), se recalculó con el método "basic bootstrap"
(`CI = [2·puntual − percentil_97.5, 2·puntual − percentil_2.5]`), que conserva el ancho del
intervalo pero lo recentra sobre el estadístico observado:

| Clasificador | Métrica | Puntual (1102 imgs) | IC95 percentil (bootstrap_ci_test.csv) | IC95 "basic", centrado en el puntual |
|---|---|---|---|---|
| **Nearest Centroid** | Accuracy | 89.20% | [87.57%, 90.74%] | [87.66%, 90.83%] |
| **Nearest Centroid** | F1-macro | **85.38%** | [84.23%, 88.21%] (sesgado +0.89pp) | **[82.55%, 86.53%]** |
| **Linear SVM** | Accuracy | 88.93% | [87.39%, 90.38%] | [87.48%, 90.47%] |
| **Linear SVM** | F1-macro | 83.98% | [81.63%, 85.79%] | [82.16%, 86.32%] |

Para Accuracy la diferencia entre ambos métodos es prácticamente nula (~0.1pp de corrimiento,
ruido). Para F1-macro de Nearest Centroid sí importa: el intervalo correcto, centrado en el
85.38% puntual, es **[82.55%, 86.53%]**, no [84.23%, 88.21%].

Con esto, los intervalos de Accuracy de ambos clasificadores se solapan ampliamente — no hay
diferencia estadísticamente significativa. El número final a citar en la tesis para el pipeline
de producción (Nearest Centroid) es:

> **Top-1 = 89.20% (IC95: 87.66% – 90.83%)** · **F1-macro = 85.38% (IC95: 82.55% – 86.53%)**,
> sobre un test set de 1102 imágenes / 87 especies, nunca antes usado para ninguna decisión de
> diseño.

Archivos: `02-benchmarking/data/benchmark_results_test/bootstrap_ci_test.csv` (IC percentil,
tal como lo produce el script original) y
`bootstrap_ci_test_point_vs_bootstrap.csv` (comparación puntual vs bootstrap y el IC "basic"
recentrado, ambos clasificadores y métricas).

**Nota**: el mismo sesgo entre `Mean` (bootstrap) y el valor puntual real aplica, sin haberse
verificado explícitamente en este informe, a `bootstrap_ci.csv` (query completo) y
`bootstrap_ci_dev.csv` — es una propiedad de `compute_bootstrap_ci()` en sí, no algo específico
de la corrida sobre test. Si se va a citar el F1-macro de esos archivos en la tesis, conviene
aplicar la misma verificación puntual-vs-bootstrap antes de reportarlo.

### Nota sobre cobertura de especies en `query_test`

`query_test` contiene **87 de las 91 especies** del catálogo (95.6% de las especies). Las 4
especies ausentes son exactamente las 4 excluidas del split por tener solo 1 imagen de query
(sección 2): *Conopophaga lineata vulgaris*, *Leopardus pardalis mitis*, *Strix virgata
borelliana* y *Tapirus terrestris terrestris* — quedaron enteras en `query_dev` y no
contribuyen imágenes a ningún cómputo de accuracy sobre test. Confirmado directamente contando
clases únicas en `predictions_bioclip_v2_test.csv` (87 valores únicos de `y_true` sobre 1102
filas).

---

## 7. Respuestas del Paso 5

### 5a — Origen del Top-1 "BioCLIP v2 + Centroide" (87.18%)

**El 87.18% documentado en el CLAUDE.md raíz y en `02-benchmarking/CLAUDE.md` como accuracy de
"BioCLIP v2 + Centroide" es en realidad el accuracy de 1-NN (KNN k=1), no el de Nearest
Centroid.**

| Fuente | Clasificador | Top-1 |
|---|---|---|
| `benchmark_summary.csv` | Nearest Centroid (sklearn, euclidiana sobre L2-norm) | 89.25% |
| `benchmark_summary.csv` | Faiss Nearest Centroid (coseno) | 89.11% |
| `benchmark_summary.csv` | **KNN (k=1)** | **87.1802%** ← coincide con el "87.18%" documentado |
| `benchmark_summary.csv` | Faiss KNN (k=1) | 87.1802% (idéntico) |
| `bootstrap_ci.csv` | Nearest Centroid | 89.27% [88.35–90.12%] |

No hay ningún script o notebook que mezcle ambos resultados — es un error de transcripción
manual en la documentación (se tomó la fila equivocada de `benchmark_summary.csv`). La
diferencia real Linear SVM vs Nearest Centroid es de ~0.08 pp (89.33% vs 89.25%), no ~2 pp como
sugiere el "87.18%" incorrecto. Tabla de extracción completa (query set completo, sin re-split)
en `02-benchmarking/data/benchmark_results/taxonomic_error_breakdown_bioclip_v2_nc_vs_svm.csv`.

### 5b — Comparación contra resultados 1-NN existentes

El único resultado 1-NN de BioCLIP v2 en el repo es la fila `KNN (k=1)` /
`Faiss KNN (k=1)` de `benchmark_summary.csv` (87.18%, idéntica en ambas implementaciones dado
que 1-NN coseno da el mismo resultado independientemente del framework de búsqueda). Es
exactamente la cifra que terminó documentada por error como resultado de "Centroide".

---

## 8. Recomendaciones de corrección de documentación

1. `CLAUDE.md` (raíz) y `02-benchmarking/CLAUDE.md`: cambiar "BioCLIP v2 + Centroide: 87.18%"
   por **89.25%** (o 89.20% si se prefiere citar el número ya validado sin sesgo sobre
   `query_test`). Ajustar el texto "pérdida de accuracy (~2pp) aceptable" — la pérdida real
   es de ~0.1pp, y en el número validado sin sesgo Centroide de hecho **gana** a Linear SVM.
2. `03-threshold-optimization/CLAUDE.md`: corregir "gap aún positivo (+0.011)" → el gap en p95
   (la variante y percentil efectivamente usados) es -0.0113 (negativo). El +0.011 es la cifra
   de p90, no de p95.
3. `02-benchmarking/CLAUDE.md`: "133 combinaciones (19×7)" está desactualizado — hay 19
   backbones × 11 clasificadores = 209 combinaciones reales en `benchmark_summary.csv` (los 4
   adicionales son las variantes Faiss).
4. Considerar documentar explícitamente que la selección de "coseno" como métrica de distancia
   no es el resultado numérico estrictamente ganador en `08_distance_benchmark.py` (la
   euclidiana cruda queda nominalmente primera por <0.2pp), sino una decisión de consistencia
   arquitectónica.

---

## 9. Archivos generados (todos nuevos)

```
02-benchmarking/
├── data/dev_test_split/
│   ├── dataset_index_dev_test_split.csv
│   └── excluded_species_query_singletons.csv
├── data/benchmark_results_dev/
│   ├── benchmark_summary_dev.csv
│   ├── predictions_<15 backbones>_dev.csv
│   ├── bootstrap_ci_dev.csv
│   ├── distance_benchmark_dev.csv
│   └── distance_distributions_dev.npz
├── data/benchmark_results_test/
│   ├── benchmark_summary_test.csv
│   ├── predictions_bioclip_v2_test.csv
│   ├── taxonomic_error_breakdown_test.csv
│   ├── threshold_validation_test.csv
│   ├── bootstrap_ci_test.csv
│   └── bootstrap_ci_test_point_vs_bootstrap.csv
├── data/benchmark_results/
│   └── taxonomic_error_breakdown_bioclip_v2_nc_vs_svm.csv   (extracción Paso 5, query completo)
├── data/reports_dev/figures/...
└── scripts/dev_test_split/
    ├── 01_generate_dev_test_index.py
    ├── 02_run_benchmark_dev.py
    ├── 03_distance_benchmark_dev.py
    ├── 04_bootstrap_ci_dev.py
    ├── 05_final_eval_test.py
    └── 06_bootstrap_ci_test.py

03-threshold-optimization/
├── data/dev_test_split/thresholds_dev.json
├── data/reports_dev/figures/...
└── scripts/dev_test_split/01_calibrate_thresholds_dev.py
```
