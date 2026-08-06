# Evaluación final held-out (una sola corrida)

El archivo real (`threshold_validation_test.csv`) no vive en esta carpeta para evitar
que existan dos copias que puedan divergir. Está en:

```
02-benchmarking/data/final_holdout_evaluation/threshold_validation_test.csv
```

Generado por `02-benchmarking/scripts/dev_test_split/05_final_eval_test.py` — no hay
un script propio en `03-threshold-optimization` para esta validación del umbral sobre
`query_test`; el umbral se calibra acá (`thresholds.json`, sobre `query_dev`) pero su
validación congelada sobre el holdout se calculó junto con la evaluación final del
clasificador en `02-benchmarking`.
