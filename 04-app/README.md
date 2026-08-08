# 04-app: SAREKO, aplicación de escritorio

## Qué es

**SAREKO** es la aplicación de escritorio del proyecto: recibe videos o imágenes de cámaras trampa y devuelve un registro biológico por especie, sin depender de GPU ni de conexión a internet durante el procesamiento. Está pensada para biólogos, ecólogos y guardaparques sin formación técnica en IA.

Se reescribe desde cero (arquitectura limpia, PySide6) tomando como referencia funcional un MVP monolítico anterior, cuyo código no se reutiliza.

## Estado actual

En desarrollo activo. Ya están implementados:

- **Pipeline de inferencia completo** (`inference/pipeline.py`): filtro de movimiento (MOG2) como pre-filtro antes de invocar el backbone, extracción de embeddings con BioCLIP v2 (`open_clip`), clasificación por centroide con árbitro KNN para los casos de baja confianza, y consenso temporal por ventana deslizante (`SlidingWindowConsensus`) sobre los frames de video.
- **GUI con las tres pestañas** especificadas (Análisis / Validación / Evaluación), ver `src/gui/tabs/`.
- **Persistencia de sesión con restauración automática** (`src/data/session.py`): la app recarga sola el último análisis al abrirse, sin necesidad de una acción manual de "cargar reporte", y mantiene un historial de corridas.
- **Explicabilidad visual** (Attention Rollout y GradCAM sobre BioCLIP v2) para inspeccionar en qué región de la imagen se apoyó el modelo, disponible en el panel de detalle de la pestaña Validación.
- **Procesamiento en segundo plano** con `QThread` (`src/workers/`), sin bloquear la GUI.

Pendiente: calibración de `N`/`K`/`M` con videos reales de cámara trampa, una herramienta de evaluación contra ground truth etiquetado, y pulido de UI/UX. La detección automática de múltiples especies en un mismo frame sigue sin resolverse (hay marcado manual y herramientas de explicabilidad para apoyar la revisión humana, pero no segmentación automática).

## Cómo ejecutar

```bash
cd 04-app
python main.py
```

Requiere el entorno virtual de la raíz del repo (`../.venv`) con `PySide6`, `torch` (build CPU), `open_clip`, `opencv-python`, `scikit-learn`, `scipy`, `pandas`, `numpy` y `Pillow` instalados. Todavía no hay un `requirements.txt` propio de este módulo; las dependencias se instalan junto con las de `02-benchmarking`.

## Estructura de carpetas

```
04-app/
├── main.py                    # Punto de entrada
├── inference/
│   ├── pipeline.py            # Pipeline de inferencia, desacoplado de la GUI
│   ├── benchmark_latency.py   # Medición de latencia del pipeline completo
│   └── test_pipeline.py       # Pruebas del pipeline vía consola
├── src/
│   ├── gui/
│   │   ├── main_window.py     # Ventana principal, navbar, orquestación de pestañas
│   │   ├── styles.py          # Hoja de estilos Qt (paleta SAREKO)
│   │   └── tabs/               # analisis_tab.py, validacion_tab.py, evaluacion_tab.py
│   ├── workers/                # QThread: processing_worker.py, analysis_worker.py
│   └── data/
│       └── session.py          # Persistencia de sesión (último análisis, historial)
├── assets/                     # SAREKO.svg (logo), fondo.png
└── data/                       # config.json + artefactos locales (no versionados)
```

## Configuración (`data/config.json`)

Todos los parámetros del pipeline están externalizados, no hardcodeados:

| Clave | Valor por defecto | Descripción |
|---|---|---|
| `confidence_threshold` | `0.1866` | Umbral de confianza (distancia coseno al centroide, percentil 95). **Desactualizado**: el umbral vigente recalibrado sobre `query_dev` y validado sobre `query_test` en `03-threshold-optimization` es `0.1869` (diferencia de 0.0003); este archivo no se actualizó después del recalibrado. |
| `rejection_threshold` | `0.25` | Distancia por encima de la cual se rechaza la predicción del árbitro KNN |
| `knn_k` | `5` | Vecinos considerados por el árbitro KNN |
| `default_N` | `30` | Submuestreo temporal: 1 frame cada N |
| `default_K` | `10` | Tamaño de ventana de consenso temporal |
| `default_M` | `6` | Quórum mínimo de coincidencias dentro de la ventana K |
| `sliding_close_quorum_P` | `3` | Quórum para especies "cercanas" en el consenso deslizante |
| `debug_mode` | `false` | Habilita logging detallado a `data/sareko_debug.log` |

`N`, `K` y `M` son los parámetros que todavía están pendientes de calibración final con videos reales de cámara trampa.

## Catálogo de referencia

El pipeline pre-calcula y cachea los centroides por especie en `data/centroides_bioclip_v2.pkl` (no versionado, se regenera automáticamente a partir de `02-benchmarking/data/dataset_index.csv` y `02-benchmarking/data/features/bioclip_v2/`). En producción se usa el dataset completo (gallery + query) para maximizar la representatividad de los centroides, a diferencia del benchmark, que usa solo el split de gallery para mantener separación estricta entre calibración y evaluación.

## Datos no versionados

Por su naturaleza local o generada en tiempo de ejecución, `.gitignore` excluye de este módulo:

- `data/*.pkl`, `data/last_session.json`, `data/history.json`, `data/sareko_debug.log`
- `SAREKO_clips/`: clips de video recortados por la app al validar eventos
