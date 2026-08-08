# 04-app: SAREKO, aplicación de escritorio

## Qué es

**SAREKO** es la aplicación de escritorio del proyecto: recibe videos o imágenes de cámaras trampa y devuelve un registro biológico por especie, sin depender de GPU ni de conexión a internet durante el procesamiento. Está pensada para biólogos, ecólogos y guardaparques sin formación técnica en IA.

Se reescribe desde cero (arquitectura limpia, PySide6) tomando como referencia funcional un MVP monolítico anterior, cuyo código no se reutiliza.

## Estado actual

En desarrollo activo, con la mayor parte del pipeline y la GUI ya funcionales. Implementado:

- **Pipeline de inferencia completo** (`inference/pipeline.py`): filtro de movimiento (MOG2) como pre-filtro antes de invocar el backbone, extracción de embeddings en batch con BioCLIP v2 (`open_clip`), y clasificación en tres niveles por distancia coseno al centroide: **alta confianza** (etiqueta directa), **baja confianza** (zona gris, resuelta por árbitro KNN) y **rechazado** (distancia por encima de `rejection_threshold`, excluido del consenso).
- **Filtro de movimiento configurable en 4 modos**: ninguno, alto contraste (diurno), bajo contraste (nocturno/infrarrojo) y adaptativo (elige automáticamente entre alto y bajo contraste según la iluminación del video).
- **Dos algoritmos de consenso temporal**, seleccionables por el usuario: ventana estática (ventanas independientes de K frames) y ventana deslizante (`SlidingWindowConsensus`, detecta inicio/fin de evento frame a frame, más preciso con fauna intermitente).
- **Tres modos de análisis con N/K/M preconfigurados**: Básico (N=60, 1 frame cada 60), Estándar (N=30) y Profundo (N=10), cada uno con su propio K y M ajustados. No es una calibración empírica formal (no hay todavía una batería de videos con ground truth que mida accuracy por combinación de parámetros), pero sí presets pensados y probados sobre clips reales de cámara trampa durante el desarrollo.
- **GUI con las tres pestañas** especificadas (Análisis / Validación / Evaluación), ver `src/gui/tabs/`. La pestaña Evaluación incluye: resumen global acumulado, tabla de latencia histórica por archivo, evaluación de errores en 5 categorías (Correctas / Top-5 / Conocida fuera del top-5 / Desconocida / Vacío-Ruido) con tabla expandible, card de especies fuera de catálogo ingresadas manualmente, card de clips marcados como multi-especie, y 4 gráficos.
- **Persistencia de sesión con restauración automática** (`src/data/session.py`): la app recarga sola el último análisis al abrirse, sin necesidad de una acción manual de "cargar reporte", y mantiene un historial acumulativo de corridas entre sesiones (`data/history.json`).
- **Explicabilidad visual** (Attention Rollout y GradCAM sobre BioCLIP v2) para inspeccionar en qué región de la imagen se apoyó el modelo, con caché por evento y exportación de la imagen del mapa, disponible en el panel de detalle de la pestaña Validación.
- **Marcado manual de eventos multi-especie**, con autocompletado de especies adicionales sobre el catálogo de 91 especies.
- **Procesamiento en segundo plano** con `QThread` (`src/workers/`), sin bloquear la GUI.

Pendiente: calibrar los umbrales del filtro de movimiento MOG2 (`motion_filter_*` en `data/config.json`), que hoy son valores estándar de partida, no ajustados contra video real de cámara trampa; una herramienta de evaluación contra ground truth etiquetado para medir accuracy de punta a punta (y con eso, calibrar `N`/`K`/`M` con datos en vez de presets razonados); y pulido de UI/UX. La detección automática de múltiples especies en un mismo frame sigue sin resolverse (hay marcado manual y herramientas de explicabilidad para apoyar la revisión humana, pero no segmentación automática).

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
| `confidence_threshold` | `0.1869` | Umbral de alta confianza (distancia coseno al centroide, percentil 95), calibrado sobre `query_dev` y validado sobre `query_test` en `03-threshold-optimization`. Por debajo de este valor, etiqueta directa. |
| `rejection_threshold` | `0.25` | Techo de la zona gris: entre `confidence_threshold` y este valor interviene el árbitro KNN; por encima, el frame se rechaza y se excluye del consenso. |
| `knn_k` | `5` | Vecinos considerados por el árbitro KNN |
| `default_N` | `30` | Submuestreo temporal por defecto (modo Estándar): 1 frame cada N |
| `default_K` | `10` | Ventana de consenso temporal por defecto (modo Estándar) |
| `default_M` | `6` | Quórum mínimo dentro de la ventana K por defecto (modo Estándar) |
| `sliding_close_quorum_P` | `3` | Frames consecutivos por debajo del umbral para cerrar un evento en consenso por ventana deslizante |
| `motion_filter_high_contrast_area` | `500` | Área mínima de contorno en movimiento para el modo "alto contraste" del `MotionFilter` |
| `motion_filter_low_contrast_area` | `100` | Área mínima de contorno en movimiento para el modo "bajo contraste" del `MotionFilter` |
| `motion_filter_adaptive_luminance_threshold` | `60` | Umbral de luminancia que usa el modo "adaptativo" para elegir entre alto y bajo contraste |
| `debug_mode` | `false` | Habilita logging detallado a `data/sareko_debug.log` |

`default_N`/`default_K`/`default_M` son la base del modo "Estándar"; la GUI (pestaña Análisis) permite elegir entre los modos Básico, Estándar y Profundo, cada uno con su propio N/K/M (ver `src/gui/tabs/analisis_tab.py`, `_MODES`), y entre consenso por ventana estática o deslizante (`_CONSENSUS_MODES`).

Los tres `motion_filter_*` son valores estándar de partida, todavía sin calibrar contra video real de cámara trampa (ver "Pendiente" arriba).

## Catálogo de referencia

El pipeline pre-calcula y cachea los centroides por especie en `data/centroides_bioclip_v2.pkl` (no versionado, se regenera automáticamente a partir de `02-benchmarking/data/dataset_index.csv` y `02-benchmarking/data/features/bioclip_v2/`). En producción se usa el dataset completo (gallery + query) para maximizar la representatividad de los centroides, a diferencia del benchmark, que usa solo el split de gallery para mantener separación estricta entre calibración y evaluación.

## Datos no versionados

Por su naturaleza local o generada en tiempo de ejecución, `.gitignore` excluye de este módulo:

- `data/*.pkl`, `data/last_session.json`, `data/history.json`, `data/sareko_debug.log`
- `SAREKO_clips/`: clips de video recortados por la app al validar eventos
