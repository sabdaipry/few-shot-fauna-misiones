# Wildlife Image Dataset Curator

Una herramienta de escritorio para auditar y curar visualmente datasets de imágenes de vida silvestre usando embeddings de DINOv2 y proyecciones UMAP.

> 📦 Este módulo vive en un repositorio separado: [sabdaipry/wildlife-image-dataset-curator](https://github.com/sabdaipry/wildlife-image-dataset-curator). Este archivo es una copia de su README para referencia rápida dentro de `few-shot-fauna-misiones`; el código fuente y las actualizaciones viven en el repo enlazado.

## Qué hace

El curador embebe cada imagen del dataset con un modelo DINOv2 ViT-B/14 preentrenado, reduce los vectores de alta dimensionalidad a 2D con UMAP, y renderiza un gráfico de dispersión interactivo. Podés hacer clic en puntos individuales o trazar un lazo para seleccionar grupos de imágenes, previsualizarlas en la aplicación, y enviar imágenes de baja calidad o mal etiquetadas a una carpeta de descarte — todo registrado en un CSV para que la operación sea reversible.

**Características**

- Gráfico de dispersión UMAP interactivo con colores por familia taxonómica
- Selección de puntos con un clic, panel de previsualización y metadatos
- Herramienta lazo para selección por lotes de clusters de imágenes
- Descarte con un clic: mueve los archivos a `data/deleted/` y los marca en el CSV
- Restauración completa del dataset (deshacer todos los descartes)
- Tabla de estadísticas por especie y por familia
- Alternancia de tema oscuro / claro

## Requisitos

- Python 3.10+
- GPU NVIDIA con CUDA fuertemente recomendada para la generación de embeddings; existe alternativa en CPU pero es lenta

**Dependencias de Python**

```
PySide6
torch
torchvision
Pillow
numpy
pandas
umap-learn
matplotlib
tqdm
```

Dependencia de desarrollo: `pytest`

## Instalación

```bash
git clone https://github.com/sabdaipry/wildlife-image-dataset-curator.git
cd wildlife-image-dataset-curator
pip install -r requirements.txt
```

## Uso

### Paso 1 — Generar embeddings

```bash
python scripts/generate_embeddings.py --dataset <ruta/a/imágenes>
```

| Argumento | Requerido | Por defecto | Descripción |
|---|---|---|---|
| `--dataset DIR` | sí | — | Directorio raíz de las imágenes a procesar |
| `--output CSV` | no | valor de `DATA_FILE` en `config.py` | Ruta del CSV de salida |
| `--batch-size N` | no | `8` | Tamaño del lote para embeddings |

Ejemplo:

```bash
python scripts/generate_embeddings.py \
    --dataset data/raw/fauna_seleccionada_bosque_atlantico/images \
    --output output/datos_clusters_vitb14.csv \
    --batch-size 16
```

El script recorre el directorio de forma recursiva, extrae los embeddings DINOv2, ejecuta UMAP y escribe un CSV con las columnas `x`, `y`, `species_id`, `scientific_name`, `common_name`, `family`, `genus`, `user`, `filename`, `absolute_path`.

### Paso 2 — Ejecutar la aplicación

```bash
python main.py
```

La aplicación lee la ruta del CSV desde `config.py` (`DATA_FILE`) y se abre maximizada.

## Formato del dataset

Las imágenes deben estar organizadas bajo una jerarquía taxonómica de tres niveles:

```
<root>/
└── <Family>/
    └── <Genus>/
        └── <Species>/
            ├── image_001.jpg
            ├── image_001.json   ← sidecar
            └── ...
```

Cada JSON sidecar debe incluir como mínimo:

```json
{
  "species_id": 41997,
  "species": "Leopardus pardalis",
  "common_name": "Ocelot",
  "user": {
    "login": "photographer_username"
  }
}
```

Los campos faltantes se completan con valores inferidos desde la ruta de carpetas. Se espera que el campo `user` sea un objeto; solo se usa `user.login`.

Para descargar imágenes de iNaturalist junto con su JSON de metadatos se recomienda usar [fscl-vision-data](https://github.com/AxelSkrauba/fscl-vision-data), que automatiza la descarga y genera la estructura de carpetas y los sidecars JSON compatibles con este proyecto.

## Configuración

Editá `config.py` para cambiar el comportamiento en tiempo de ejecución:

| Clave | Por defecto | Descripción |
|---|---|---|
| `DATA_FILE` | `output/datos_clusters_vitb14.csv` | CSV generado por el script de embeddings |
| `TRASH_FOLDER` | `data/deleted` | Carpeta de destino para las imágenes descartadas |
| `EMBEDDING_MODEL_NAME` | `"DINOv2 ViT-B/14"` | Nombre de visualización en el título del gráfico UMAP |
| `THEMES` | `{"dark": ..., "light": ...}` | Hojas de estilo Qt; editá para personalizar los colores |

## Ejecutar tests

```bash
pytest tests/
```

La suite de tests cubre `DataManager`: carga del CSV, resumen global, filtrado por lazo y cálculo de rutas de destino (10 tests).

## Roadmap

- **i18n** — Las cadenas de la UI están actualmente en español; extraerlas para una internacionalización adecuada
- **Tests adicionales** — Cobertura para las señales de `UMAPWidget` y el panel de estadísticas
- **Selector de modelo en la GUI** — Permitir cambiar entre variantes de DINOv2 (ViT-S/14, ViT-L/14) sin editar la configuración
