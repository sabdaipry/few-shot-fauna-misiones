"""Genera el índice del dataset (dataset_index.csv) escaneando las imágenes,
integrando puntajes IVC y asignando splits galería/query por especie."""
import sys
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Obtenemos la ruta absoluta de ESTE archivo (scripts/01_generate_index.py)
current_script_path = Path(__file__).resolve()
# Obtenemos el directorio raíz del proyecto (el padre de 'scripts')
project_root = current_script_path.parent.parent

# Agregamos la raíz al sys.path para poder importar 'src'
sys.path.append(str(project_root))

# --- IMPORTACIÓN DEL LOGGER ---
try:
    from src.utils.logger import setup_logger
    from src.config import DATA_DIR
except ImportError:
    print("Error: No se encuentra el módulo 'src.utils.logger' o 'src.config'.")
    sys.exit(1)


# --- CONFIGURACIÓN ---
# Ajusta esta ruta a donde tengas tu carpeta 'images'
DATASET_ROOT = project_root / "data/fauna_seleccionada_bosque_atlantico/images"
OUTPUT_CSV = project_root / "data/dataset_index.csv"
IVC_CSV_PATH = project_root / "data/Indice_Valor_Conservacion_Misiones.csv"
MANUAL_FIXES_PATH = DATA_DIR / "manual_fixes.yaml"
SEED = 29  # Semilla para que el random sea siempre igual

logger = setup_logger("index_generator", log_dir=project_root / "logs")


def load_manual_fixes(yaml_path: Path) -> tuple[dict, dict]:
    """Carga aliases y hardcoded desde el YAML de correcciones manuales."""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        aliases = data.get("aliases", {}) or {}
        hardcoded = data.get("hardcoded", {}) or {}
        logger.info(f"-> Manual fixes cargados: {len(aliases)} aliases, {len(hardcoded)} hardcoded.")
        return aliases, hardcoded
    except FileNotFoundError:
        logger.error(f"Archivo de correcciones manuales no encontrado: {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error leyendo {yaml_path}: {e}")
        sys.exit(1)


ALIASES, HARDCODED = load_manual_fixes(MANUAL_FIXES_PATH)

def normalize_name(name):
    """
    Normaliza strings para comparaciones (minusculas, sin guiones bajos).
    """
    return str(name).replace("_", " ").strip().lower()

def load_ivc_metadata(csv_path) -> dict:
    """
    Carga el CSV de conservación y crea un diccionario de búsqueda rápida.
    Normaliza los nombres a minúsculas para evitar errores de tipeo.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Archivo de IVC no encontrado en: {csv_path}. Se usará IVC=0.")
        return {}

    try:
        df = pd.read_csv(csv_path)
        lookup = {}
        # Asumimos columnas: 'Especie', 'IVC', 'Estado (Scons)'
        for _, row in df.iterrows():
            # Limpieza: quitar espacios extra y pasar a minúsculas
            sci_name = normalize_name(row['Especie'])
            lookup[sci_name] = {
                'ivc_score': row['IVC']
            }
        logger.info(f"-> Metadatos IVC cargados: {len(lookup)} especies en la base de datos.")
        return lookup
    except Exception as e:
        logger.error(f"Error leyendo el CSV de IVC: {e}")
        return {}

def calculate_ivc_category(score):
    """
    Define la categoría según las reglas del proyecto basándose en el puntaje IVC.
    """
    if score == 0:
        return "Exótica/Invasora/Desconocida"
    elif score <= 2:
        return "Bajo"
    elif score < 4: # Mayor a 2 y menor a 4 (es decir, 3)
        return "Medio"
    elif score < 5: # Mayor o igual a 4 y menor a 5 (es decir, 4)
        return "Alto"
    else: # Mayor o igual a 5
        return "Crítico"

def generate_index():
    """Escanea el directorio de imágenes, integra IVC y escribe dataset_index.csv con splits."""
    # 1. Cargar Metadatos
    ivc_lookup = load_ivc_metadata(IVC_CSV_PATH)
    
    # Recorremos el directorio: images/Familia/Genero/Especie
    root_path = Path(DATASET_ROOT)
    
    # Validamos que exista
    if not root_path.exists():
        logger.error(f"No se encontró la carpeta del dataset en: {DATASET_ROOT}")
        return

    logger.info(f"-> Escaneando directorio {DATASET_ROOT} e integrando datos de {IVC_CSV_PATH}...")

    # Obtenemos todas las carpetas de especies (asumiendo profundidad 3)
    # Estructura esperada: root / familia / genero / especie / imagen.jpg
    
    # Estrategia: Listar todas las imágenes primero
    # Listar imágenes
    all_images = []
    # Extensiones válidas (agregué .JPG por las dudas)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'} 
    
    for path in root_path.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_exts:
            all_images.append(path)

    if not all_images:
        logger.warning(f"No se encontraron imágenes. Revisa la ruta {DATASET_ROOT}.")
        return

    logger.info(f"-> Imágenes encontradas: {len(all_images)}")

    # Agrupar por especie (nombre de la carpeta padre)
    species_groups = {}
    for img_path in all_images:
        try:
            # Asumiendo estructura: images/Familia/Genero/Especie/foto.jpg
            parts = img_path.parts
            # Ajuste dinámico: buscamos 'images' y tomamos las siguientes carpetas
            # Esto hace el script más robusto si 'images' no está en la raiz relativa directa
            if 'images' in parts:
                idx = parts.index('images')
                # Verificamos profundidad suficiente
                if len(parts) > idx + 3:
                    family = parts[idx+1]
                    genus = parts[idx+2]
                    species_folder = parts[idx+3] # Puede ser "onca" o "Panthera onca"
                    
                    # --- CORRECCIÓN AQUÍ ---
                    # 1. Limpiamos el nombre de la carpeta (ej. "Panthera_onca" -> "panthera onca")
                    clean_folder_name = normalize_name(species_folder)
                    clean_genus = normalize_name(genus)
                    
                    # 2. Generamos candidatos
                    # Candidato A: La carpeta ya es el nombre completo (ej. "panthera onca")
                    candidate_name_a = clean_folder_name
                    
                    # Candidato B: La carpeta es solo la especie, hay que unir (ej. "panthera" + " " + "onca")
                    # Nota: evitamos duplicar si el nombre de carpeta ya contenía el género
                    if clean_genus in clean_folder_name:
                         candidate_name_b = clean_folder_name
                    else:
                         candidate_name_b = f"{clean_genus} {clean_folder_name}"
                    
                    # --- LÓGICA DE MATCHING CON MANUAL FIXES ---
                    ivc_data = {'ivc_score': 0}
                    final_species_name = species_folder.replace("_", " ") # Default display name

                    # A) Revisar correcciones manuales primero
                    match_found = False

                    # Chequeamos ambos candidatos contra aliases y hardcoded
                    for cand in [candidate_name_a, candidate_name_b]:
                        if cand in HARDCODED:
                            ivc_data = HARDCODED[cand]
                            match_found = True
                        elif cand in ALIASES:
                            alias_target = ALIASES[cand]
                            if alias_target in ivc_lookup:
                                ivc_data = ivc_lookup[alias_target]
                                match_found = True
                            else:
                                logger.warning(f"Mapeo manual fallido: {cand} -> {alias_target} (No está en CSV)")
                        if match_found:
                            break
                    
                    # B) Si no hubo fix manual, buscar en CSV normal
                    if not match_found:
                        if candidate_name_a in ivc_lookup:
                            ivc_data = ivc_lookup[candidate_name_a]
                        elif candidate_name_b in ivc_lookup:
                            ivc_data = ivc_lookup[candidate_name_b]
                            final_species_name = f"{genus} {species_folder}".replace("_", " ")
                    

                    current_score = ivc_data.get('ivc_score', 0)
                    
                    # 1. Si el Manual Fix ya traía una categoría explicita (ej: "Doméstica"), úsala.
                    if 'ivc_category' in ivc_data:
                        final_category = ivc_data['ivc_category']
                    else:
                        # 2. Si no, calcúlala basada en el score (Reglas del Proyecto)
                        final_category = calculate_ivc_category(current_score)

                    if final_species_name not in species_groups:
                        species_groups[final_species_name] = []
                    
                    species_groups[final_species_name].append({
                        'filepath': str(img_path),
                        'family': family,
                        'genus': genus,
                        'species': final_species_name,
                        'ivc_score': ivc_data['ivc_score'],
                        'ivc_category': final_category
                    })
                else:
                    logger.debug(f"Saltando archivo por estructura irreconocible: {img_path}")
        except Exception as e:
            logger.warning(f"Error procesando path {img_path}: {e}")

    logger.info(f"-> Especies encontradas: {len(species_groups)}")

    # Procesar cada especie y aplicar el Split
    logger.info("Generando splits de Galería y Query...")
    data_rows = []
    np.random.seed(SEED)

    stats_ivc_found = 0

    # Estadísticas para el reporte
    category_counts = {}
    
    for species_name, items in species_groups.items():
        n_samples = len(items)

        # Contamos si tiene score asignado (mayor a 0 o categoría conocida)
        if items[0]['ivc_category'] != 'Desconocido':
            stats_ivc_found += 1
        else:
            logger.warning(f"Especie sin IVC asignado: {species_name}")
            

        # Conteo de categorías
        cat = items[0]['ivc_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        gallery_indices = []
        
        # --- LÓGICA DE SPLIT (Tu Criterio) ---
        if n_samples == 2:
            gallery_indices = indices[:1] # 1 Galeria / 1 Query
        elif n_samples == 3:
            gallery_indices = indices[:2] # 2 Galeria / 1 Query
        else:
            # Regla del 20% (mínimo 2 si es posible)
            n_gallery = max(1, int(n_samples * 0.20))
            if n_samples >= 5 and n_gallery < 2:
                n_gallery = 2
            gallery_indices = indices[:n_gallery]

        # Asignar a la lista maestra
        for i in range(n_samples):
            role = 'gallery' if i in gallery_indices else 'query'
            item = items[i]
            
            data_rows.append({
                'filepath': item['filepath'],
                'species': item['species'],
                'genus': item['genus'],
                'family': item['family'],
                'split': role,
                'ivc_score': item['ivc_score'],
                'ivc_category': item['ivc_category']
            })

    # Guardar
    df = pd.DataFrame(data_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, lineterminator="\n")
    
    logger.info(f"-> Índice generado exitosamente en: {OUTPUT_CSV}")
    
    # Reporte final en log
    train_count = len(df[df['split']=='gallery'])
    test_count = len(df[df['split']=='query'])
    logger.info("="*50)
    logger.info(" RESUMEN DEL DATASET")
    logger.info("="*50)
    logger.info(f"   Total muestras: {len(df)}")
    logger.info(f"   Galery (Referencias): {train_count}")
    logger.info(f"   Query (Evaluación):    {test_count}")
    logger.info(f"   Especies con IVC: {stats_ivc_found}/{len(species_groups)}")

    if stats_ivc_found < len(species_groups):
        logger.info(f"  Hay {len(species_groups) - stats_ivc_found} especies sin datos de conservación. Revisa los nombres en el CSV.")
    
    logger.info("   Distribución por Categoría de Conservación:")
    
    for cat, count in category_counts.items():
        logger.info(f"    - {cat}: {count} especies")
        
if __name__ == "__main__":
    try:
        generate_index()
    except Exception as e:
        logger.exception("Ocurrió un error fatal durante la ejecución")