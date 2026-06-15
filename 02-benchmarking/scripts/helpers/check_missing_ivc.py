"""Inspección rápida de especies en dataset_index.csv que quedaron sin IVC asignado."""
import pandas as pd

# Cargar el índice que acabamos de generar
df = pd.read_csv("data/dataset_index.csv")

# Filtrar las especies que quedaron con ivc_score 0
missing_ivc = df[df['ivc_score'] == 0]['species'].unique()

print(f" Hay {len(missing_ivc)} especies sin datos de conservación:")
for sp in sorted(missing_ivc):
    print(f" - {sp}")