import timm
print("Buscando modelos de iNaturalist disponibles...")
# Buscamos cualquier cosa que contenga 'inat'
nombres = timm.list_models('*inat*')

print(f"Encontrados {len(nombres)} modelos.")
# Filtramos solo los resnet50 para no llenar la pantalla
resnet_inat = [m for m in nombres if 'resnet50' in m]

if resnet_inat:
    print("✅ ¡Estos son los nombres válidos en TU versión!:")
    for m in resnet_inat:
        print(f"   '{m}'")
else:
    print("⚠️ No encontré Resnet50, mostrando todos los iNat:")
    print(nombres[:10])