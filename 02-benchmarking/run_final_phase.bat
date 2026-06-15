@echo off
echo ==================================================
echo   FASE FINAL: COMPLETANDO EL BENCHMARK
echo   Orden: Ya Corridos -> Nuevos Rapidos -> Nuevos Lentos
echo ==================================================

echo ==================================================
echo   CONFIGURANDO ENTORNO VIRTUAL...
echo ==================================================

:: Activamos entorno
call "C:\Users\Win 10\Desktop\CEIA\TF\TF-Private-Sketches\.venv\Scripts\activate.bat"

:: Verificamos que se activo (deberia mostrar la ruta dentro de venv)
where python
echo.

echo ==================================================
echo   GRUPO 1: YA CORRIDOS (Verificacion rapida)
echo ==================================================

echo [1/19] ResNet50 (Checkeo)...
python scripts/02_extract_features.py --model resnet50

echo ==================================================

echo [2/19] ConvNeXt Tiny (Checkeo)...
python scripts/02_extract_features.py --model convnext_tiny

echo ==================================================

echo [3/19] DINOv2 Small (Checkeo)...
python scripts/02_extract_features.py --model dinov2_small

echo ==================================================

echo [4/19] DINOv3 Small (Checkeo)...
python scripts/02_extract_features.py --model dinov3_small

echo ==================================================

echo [5/19] SigLIP Base (Checkeo)...
python scripts/02_extract_features.py --model siglip_base

echo ==================================================

echo [6/19] BioCLIP v1 (Checkeo)...
python scripts/02_extract_features.py --model bioclip_v1

echo ==================================================

echo [7/19] ConvNeXt Base (Checkeo)...
python scripts/02_extract_features.py --model convnext_base

echo ==================================================

echo [8/19] DINOv2 Base (Checkeo)...
python scripts/02_extract_features.py --model dinov2_base

echo ==================================================

echo [9/19] DINOv3 Base (Checkeo)...
python scripts/02_extract_features.py --model dinov3_base

echo ==================================================

echo [10/19] SigLIP SO400M (Checkeo)...
python scripts/02_extract_features.py --model siglip_so400m


echo ==================================================
echo   GRUPO 2: NUEVOS - RAPIDOS Y MEDIANOS
echo ==================================================

echo [11/19] CLIP Base (Nuevo - Rapido)...
python scripts/02_extract_features.py --model clip_base

echo ==================================================

echo [12/19] SigLIP 2 Base (Nuevo - Rapido con NaFlex)...
python scripts/02_extract_features.py --model siglip2_base

echo ==================================================

echo [13/19] DINOv2 Small GAP (Nuevo - Rapido Denso)...
python scripts/02_extract_features.py --model dinov2_small_gap

echo ==================================================

echo [14/19] DINOv3 Small GAP (Nuevo - Rapido Denso)...
python scripts/02_extract_features.py --model dinov3_small_gap

echo ==================================================

echo [15/19] DINOv2 Base GAP (Nuevo - Medio Denso)...
python scripts/02_extract_features.py --model dinov2_base_gap

echo ==================================================

echo [16/19] DINOv3 Base GAP (Nuevo - Medio Denso)...
python scripts/02_extract_features.py --model dinov3_base_gap

echo ==================================================
echo   GRUPO 3: NUEVOS - PESADOS 
echo ==================================================

echo [17/19] CLIP Large (Nuevo - Pesado)...
python scripts/02_extract_features.py --model clip_large

echo ==================================================

echo [18/19] BioCLIP v2 (Nuevo - Pesado SOTA)...
python scripts/02_extract_features.py --model bioclip_v2

echo ==================================================

echo [19/19] SigLIP 2 SO400M (Nuevo - Muy Pesado)...
python scripts/02_extract_features.py --model siglip2_so400m

echo ==================================================
echo   ¡TODO LISTO! 
echo ==================================================
pause