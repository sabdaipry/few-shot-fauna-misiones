@echo off
echo ==================================================
echo   CONFIGURANDO ENTORNO VIRTUAL...
echo ==================================================

:: Activamos el entorno virtual
call "C:\Users\Win 10\Desktop\CEIA\TF\TF-Private-Sketches\.venv\Scripts\activate.bat"

:: Verificamos que se activo (deberia mostrar la ruta dentro de venv)
where python
echo.

echo ==================================================
echo   FASE 1: MODELOS LIGEROS (Prioridad Alta)
echo   Tiempo estimado: 3-4 horas
echo ==================================================

echo [1/10] ResNet50 (Baseline)...
python scripts/02_extract_features.py --model resnet50

echo ==================================================

echo [2/10] DINOv2 Small...
python scripts/02_extract_features.py --model dinov2_small

echo ==================================================

echo [3/10] ConvNeXt Tiny...
python scripts/02_extract_features.py --model convnext_tiny

echo ==================================================

echo [4/10] SigLIP Base...
python scripts/02_extract_features.py --model siglip_base

echo ==================================================
echo   FASE 2: MODELOS MEDIANOS (Prioridad Media)
echo   Tiempo estimado: 6-8 horas
echo ==================================================

echo [5/10] BioCLIP...
python scripts/02_extract_features.py --model bioclip

echo ==================================================

echo [6/10] DINOv2 Base...
python scripts/02_extract_features.py --model dinov2_base

echo ==================================================

echo [7/10] DINOv3 Small...
python scripts/02_extract_features.py --model dinov3_small

echo ==================================================

echo [8/10] ConvNeXt Base...
python scripts/02_extract_features.py --model convnext_base

echo ==================================================
echo   FASE 3: LOS GIGANTES (Riesgo de RAM)
echo   Tiempo estimado: 10-12 horas 
echo ==================================================

echo [9/10] DINOv3 Base...
python scripts/02_extract_features.py --model dinov3_base

echo ==================================================

echo [10/10] SigLIP SO400M...
python scripts/02_extract_features.py --model siglip_so400m

echo ==================================================
echo FIN DEL PROCESO.
pause