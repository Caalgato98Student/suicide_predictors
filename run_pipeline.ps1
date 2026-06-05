#Requires -Version 5.1
<#
.SYNOPSIS
    Ejecuta el pipeline de analisis descriptivo y Machine Learning ENSANUT 2024.

.DESCRIPTION
    Verifica prerequisitos, activa el entorno virtual, instala dependencias,
    ejecuta los scripts del flujo KDD en Python en orden secuencial
    y ofrece lanzar el dashboard interactivo al terminar.

.PARAMETER SoloDashboard
    Omite el pipeline de modelado y lanza directamente el dashboard.

.EXAMPLE
    .\run_pipeline.ps1
    .\run_pipeline.ps1 -SoloDashboard
#>

param(
    [switch]$SoloDashboard
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -- Helpers ------------------------------------------------------------------

function Write-Step  { param($msg) Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Fail  { param($msg) Write-Host "    ERR $msg" -ForegroundColor Red; exit 1 }
function Write-Warn  { param($msg) Write-Host "    !   $msg" -ForegroundColor Yellow }

# -- Verificar directorio de trabajo ------------------------------------------

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
Write-Step "Directorio de trabajo: $ProjectRoot"

if (-not (Test-Path "data/adolescentes_ensanut2024_w.csv")) {
    Write-Fail "No se encontro 'data/adolescentes_ensanut2024_w.csv'. Verifica que estas en la raiz del proyecto."
}

# -- Verificar prerequisitos --------------------------------------------------

Write-Step "Verificando prerequisitos"

try {
    $pyVersion = & python --version 2>&1
    Write-Ok "Python: $pyVersion"
} catch {
    Write-Fail "Python no encontrado. Instalalo desde https://python.org"
}

# -- Entorno virtual Python ----------------------------------------------------

Write-Step "Configurando entorno virtual Python"

if (-not (Test-Path ".venv/Scripts/Activate.ps1")) {
    Write-Warn "Entorno virtual no encontrado. Creandolo..."
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Fail "No se pudo crear el entorno virtual." }
    Write-Ok "Entorno virtual creado en .venv"
} else {
    Write-Ok "Entorno virtual encontrado"
}

# Activar
. .venv/Scripts/Activate.ps1

# Instalar/verificar dependencias
Write-Warn "Verificando dependencias Python (esto puede tomar un momento la primera vez)..."
& pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Error al instalar dependencias de requirements.txt" }

# Asegurar que 'shap' este instalado
& pip install shap --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "No se pudo instalar 'shap'." }

Write-Ok "Dependencias Python instaladas correctamente"

# -- Pipeline de analisis (descriptivo & ML) ------------------------------------

if (-not $SoloDashboard) {

    # Paso 1: Preparacion descriptiva
    Write-Step "Paso 1/5 - Preparacion descriptiva"
    & python src/01_an_descriptivo.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "01_an_descriptivo.py fallo." }
    Write-Ok "Dataset descriptivo generado: adolescentes_limpio.csv"

    # Paso 2: Preprocesamiento ML
    Write-Step "Paso 2/5 - Preprocesamiento ML"
    & python src/02_preprocesamiento.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "02_preprocesamiento.py fallo." }
    Write-Ok "Dataset ML generado: adolescentes_ml.csv"

    # Paso 3: Analisis Exploratorio
    Write-Step "Paso 3/5 - Analisis exploratorio"
    & python src/03_exploracion.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "03_exploracion.py fallo." }
    Write-Ok "Figuras exploratorias guardadas en figuras\exploracion"

    # Paso 4: Entrenamiento y Clasificacion
    Write-Step "Paso 4/5 - Entrenamiento de clasificadores"
    & python src/04_clasificacion.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "04_clasificacion.py fallo." }
    Write-Ok "Modelos entrenados guardados en output\clasificacion\"

    # Paso 5: Interpretabilidad con SHAP
    Write-Step "Paso 5/5 - Analisis de interpretabilidad SHAP"
    & python src/05_shap.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "05_shap.py fallo." }
    Write-Ok "Valores SHAP y figuras guardados en output\shap\ y figuras\shap\"

    Write-Host "`n============================================" -ForegroundColor Green
    Write-Host "  Pipeline completado exitosamente" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Archivos clave generados:"
    Write-Host "    data/adolescentes_limpio.csv              (Datos limpios etiquetados)"
    Write-Host "    data/adolescentes_ml.csv                  (Matriz codificada para ML)"
    Write-Host "    figuras/preprocesamiento/14_*.png         (Distribucion del target)"
    Write-Host "    figuras/clasificacion/16_matrices_*.png   (Matrices de confusion)"
    Write-Host "    figuras/clasificacion/17_comparacion_*.png(Comparacion MCC)"
    Write-Host "    figuras/shap/18_shap_importancia_*.png    (Importancia global SHAP)"
    Write-Host "    figuras/shap/19_shap_beeswarm.png         (Beeswarm SHAP)"
    Write-Host "    output/clasificacion/mejor_modelo.joblib  (Mejor modelo entrenado)"
    Write-Host "    output/shap/shap_importancia.csv          (Tabla de importancia SHAP)"
    Write-Host ""
}

# -- Lanzar dashboard ----------------------------------------------------------

Write-Step "Dashboard interactivo"

if (-not (Test-Path "data/adolescentes_ml.csv")) {
    Write-Warn "No se encontraron los archivos de datos. Ejecuta el pipeline completo primero."
}

$respuesta = Read-Host "  Lanzar el dashboard en el navegador? [s/N]"
if ($respuesta -match '^[sS]$') {
    Write-Ok "Iniciando dashboard en http://localhost:8501 ..."
    & streamlit run dashboard/app.py
} else {
    Write-Host ""
    Write-Host "  Para lanzar el dashboard mas tarde ejecuta:" -ForegroundColor Gray
    Write-Host "    streamlit run dashboard/app.py" -ForegroundColor Gray
    Write-Host ""
}
