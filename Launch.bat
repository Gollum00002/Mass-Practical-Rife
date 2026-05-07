@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: Launch.bat — RIFE Video Frame Interpolation
:: Mirrors Launch.sh: straightforward venv setup, no fancy
:: scanning. Works natively on Windows and under Wine on Linux.
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "PYTHON_SCRIPT=%SCRIPT_DIR%\inference-windows.py"
set "VENV_DIR=%SCRIPT_DIR%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "REQUIREMENTS=%SCRIPT_DIR%\requirements.txt"

:: ---- Verify inference script exists ----
if not exist "%PYTHON_SCRIPT%" (
    echo Error: inference-windows.py not found at:
    echo %PYTHON_SCRIPT%
    echo Make sure Launch.bat is in the same folder as the Python script.
    pause
    exit /b 1
)

:: ================================================================
:: Python Detection
:: ================================================================
set "PYTHON_CMD="

where py >nul 2>&1
if %errorlevel% == 0 set "PYTHON_CMD=py"

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if %errorlevel% == 0 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "C:\Python314\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
    ) do (
        if not defined PYTHON_CMD if exist "%%P" (
            set "PYTHON_CMD=%%P"
        )
    )
)

if not defined PYTHON_CMD (
    echo.
    echo Error: Python not found.
    pause
    exit /b 1
)

echo Found Python: %PYTHON_CMD%

:: ================================================================
:: Virtual Environment Setup
:: ================================================================
if not exist "%VENV_PYTHON%" (
    echo.
    echo No virtual environment found. Creating one at %VENV_DIR%...
    "%PYTHON_CMD%" -m venv "%VENV_DIR%"
    "%VENV_PIP%" install --upgrade pip
    "%VENV_PIP%" install -r "%REQUIREMENTS%"
)
set "PYTHON_CMD=%VENV_PYTHON%"

:: ================================================================
:: Default Parameters (Hardware Check Delegated to Python)
:: ================================================================
set "SCALE=1"
set "TARGET_FPS=120"
set "FP16=--fp16"

set "INPUT_DIR_ARG="
set "OUTPUT_DIR=fpsConv"
set "MODEL_DIR_ARG="
set "PNG_ARG="
set "EXT_ARG="
set "DISABLE_AUTO_SCALE_ARG="

:: ---- Parse command-line arguments ----
:parse_args
if "%~1"=="" goto end_parse

if /i "%~1"=="--scale" (
    set "SCALE=%~2"
    shift & shift & goto parse_args
)
if /i "%~1"=="--target-fps" (
    set "TARGET_FPS=%~2"
    shift & shift & goto parse_args
)
if /i "%~1"=="--no-fp16" (
    set "FP16="
    shift & goto parse_args
)
if /i "%~1"=="--input-dir" (
    set "INPUT_DIR_ARG=--input-dir "%~f2""
    shift & shift & goto parse_args
)
if /i "%~1"=="--output" (
    set "OUTPUT_DIR=%~2"
    shift & shift & goto parse_args
)
if /i "%~1"=="--model" (
    set "MODEL_DIR_ARG=--model "%~f2""
    shift & shift & goto parse_args
)
if /i "%~1"=="--png" (
    set "PNG_ARG=--png"
    shift & goto parse_args
)
if /i "%~1"=="--ext" (
    set "EXT_ARG=--ext %~2"
    shift & shift & goto parse_args
)
if /i "%~1"=="--disable-auto-scale" (
    set "DISABLE_AUTO_SCALE_ARG=--disable-auto-scale"
    shift & goto parse_args
)
if /i "%~1"=="--help" (
    goto show_help
)
:end_parse

:: ---- Summary ----
echo.
echo Running RIFE with the following settings:
echo   Project Directory : %SCRIPT_DIR%
echo   Python            : %PYTHON_CMD%
echo   Scale             : %SCALE%
echo   Target FPS        : %TARGET_FPS%
if defined FP16 (echo   FP16              : Enabled ^(Auto-fallback on CPU^)) else (echo   FP16              : Disabled)
echo   Output Subfolder  : %OUTPUT_DIR%
if defined MODEL_DIR_ARG (echo   Model             : Custom path provided) else (echo   Model             : Default)
echo.

:: ---- Run ----
cd /d "%SCRIPT_DIR%"

:: Tell OpenMP to skip the unsupported Wine NUMA API calls to prevent crashes
set KMP_AFFINITY=disabled

"%PYTHON_CMD%" "%PYTHON_SCRIPT%" ^
    --scale %SCALE% ^
    --target-fps %TARGET_FPS% ^
    %FP16% ^
    %INPUT_DIR_ARG% ^
    --output "%OUTPUT_DIR%" ^
    %MODEL_DIR_ARG% ^
    %PNG_ARG% ^
    %EXT_ARG% ^
    %DISABLE_AUTO_SCALE_ARG%

echo.
echo Done. Press any key to close this window...
pause >nul
endlocal