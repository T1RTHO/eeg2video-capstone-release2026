@echo off
setlocal
pushd "%~dp0"

if not defined EEG2VIDEO_ASSETS_DIR (
    echo ERROR: Set EEG2VIDEO_ASSETS_DIR to the private runtime asset directory.
    exit /b 1
)
if not exist "%EEG2VIDEO_ASSETS_DIR%\" (
    echo ERROR: EEG2VIDEO_ASSETS_DIR does not exist: "%EEG2VIDEO_ASSETS_DIR%"
    exit /b 1
)
if not defined EEG2VIDEO_WORKER_PYTHON (
    echo ERROR: Set EEG2VIDEO_WORKER_PYTHON to the inference worker's Python executable.
    exit /b 1
)
if not exist "%EEG2VIDEO_WORKER_PYTHON%" (
    echo ERROR: Worker Python executable does not exist: "%EEG2VIDEO_WORKER_PYTHON%"
    exit /b 1
)
if not defined EEG2VIDEO_ASSET_CONFIG set "EEG2VIDEO_ASSET_CONFIG=%EEG2VIDEO_ASSETS_DIR%\asset_config.json"
if not exist "%EEG2VIDEO_ASSET_CONFIG%" (
    echo ERROR: Asset configuration does not exist: "%EEG2VIDEO_ASSET_CONFIG%"
    echo Run tools\prepare_webapp_assets.py or copy webapp\asset_config.example.json into the asset directory.
    exit /b 1
)

set "EEG2VIDEO_UI_PYTHON=%~dp0webapp\.venv\Scripts\python.exe"
if not exist "%EEG2VIDEO_UI_PYTHON%" (
    echo ERROR: Frontend Python environment is missing: "%EEG2VIDEO_UI_PYTHON%"
    echo Create it with: py -3.11 -m venv webapp\.venv
    echo Then install webapp\requirements.txt. This launcher does not install packages.
    exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
"%EEG2VIDEO_UI_PYTHON%" -m webapp.app
set "APP_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %APP_EXIT_CODE%
