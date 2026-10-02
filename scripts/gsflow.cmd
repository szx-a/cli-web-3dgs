@echo off
REM gsflow CLI launcher - uses the project virtual environment only.
REM NOTE: keep this file pure ASCII; cmd.exe reads .cmd as ANSI on zh-CN Windows
REM and non-ASCII comments corrupt the whole script.
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM Third-party config/cache dirs must stay inside the project too.
set "YOLO_CONFIG_DIR=%ROOT%\env\config\ultralytics"
set "MPLCONFIGDIR=%ROOT%\env\config\matplotlib"
set "TORCH_EXTENSIONS_DIR=%ROOT%\env\torch_extensions"
if not exist "%PY%" (
  echo [gsflow] virtual env not found: %PY%
  echo [gsflow] run: powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
  exit /b 1
)
"%PY%" -m gsflow.cli %*
exit /b %ERRORLEVEL%
