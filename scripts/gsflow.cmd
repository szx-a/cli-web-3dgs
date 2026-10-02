@echo off
REM gsflow CLI 启动器：直接用工作区内的虚拟环境，不污染系统 Python
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM 第三方库的配置/缓存目录全部关进工作区（ultralytics 默认写 AppData 会被沙箱拒绝）
set "YOLO_CONFIG_DIR=%ROOT%\env\config\ultralytics"
set "MPLCONFIGDIR=%ROOT%\env\config\matplotlib"
if not exist "%PY%" (
  echo [gsflow] 未找到虚拟环境：%PY%
  echo [gsflow] 请先运行：powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
  exit /b 1
)
"%PY%" -m gsflow.cli %*
exit /b %ERRORLEVEL%
