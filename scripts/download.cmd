@echo off
REM Download helper for bundled tools.
REM Windows PowerShell's curl/Schannel cannot fetch certificates on this box,
REM so downloads go through Python urllib instead.
REM NOTE: keep this file pure ASCII (see gsflow.cmd for why).
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
"%PY%" "%~dp0download_tools.py" %*
