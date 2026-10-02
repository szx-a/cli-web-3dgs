@echo off
REM 兜底下载器：Windows PowerShell 的 curl/Schannel 在本机沙箱下取不到证书，
REM 所以统一改用 Python 的 urllib（走 OpenSSL，正常工作）。
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
"%PY%" "%~dp0download_tools.py" %*
