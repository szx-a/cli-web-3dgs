@echo off
REM ===========================================================================
REM  Create the gsflow desktop / Start Menu shortcuts.
REM
REM  Double-click this file (no admin needed). It must be run by YOU because
REM  the agent's sandbox cannot write to the Desktop folder.
REM
REM  Creates:
REM     Desktop : "gsflow Web GUI.lnk"   -> starts the Web GUI and opens browser
REM               "gsflow CLI.lnk"       -> command prompt at the project root
REM     Start   : group "gsflow" with Web GUI / CLI / doctor / portable CUDA
REM
REM  NOTE: keep this file pure ASCII (cmd.exe reads .cmd as ANSI on zh-CN Windows).
REM ===========================================================================
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"

if not exist "%PY%" goto no_python

echo Creating shortcuts for: %ROOT%
echo.
"%PY%" "%ROOT%\scripts\make_shortcuts.py"
echo.
echo Done. Look on your Desktop and in the Start Menu.
echo.
pause
exit /b 0

:no_python
echo [error] virtual env not found: %PY%
echo         run scripts\bootstrap.ps1 first.
pause
exit /b 1
