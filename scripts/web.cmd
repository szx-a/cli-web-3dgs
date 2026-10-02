@echo off
REM Start the gsflow Web GUI (default http://127.0.0.1:8787).
REM NOTE: keep this file pure ASCII; cmd.exe reads .cmd as ANSI on zh-CN Windows
REM and non-ASCII comments corrupt the whole script.
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
call "%ROOT%\scripts\gsflow.cmd" web --open %*
