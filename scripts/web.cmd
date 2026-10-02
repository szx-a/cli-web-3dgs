@echo off
REM 启动 Web GUI（默认 127.0.0.1:8787）
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
call "%ROOT%\scripts\gsflow.cmd" web --open %*
