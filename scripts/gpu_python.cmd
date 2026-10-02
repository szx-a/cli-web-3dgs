@echo off
REM Run an arbitrary python script with MSVC + portable CUDA environment loaded.
REM Usage: scripts\gpu_python.cmd scripts\probe_pose_convention.py [args...]
REM NOTE: keep this file pure ASCII (cmd.exe reads .cmd as ANSI on zh-CN Windows).
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TORCH_EXTENSIONS_DIR=%ROOT%\env\torch_extensions"
set "VSLANG=1033"
set "MAX_JOBS=4"

set "VCVARS="
set "CAND=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
set "CAND=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
set "CAND=D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"

:have_vcvars
if not defined VCVARS goto skip_vcvars
call "%VCVARS%" >nul
:skip_vcvars

if exist "%ROOT%\tools\cuda\bin\nvcc.exe" goto portable
goto run

:portable
set "CUDA_HOME=%ROOT%\tools\cuda"
set "CUDA_PATH=%ROOT%\tools\cuda"
set "PATH=%ROOT%\tools\cuda\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"

:run
"%PY%" %*
exit /b %ERRORLEVEL%
