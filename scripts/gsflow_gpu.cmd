@echo off
REM ===========================================================================
REM  Run a gsflow command with the full MSVC + portable CUDA environment.
REM
REM  Why: nvcc needs cl.exe on PATH (it is not a standalone compiler), and gsplat
REM       JIT-compiles its CUDA kernels on first use. This wrapper loads the
REM       Visual Studio x64 developer environment (vcvars64.bat) and points at
REM       the portable CUDA in tools\cuda, then runs the given command.
REM
REM  Note: labels + goto are used instead of parenthesised if-blocks, because
REM        paths like "Program Files (x86)" contain parentheses that cmd.exe
REM        mis-parses inside blocks.
REM
REM  Usage:
REM     scripts\gsflow_gpu.cmd run --images D:\data\photos --stages train,export
REM     scripts\gsflow_gpu.cmd doctor
REM ===========================================================================
setlocal enableextensions
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "YOLO_CONFIG_DIR=%ROOT%\env\config\ultralytics"
set "MPLCONFIGDIR=%ROOT%\env\config\matplotlib"
set "TORCH_EXTENSIONS_DIR=%ROOT%\env\torch_extensions"
REM Force English MSVC/cl output. torch's cpp_extension parses the compiler
REM version out of cl.exe's banner; a localized (e.g. Chinese) banner makes it
REM fail with "UnicodeDecodeError: 'cp1' codec can't decode bytes".
set "VSLANG=1033"

if not exist "%PY%" goto no_venv

set "VCVARS="
set "CAND=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
set "CAND=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
set "CAND=D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
set "CAND=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
set "CAND=D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%CAND%" set "VCVARS=%CAND%"
if defined VCVARS goto have_vcvars
goto no_vcvars

:have_vcvars
echo [gsflow] MSVC env: %VCVARS%
call "%VCVARS%" >nul
goto cuda_check

:no_vcvars
where cl.exe >nul 2>nul
if errorlevel 1 goto no_cl
echo [gsflow] using cl.exe already present on PATH
goto cuda_check

:no_cl
echo [gsflow] WARNING: vcvars64.bat not found and cl.exe not on PATH.
echo [gsflow]          CUDA kernel compilation will fail. Install the
echo [gsflow]          "Desktop development with C++" workload in VS2022.

:cuda_check
if exist "%ROOT%\tools\cuda\bin\nvcc.exe" goto portable_cuda
where nvcc >nul 2>nul
if errorlevel 1 goto no_nvcc
echo [gsflow] using nvcc from PATH
set "TORCH_CUDA_ARCH_LIST=12.0"
goto run

:portable_cuda
set "CUDA_HOME=%ROOT%\tools\cuda"
set "CUDA_PATH=%ROOT%\tools\cuda"
set "PATH=%ROOT%\tools\cuda\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
echo [gsflow] portable CUDA: %ROOT%\tools\cuda
"%ROOT%\tools\cuda\bin\nvcc.exe" --version | findstr /C:"release"
goto run

:no_nvcc
echo [gsflow] WARNING: no nvcc found - neither tools\cuda nor PATH.
echo [gsflow]          Build it with: scripts\build_portable_cuda.cmd

:run
echo [gsflow] running: %*
echo.
"%PY%" -m gsflow.cli %*
exit /b %ERRORLEVEL%

:no_venv
echo [gsflow] virtual env not found: %PY%
echo [gsflow] run: powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
exit /b 1
