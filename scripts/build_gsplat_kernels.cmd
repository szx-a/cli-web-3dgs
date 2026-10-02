@echo off
REM ===========================================================================
REM  Trigger the gsplat CUDA kernel JIT build with full verbose output.
REM
REM  gsplat compiles its kernels on first import. Failures inside that compilation
REM  are swallowed (it only prints "No CUDA toolkit found" or leaves _C = None),
REM  so this script sets VERBOSE=1 and runs the import in-process.
REM
REM  Usage: scripts\build_gsplat_kernels.cmd
REM ===========================================================================
setlocal enableextensions
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\env\venv\Scripts\python.exe"
set "TEMP=%ROOT%\env\tmp"
set "TMP=%ROOT%\env\tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TORCH_EXTENSIONS_DIR=%ROOT%\env\torch_extensions"
set "VSLANG=1033"
REM First build of gsplat's kernels: keep parallelism modest on 15 GB RAM.
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
if defined VCVARS goto have_vcvars

:have_vcvars
if not defined VCVARS goto skip_vcvars
echo [build] MSVC: %VCVARS%
call "%VCVARS%" >nul
:skip_vcvars

if not exist "%ROOT%\tools\cuda\bin\nvcc.exe" goto no_cuda
set "CUDA_HOME=%ROOT%\tools\cuda"
set "CUDA_PATH=%ROOT%\tools\cuda"
set "PATH=%ROOT%\tools\cuda\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
echo [build] portable CUDA: %CUDA_HOME%
goto compile

:no_cuda
echo [build] ERROR: no nvcc at %ROOT%\tools\cuda - run scripts\build_portable_cuda.cmd
exit /b 1

:compile
REM Remove stale ninja locks: if a previous build was interrupted, gsplat/ninja
REM can hang forever waiting on the lock file.
if exist "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\lock" del /q "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\lock"
if exist "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\.ninja_lock" del /q "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\.ninja_lock"

echo [build] compiling gsplat kernels (first run: a few minutes)...
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
set "VERBOSE=1"
"%PY%" -c "import sys; sys.path.insert(0, r'%ROOT%\src'); from gsflow import paths; paths.apply_runtime_env(); from gsplat.cuda._backend import _C; print('RESULT _C is None:', _C is None)" > "%ROOT%\logs\gsplat_build.log" 2>&1
set "RC=%ERRORLEVEL%"
echo [build] exit=%RC%  log: %ROOT%\logs\gsplat_build.log
findstr /C:"RESULT _C" "%ROOT%\logs\gsplat_build.log"
exit /b %RC%
