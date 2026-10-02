@echo off
REM ===========================================================================
REM  Trigger the gsplat CUDA kernel JIT build (verbose, with logging).
REM
REM  gsplat compiles its kernels on first import; failures inside that build are
REM  swallowed (it only prints "No CUDA toolkit found" or leaves _C = None).
REM
REM  This box has two extra quirks handled by gsflow:
REM    * torch decodes cl.exe output with the OEM code page, which fails on
REM      zh-CN Windows -> paths.apply_runtime_env() switches it to utf-8.
REM    * ninja completes its work but never exits under this sandbox, and torch
REM      waits for it synchronously -> apply_runtime_env() detects that and
REM      switches to gsflow's built-in direct-compiler builder.
REM
REM  NOTE: keep this file pure ASCII. cmd.exe reads .cmd as ANSI on zh-CN
REM        Windows; non-ASCII comments corrupt the whole script.
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
REM Drop stale ninja locks: an interrupted build can leave ninja waiting forever.
if exist "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\lock" del /q "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\lock"
if exist "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\.ninja_lock" del /q "%TORCH_EXTENSIONS_DIR%\gsplat_cuda\.ninja_lock"
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

echo [build] compiling gsplat kernels (first run: a few minutes)...
set "VERBOSE=1"
"%PY%" -c "import sys, os; sys.path.insert(0, r'%ROOT%\src'); from gsflow import paths; paths.apply_runtime_env(); import torch.utils.cpp_extension as ce; paths.apply_runtime_env(); from gsplat.cuda._backend import _C; print('NINJA_BYPASS:', os.environ.get('GSFLOW_NINJA_BYPASS')); print('RESULT _C is None:', _C is None)" > "%ROOT%\logs\gsplat_build.log" 2>&1
set "RC=%ERRORLEVEL%"
echo [build] exit=%RC%
echo [build] log: %ROOT%\logs\gsplat_build.log
findstr /C:"NINJA_BYPASS" /C:"RESULT _C" "%ROOT%\logs\gsplat_build.log"
exit /b %RC%
