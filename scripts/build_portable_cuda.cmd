@echo off
REM ===========================================================================
REM  Build a portable, admin-free CUDA toolkit inside tools\cuda
REM
REM  The official CUDA installer is a 7z self-extracting archive, and Windows
REM  ships bsdtar (tar.exe) which can read 7z. So we only need three sub-packages
REM  (cuda_nvcc: the compiler; cuda_cudart: runtime headers+libs; cuda_cccl:
REM  thrust/cub) to get a working nvcc without any administrator install.
REM
REM  Result: tools\cuda with bin\nvcc.exe, include\, lib\x64\  (about 360 MB)
REM
REM  Usage: scripts\build_portable_cuda.cmd
REM ===========================================================================
setlocal enableextensions
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PKG=%ROOT%\env\cuda_pkg"
set "CUDA=%ROOT%\tools\cuda"
set "EXE=%ROOT%\downloads\cuda_12.8.1_572.61_windows.exe"

if not exist "%EXE%" (
  echo [gsflow] installer not found: %EXE%
  echo [gsflow] download it first: scripts\download.cmd cuda
  exit /b 1
)

if exist "%CUDA%\bin\nvcc.exe" (
  echo [gsflow] portable CUDA already present: %CUDA%
  "%CUDA%\bin\nvcc.exe" --version | findstr /C:"release"
  exit /b 0
)

echo [gsflow] extracting sub-packages with bsdtar (a few minutes, several GB read)...
if not exist "%PKG%" mkdir "%PKG%"
tar -xf "%EXE%" -C "%PKG%" cuda_nvcc cuda_cudart cuda_cccl
if errorlevel 1 (
  echo [gsflow] ERROR: extraction failed.
  exit /b 1
)

echo [gsflow] assembling %CUDA% ...
if exist "%CUDA%" rmdir /s /q "%CUDA%"
mkdir "%CUDA%\bin" 2>nul
mkdir "%CUDA%\include" 2>nul
mkdir "%CUDA%\lib\x64" 2>nul

xcopy /e /i /y /q "%PKG%\cuda_nvcc\nvcc\*" "%CUDA%\" >nul
xcopy /e /i /y /q "%PKG%\cuda_cudart\cudart\include\*" "%CUDA%\include\" >nul
xcopy /e /i /y /q "%PKG%\cuda_cudart\cudart\lib\x64\*" "%CUDA%\lib\x64\" >nul
xcopy /e /i /y /q "%PKG%\cuda_cudart\cudart\bin\*" "%CUDA%\bin\" >nul
xcopy /e /i /y /q "%PKG%\cuda_cccl\thrust\include\*" "%CUDA%\include\" >nul
if exist "%CUDA%\bin\crt" (
  mkdir "%CUDA%\include\crt" 2>nul
  xcopy /e /i /y /q "%CUDA%\bin\crt\*" "%CUDA%\include\crt\" >nul
)

if not exist "%CUDA%\bin\nvcc.exe" (
  echo [gsflow] ERROR: nvcc.exe missing after assembly.
  exit /b 1
)

echo.
echo [gsflow] portable CUDA ready: %CUDA%
"%CUDA%\bin\nvcc.exe" --version | findstr /C:"release"
echo.
echo [gsflow] use it via: scripts\gsflow_gpu.cmd ^<gsflow args^>
endlocal
