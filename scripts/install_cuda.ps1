# ============================================================================
#  Install CUDA Toolkit 12.8.1 silently (RUN AS ADMINISTRATOR)
#
#  Why: gsplat and the official 3DGS rasterizer both need nvcc to JIT-compile
#       CUDA kernels. Without nvcc, gsplat prints "No CUDA toolkit found"
#       and silently disables the GPU path.
#
#  Usage:
#    1) download the installer first (no admin needed):
#         scripts\download.cmd cuda
#    2) then, in an ADMIN PowerShell:
#         powershell -ExecutionPolicy Bypass -File scripts\install_cuda.ps1
#
#  Defaults are deliberately conservative: the official "-s" (silent) with
#  "-n" (no reboot) and NO -InstallPath override. The earlier attempt passed
#  "-nrm" plus an -InstallPath and died with exit code 0xE0E00019; the installer
#  log showed it had loaded all 66 sub-packages fine, so we avoid the extras.
#
#  Switches:
#    -DryRun        : only stage the extracted package, do not install
#    -Components X  : comma list of sub-packages (e.g. "nvcc_12.8,cudart_12.8")
#    -InstallDir D  : custom install path (passed as -InstallPath)
#
#  NOTE: this file is intentionally pure ASCII. Windows PowerShell 5.1 decodes
#        BOM-less .ps1 files as system ANSI (GBK on zh-CN), which corrupts
#        non-ASCII characters and can produce bogus syntax errors.
# ============================================================================

param(
    [string]$Installer = '',
    [string]$Components = '',
    [string]$InstallDir = '',
    [switch]$DryRun,
    [switch]$SkipDownload
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $Installer) { $Installer = Join-Path $Root 'downloads\cuda_12.8.1_572.61_windows.exe' }
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "ERROR: run this script in an ADMINISTRATOR PowerShell." -ForegroundColor Red
        Write-Host "       Right-click Start -> Terminal (Admin) / Windows PowerShell (Admin)."
        exit 1
    }
}

function Show-ExitCode([int]$code) {
    $hex = ('0x{0:X8}' -f ($code -band 0xFFFFFFFF))
    Write-Host ("   exit code: {0}  ({1})" -f $code, $hex) -ForegroundColor Yellow
    if ($code -eq -522190823 -or $hex -eq '0xE0E00019') {
        Write-Host "   hint: 0xE0E00019 = launcher/package failure." -ForegroundColor Yellow
        Write-Host "         Check the NVIDIA setup log under:"
        Write-Host "         $env:LOCALAPPDATA\Temp\NvidiaLogging\Log.*setup.exe.log"
    }
}

Assert-Admin

if (-not (Test-Path $Installer)) {
    if ($SkipDownload) { Write-Host "ERROR: installer not found: $Installer" -ForegroundColor Red; exit 1 }
    Write-Host "Installer not found, downloading first (~3.2 GB, slow link, resumable)..." -ForegroundColor Yellow
    $py = Join-Path $Root 'env\venv\Scripts\python.exe'
    & $py (Join-Path $Root 'scripts\download_tools.py') cuda
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: download failed." -ForegroundColor Red; exit 1 }
}
if (-not (Test-Path $Installer)) { Write-Host "ERROR: still missing: $Installer" -ForegroundColor Red; exit 1 }

$sizeGB = (Get-Item $Installer).Length / 1GB
Write-Host ("Installer: {0} ({1:N2} GB)" -f $Installer, $sizeGB)
Write-Host ""

# Build the argument list: keep it minimal and official.
$installArgs = @()
if ($DryRun) {
    $installArgs += '-s'
    Write-Host "MODE: dry run - extract packages only, no install" -ForegroundColor Cyan
} else {
    $installArgs += '-s'
    if ($Components) {
        foreach ($c in ($Components -split ',')) {
            $name = $c.Trim()
            if ($name) { $installArgs += $name }
        }
        Write-Host ("MODE: silent install, components: {0}" -f $Components) -ForegroundColor Cyan
    } else {
        Write-Host "MODE: silent install, installer default component selection" -ForegroundColor Cyan
    }
    $installArgs += '-n'          # do not reboot afterwards
    if ($InstallDir) {
        $installArgs += "-InstallPath=`"$InstallDir`""
        Write-Host "      custom path: $InstallDir"
    }
}

Write-Host ("Running: `"{0}`" {1}" -f $Installer, ($installArgs -join ' '))
Write-Host "This can take 5-20 minutes. Do not close this window." -ForegroundColor Yellow
Write-Host ""

# Give the launcher a stable, writable temp root. The launcher extracts the
# whole package (about 5 GB) into %TEMP%\cuda and then relaunches setup.exe
# from there; if %TEMP% points at a per-session sandbox folder that is gone by
# then, the launcher can die with 0xE0E00019 and leave no install log at all.
$workTemp = Join-Path $LogDir 'cuda_temp'
New-Item -ItemType Directory -Force -Path $workTemp | Out-Null
Write-Host "Launcher temp root: $workTemp"
$env:TEMP = $workTemp
$env:TMP = $workTemp

$proc = Start-Process -FilePath $Installer -ArgumentList $installArgs -Wait -PassThru
Write-Host "Installer finished."
Show-ExitCode $proc.ExitCode

# Environment variables (user scope is enough; torch finds CUDA_HOME / CUDA_PATH)
$nvccFound = $null
foreach ($cand in @(
    'D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe',
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe'
)) {
    if (Test-Path $cand) { $nvccFound = $cand; break }
}

Write-Host ""
Write-Host "== Verify ==" -ForegroundColor Cyan
if ($nvccFound) {
    $dir = Split-Path -Parent (Split-Path -Parent $nvccFound)
    [Environment]::SetEnvironmentVariable('CUDA_PATH', $dir, 'User')
    [Environment]::SetEnvironmentVariable('CUDA_HOME', $dir, 'User')
    $env:CUDA_PATH = $dir
    $env:CUDA_HOME = $dir
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    foreach ($p in @((Join-Path $dir 'bin'), (Join-Path $dir 'lib\x64'))) {
        if ($userPath -notlike "*$p*") { $userPath = "$userPath;$p" }
    }
    [Environment]::SetEnvironmentVariable('Path', $userPath, 'User')

    & $nvccFound --version | Select-Object -Last 4
    Write-Host "nvcc ready: $nvccFound" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next (open a NEW terminal so PATH is picked up):"
    Write-Host "  env\venv\Scripts\python.exe scripts\status.py"
} else {
    Write-Host "nvcc NOT found in the default locations." -ForegroundColor Red
    Write-Host "Check the NVIDIA setup log:"
    Write-Host "  $env:LOCALAPPDATA\Temp\NvidiaLogging\"
    Write-Host "  $workTemp\cuda\   (extracted package + setup logs)"
    Write-Host ""
    Write-Host "Fallbacks, in order of preference:" -ForegroundColor Yellow
    Write-Host "  1) retry with an explicit component list:"
    Write-Host "     powershell -ExecutionPolicy Bypass -File scripts\install_cuda.ps1 -Components `"nvcc_12.8,cudart_12.8,nvrtc_12.8,nvrtc_dev_12.8,thrust_12.8,crt_12.8,cublas_12.8,cublas_dev_12.8,visual_studio_integration_12.8`""
    Write-Host "  2) run setup.exe interactively from the extracted temp folder and pick"
    Write-Host "     'Custom (Advanced)' -> uncheck everything except CUDA -> Development components"
    Write-Host "     (extracted folder: $workTemp\cuda)"
    Write-Host "  3) skip CUDA entirely: only the GPU training stage needs it;"
    Write-Host "     frames / mask / colmap / export all work without it."
}
