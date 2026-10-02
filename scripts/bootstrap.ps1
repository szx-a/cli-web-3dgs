# ============================================================================
#  gsflow one-shot environment bootstrap (Windows / PowerShell)
#
#  Why not "python -m venv + pip":
#    Under the restricted sandbox the Python tempfile cleanup fails to chmod
#    (WinError 5), which breaks ensurepip and PEP517 builds (pip install -e .).
#    So we always create the env and install packages with uv, and mount src/
#    through a .pth file instead of an editable install.
#
#  Usage:
#    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1            # core deps only
#    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Gpu       # + torch(cu128) + gsplat
#    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Segment   # + segmentation backends
#    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -All
#
#  NOTE: this file is intentionally pure ASCII. Windows PowerShell 5.1 decodes
#        BOM-less .ps1 files as system ANSI (GBK on zh-CN), which corrupts
#        non-ASCII characters and can produce bogus syntax errors.
# ============================================================================

param(
    [switch]$Gpu,
    [switch]$Segment,
    [switch]$All,
    [switch]$Recreate,
    [string]$Python = 'D:\python\python.exe'
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Venv = Join-Path $Root 'env\venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Tmp = Join-Path $Root 'env\tmp'
$UvCache = Join-Path $Root 'env\uv-cache'

# Keep temp dir and caches inside the project (sandbox cannot write elsewhere)
$env:TEMP = $Tmp
$env:TMP = $Tmp
$env:UV_CACHE_DIR = $UvCache
$env:PIP_CACHE_DIR = Join-Path $Root 'env\pip-cache'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
# Third-party config/cache dirs must stay inside the project too:
# ultralytics defaults to AppData\Roaming (denied), rembg to %USERPROFILE%\.rembg
$env:YOLO_CONFIG_DIR = Join-Path $Root 'env\config\ultralytics'
$env:MPLCONFIGDIR = Join-Path $Root 'env\config\matplotlib'
$env:REMBG_HOME = Join-Path $Root 'models\rembg'
$env:U2NET_HOME = $env:REMBG_HOME

New-Item -ItemType Directory -Force -Path $Tmp, $UvCache, $env:YOLO_CONFIG_DIR, $env:MPLCONFIGDIR, $env:REMBG_HOME | Out-Null
foreach ($d in 'tools','models','workspace','downloads','logs','config') {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}

Write-Host "== gsflow bootstrap ==" -ForegroundColor Cyan
Write-Host "project root : $Root"
Write-Host "virtual env  : $Venv"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install it first: winget install astral-sh.uv   (or: pip install uv)" -ForegroundColor Red
    exit 1
}

if ($Recreate -and (Test-Path $Venv)) {
    Write-Host "removing old venv ..."
    Remove-Item $Venv -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $Py)) {
    Write-Host "[1/4] creating virtual environment ..."
    if (Test-Path $Python) {
        uv venv $Venv --python $Python
    } else {
        uv venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { uv venv $Venv }
} else {
    Write-Host "[1/4] reusing existing virtual environment"
}

Write-Host "[2/4] ensuring pip (uv-created venvs have none by default) ..."
uv pip install --python $Py pip 2>&1 | Out-Null

Write-Host "[3/4] installing core dependencies ..."
uv pip install --python $Py -r (Join-Path $Root 'requirements.txt')

# Mount src/ via .pth instead of an editable install:
# editable installs need a PEP517 build, which fails on this sandbox (chmod).
$SitePackages = Join-Path $Venv 'Lib\site-packages'
Set-Content -Path (Join-Path $SitePackages 'gsflow-dev.pth') -Value (Join-Path $Root 'src') -Encoding ASCII
Write-Host "mounted source path into site-packages (gsflow-dev.pth)"

if ($Gpu -or $All) {
    Write-Host "[4/4] installing torch(cu128) + gsplat ..." -ForegroundColor Yellow
    Write-Host "      torch is about 2.6 GB, keep the network up"
    uv pip install --python $Py --index-url https://download.pytorch.org/whl/cu128 torch torchvision
    uv pip install --python $Py gsplat
    & $Py -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());print('cap',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
} else {
    Write-Host "[4/4] skipping GPU deps (use -Gpu or -All)"
}

if ($Segment -or $All) {
    Write-Host "installing segmentation backends (ultralytics / rembg)..." -ForegroundColor Yellow
    # --no-deps on the heavy packages: torch is already installed, and a plain
    # install would try to re-download it (2.6 GB) on this slow link.
    uv pip install --python $Py --no-deps ultralytics rembg
    uv pip install --python $Py onnxruntime opencv-python-headless numpy pillow pyyaml tqdm psutil py-cpuinfo requests scipy matplotlib pandas
    uv pip install --python $Py pymatting pooch scikit-image numba lazy-loader platformdirs jsonschema attrs
    Write-Host "downloading FastSAM weights with Python urllib (curl fails on this box)..."
    & $Py (Join-Path $Root 'scripts\download_weights.py') fastsam
}

Write-Host ""
Write-Host "done. next steps:" -ForegroundColor Green
Write-Host "  scripts\gsflow.cmd doctor           # environment self-check"
Write-Host "  scripts\web.cmd                     # start the Web GUI"
Write-Host "  scripts\gsflow.cmd run --images <dir> --stages frames,mask"
