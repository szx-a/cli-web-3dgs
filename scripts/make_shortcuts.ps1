# ============================================================================
#  Create gsflow desktop / Start Menu shortcuts.
#
#  RUN THIS YOURSELF (no admin needed) - the agent's sandbox cannot write to
#  the Desktop folder, and elevation does not propagate to child processes.
#
#    Right-click -> "Run with PowerShell"
#    or in a terminal:  powershell -ExecutionPolicy Bypass -File scripts\make_shortcuts.ps1
#
#  Creates on the Desktop:
#     gsflow Web GUI.lnk   - starts the Web GUI and opens the browser
#     gsflow CLI.lnk       - command prompt at the project root
#  Creates in the Start Menu under "gsflow":
#     1. Web GUI / 2. CLI (GPU env) / 3. Doctor / 4. Build portable CUDA / 5. Project folder
#
#  NOTE: keep this file pure ASCII + UTF-8 BOM (Windows PowerShell 5.1 decodes
#        BOM-less .ps1 as ANSI and corrupts non-ASCII text).
# ============================================================================

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$WebCmd = Join-Path $Root 'scripts\web.cmd'
$GpuCmd = Join-Path $Root 'scripts\gsflow_gpu.cmd'
$GsflowCmd = Join-Path $Root 'scripts\gsflow.cmd'
$BuildCuda = Join-Path $Root 'scripts\build_portable_cuda.cmd'

$Desktop = [Environment]::GetFolderPath('Desktop')
$StartMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\gsflow'

Write-Host "project   : $Root"
Write-Host "desktop   : $Desktop"
Write-Host "start menu: $StartMenu"
Write-Host ""

$shell = New-Object -ComObject WScript.Shell

function New-Lnk {
    param([string]$Path, [string]$Target, [string]$Arguments, [string]$WorkDir, [string]$Desc)
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $sc = $shell.CreateShortcut($Path)
    $sc.TargetPath = $Target
    $sc.Arguments = $Arguments
    $sc.WorkingDirectory = $WorkDir
    $sc.Description = $Desc
    $sc.WindowStyle = 1
    $sc.Save()
    if (Test-Path $Path) {
        Write-Host "  [OK]   $Path" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $Path" -ForegroundColor Red
    }
}

$cmdExe = Join-Path $env:SystemRoot 'System32\cmd.exe'
$cliArgs = '/k "chcp 65001 >nul && title gsflow && cd /d ' + $Root + '"'

Write-Host "Desktop shortcuts:"
New-Lnk -Path (Join-Path $Desktop 'gsflow Web GUI.lnk') -Target $WebCmd -Arguments '' -WorkDir $Root -Desc 'Start the gsflow Web GUI (127.0.0.1:8787)'
New-Lnk -Path (Join-Path $Desktop 'gsflow CLI.lnk') -Target $cmdExe -Arguments $cliArgs -WorkDir $Root -Desc 'gsflow command prompt at the project root'

Write-Host ""
Write-Host "Start Menu shortcuts:"
New-Lnk -Path (Join-Path $StartMenu '1. Web GUI.lnk') -Target $WebCmd -Arguments '' -WorkDir $Root -Desc 'gsflow Web GUI'
New-Lnk -Path (Join-Path $StartMenu '2. CLI with GPU env.lnk') -Target $GpuCmd -Arguments '' -WorkDir $Root -Desc 'gsflow CLI with MSVC + portable CUDA environment'
New-Lnk -Path (Join-Path $StartMenu '3. Doctor.lnk') -Target $cmdExe -Arguments ('/k "chcp 65001 >nul && cd /d ' + $Root + ' && scripts\gsflow.cmd doctor"') -WorkDir $Root -Desc 'gsflow environment self-check'
New-Lnk -Path (Join-Path $StartMenu '4. Build portable CUDA.lnk') -Target $BuildCuda -Arguments '' -WorkDir $Root -Desc 'Build the admin-free CUDA toolchain from the downloaded installer'
New-Lnk -Path (Join-Path $StartMenu '5. Project folder.lnk') -Target $Root -Arguments '' -WorkDir $Root -Desc 'Open the gsflow project folder'

Write-Host ""
Write-Host "Done. Check your Desktop and the Start Menu group 'gsflow'." -ForegroundColor Cyan
Write-Host "Tip: to start training, use 'gsflow CLI with GPU env' (it loads cl.exe + nvcc)."
