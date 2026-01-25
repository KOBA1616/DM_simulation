param(
    [Parameter(Position = 0)]
    [string]$Config = "Release",
    [ValidateSet('msvc','mingw')]
    [string]$Toolchain = 'msvc',
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDirName = if ($Toolchain -eq 'mingw') { 'build-mingw' } else { 'build-msvc' }
$buildDir = Join-Path $projectRoot $buildDirName

# Ensure Python output is UTF-8 regardless of Windows locale.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Prefer venv python if available.
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if ($Build) {
    Write-Host "Building project before launching headless runner..."
    & "$scriptDir/build.ps1" -Config $Config -Toolchain $Toolchain
}

if (-not (Test-Path $buildDir)) {
    Write-Host "Warning: Build directory not found at $buildDir. Proceeding anyway."
}

$env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"

Write-Host "Starting headless runner..."
# Default to headless_mainwindow; pass through any extra args to the Python script
& $pythonExe "$projectRoot\scripts\headless_mainwindow.py" @args
