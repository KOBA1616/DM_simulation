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
    Write-Host "Building project before launching GUI..."
    & "$scriptDir/build.ps1" -Config $Config -Toolchain $Toolchain
}

if (-not (Test-Path $buildDir)) {
    # If build dir missing, we might still have a pyd elsewhere, but it's safer to warn or rely on dm_ai_module.py to fail.
    # However, for consistency with 'Build on demand', we rely on user passing -Build or valid existing.
    Write-Host "Warning: Build directory not found at $buildDir. Assuming native module exists in standard location or irrelevant."
}

# Optional: ensure MinGW DLLs are present (only when explicitly configured)
$mingwBin = $env:DM_MINGW_BIN
if (-not [string]::IsNullOrWhiteSpace($mingwBin) -and (Test-Path $mingwBin)) {
    Write-Host "Copying MinGW DLLs from '$mingwBin' to build directory..."
    Get-ChildItem "$mingwBin\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        $dest = Join-Path $buildDir $_.Name
        if (-not (Test-Path $dest)) {
            Copy-Item $_.FullName -Destination $buildDir
        }
    }
}

# PYTHONPATH: Only add project root (and build dir for safety of other imports), do NOT search for pyd.
$env:PYTHONPATH = "$buildDir;$projectRoot;$env:PYTHONPATH"

Write-Host "Starting GUI..."
& $pythonExe "$projectRoot/dm_toolkit/gui/app.py"
