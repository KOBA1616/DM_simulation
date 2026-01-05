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

function Find-DmAiModuleDir {
    param([Parameter(Mandatory)][string]$Root)

    $candidates = @(
        (Join-Path $Root "bin"),
        (Join-Path $Root "build-msvc"),
        (Join-Path $Root "build-mingw"),
        (Join-Path $Root "build")
    ) | Where-Object { Test-Path $_ }

    foreach ($dir in $candidates) {
        $pyd = Get-ChildItem -Path $dir -Recurse -File -Filter "dm_ai_module*.pyd" -ErrorAction SilentlyContinue |
            Sort-Object FullName |
            Select-Object -First 1
        if ($pyd) {
            return (Split-Path -Parent $pyd.FullName)
        }
    }
    return $null
}

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if ($Build -or -not (Find-DmAiModuleDir -Root $projectRoot)) {
    Write-Host "Building project before launching GUI..."
    & "$scriptDir/build.ps1" -Config $Config -Toolchain $Toolchain

   
}

if (-not (Test-Path $buildDir)) {
    throw "Build directory not found at $buildDir"
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

$dmDir = Find-DmAiModuleDir -Root $projectRoot
if ($dmDir) {
    $env:PYTHONPATH = "$dmDir;$buildDir;$projectRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = "$buildDir;$projectRoot;$env:PYTHONPATH"
}

Write-Host "Starting GUI..."
& $pythonExe "$projectRoot/dm_toolkit/gui/app.py"
