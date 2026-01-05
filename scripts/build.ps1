param(
    [string]$Config = "Release",
    [string]$Generator = "MinGW Makefiles",
    [switch]$Clean,
    [switch]$UseLibTorch = $false
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectRoot "build"

# Ensure UTF-8 for console output
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

Set-Location $projectRoot

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item -Path $buildDir -Recurse -Force
}

if (-not (Test-Path $buildDir)) {
    
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

$cmakeArgs = @("-S", $projectRoot, "-B", $buildDir, "-G", $Generator, "-DCMAKE_BUILD_TYPE=$Config")

if ($UseLibTorch) {
    Write-Host "Detecting LibTorch path from Python..."
    
    # Try to find python in .venv first
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $pythonCmd = $venvPython
    } else {
        $pythonCmd = "python"
    }

    try {
        $torchPath = & $pythonCmd -c "import torch; print(torch.utils.cmake_prefix_path)"
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($torchPath)) {
            Write-Host "Found LibTorch at: $torchPath"
            $cmakeArgs += "-DUSE_LIBTORCH=ON"
            $cmakeArgs += "-DCMAKE_PREFIX_PATH=$torchPath"
        } else {
            Write-Warning "Could not detect LibTorch path via '$pythonCmd'. Ensure 'torch' is installed in the active Python environment."
            $cmakeArgs += "-DUSE_LIBTORCH=ON"
        }
    } catch {
        Write-Warning "Error detecting LibTorch: $_"
        $cmakeArgs += "-DUSE_LIBTORCH=ON"
    }
} else {
    $cmakeArgs += "-DUSE_LIBTORCH=OFF"
}

Write-Host "Configuring (Generator=$Generator, Config=$Config)..."
cmake @cmakeArgs

Write-Host "Building..."
cmake --build $buildDir --config $Config

Write-Host "Build complete." 
