param(
    [ValidateSet('msvc','mingw')]
    [string]$Toolchain = 'msvc',
    [string]$Config = "Release",
    [string]$Generator = "",
    [switch]$Clean,
    [switch]$UseLibTorch = $false
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDirName = if ($Toolchain -eq 'mingw') { 'build-mingw' } else { 'build-msvc' }
$buildDir = Join-Path $projectRoot $buildDirName

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

$cmakeArgs = @("-S", $projectRoot, "-B", $buildDir, "-DCMAKE_BUILD_TYPE=$Config")

if ([string]::IsNullOrWhiteSpace($Generator)) {
    if ($Toolchain -eq 'mingw') {
        $Generator = 'MinGW Makefiles'
    } else {
        $Generator = 'Visual Studio 17 2022'
    }
}

if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $cmakeArgs += @("-G", $Generator)
}

if ($Toolchain -eq 'msvc') {
    $cmakeArgs += @('-A', 'x64')
}

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
