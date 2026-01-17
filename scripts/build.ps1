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

function Invoke-VsDevCmd {
    param(
        [ValidateSet('x64','x86')]
        [string]$Arch = 'x64'
    )

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found at: $vswhere"
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installPath)) {
        throw "Visual Studio with MSVC tools not found via vswhere."
    }

    $vsDevCmd = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat not found at: $vsDevCmd"
    }

    # Import the environment variables produced by VsDevCmd into this PowerShell session.
    # Using cmd.exe because VsDevCmd is a .bat file.
    $cmd = "`"$vsDevCmd`" -no_logo -arch=$Arch -host_arch=$Arch && set"
    cmd.exe /s /c $cmd | ForEach-Object {
        $line = $_
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) {
            $name = $line.Substring(0, $idx)
            $value = $line.Substring($idx + 1)
            try { Set-Item -Path "Env:$name" -Value $value } catch { }
        }
    }
}

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
    # Ensure MSVC standard library include paths etc. are available even when
    # running from a plain PowerShell (not Developer Command Prompt).
    Invoke-VsDevCmd -Arch 'x64'
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
