param(
    [Parameter(Position = 0)]
    [string]$Config = "Release",
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectRoot "build"

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if ($Build -or -not (Test-Path (Join-Path $buildDir "dm_ai_module*.pyd"))) {
    Write-Host "Building project before launching GUI..."
    & "$scriptDir/build.ps1" -Config $Config

   
}

if (-not (Test-Path $buildDir)) {
    throw "Build directory not found at $buildDir"
}

# Ensure DLLs are present (Copy from MinGW if available)
$mingwBin = "C:\Users\mediastation36\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20251118-ucrt-x86_64\bin"
if (Test-Path $mingwBin) {
    Write-Host "Copying MinGW DLLs to build directory..."
    Get-ChildItem "$mingwBin\*.dll" | ForEach-Object {
        $dest = Join-Path $buildDir $_.Name
        if (-not (Test-Path $dest)) {
            Copy-Item $_.FullName -Destination $buildDir
        }
    }
}

# Add build directory and project root to PYTHONPATH
$env:PYTHONPATH = "$buildDir;$projectRoot;$env:PYTHONPATH"

Write-Host "Starting GUI..."
python "$projectRoot/dm_toolkit/gui/app.py"
