param(
    [string]$Config = "Release",
    [string]$Generator = "MinGW Makefiles",
    [switch]$Clean
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

Write-Host "Configuring (Generator=$Generator, Config=$Config)..."
cmake -S $projectRoot -B $buildDir -G $Generator -DCMAKE_BUILD_TYPE=$Config

Write-Host "Building..."
cmake --build $buildDir --config $Config

Write-Host "Build complete." 
