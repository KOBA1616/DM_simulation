# PowerShell helper to run the GUI with a real PyQt6 installation.
# This script ensures PyQt6 is installed and then runs the standard GUI launcher.

param(
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"

# 1. Ensure python is available
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-Error "Python not found on PATH. Please activate your virtual environment."
    exit 1
}

# 2. Ensure PyQt6 is installed
if (-not $NoInstall) {
    Write-Host "Ensuring PyQt6 is installed..."
    & $python -m pip install PyQt6
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyQt6."
        exit 1
    }
}

# 3. Run the GUI launcher
$RunGuiScript = Join-Path $PSScriptRoot "run_gui.ps1"

if (Test-Path $RunGuiScript) {
    Write-Host "Launching run_gui.ps1..."
    & $RunGuiScript
} else {
    Write-Error "Could not find run_gui.ps1 in $PSScriptRoot"
    exit 1
}
