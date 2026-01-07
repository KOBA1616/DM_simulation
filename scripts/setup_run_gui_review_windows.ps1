<#
GUI review minimal setup (Windows) for run_gui_review.ps1.

Purpose:
  - Create .venv (if missing)
  - Install minimal GUI dependencies required to run the Card Editor (PyQt6)
  - Does NOT launch the GUI (use run_gui_review.ps1 for that)

Usage:
  pwsh -File .\scripts\setup_run_gui_review_windows.ps1
  pwsh -File .\scripts\setup_run_gui_review_windows.ps1 -Python py
  pwsh -File .\scripts\setup_run_gui_review_windows.ps1 -Python python
#>

[CmdletBinding()]
param(
  [string]$Python = ''
)

$ErrorActionPreference = 'Stop'

function Info([string]$m) { Write-Host "[info] $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Error $m; exit 1 }

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path

Push-Location $repoRoot
try {
  Info "Repo root: $repoRoot"

  # Choose base python
  $basePython = $null
  if (-not [string]::IsNullOrWhiteSpace($Python)) {
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { Fail "-Python '$Python' not found on PATH." }
    $basePython = $Python
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = 'py'
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $basePython = 'python'
  } else {
    Fail "Python not found. Install Python 3.8+ and ensure it is on PATH (or install the Python Launcher 'py')."
  }

  # Create venv
  $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
  if (-not (Test-Path $venvPython)) {
    Info "Creating venv: .venv"
    if ($basePython -eq 'py') {
      & py -3 -m venv .venv
    } else {
      & $basePython -m venv .venv
    }
  }

  if (-not (Test-Path $venvPython)) { Fail "venv python not found at: $venvPython" }

  Info "Installing minimal GUI deps (PyQt6)"
  & $venvPython -m pip install --upgrade pip setuptools wheel
  & $venvPython -m pip install PyQt6

  Info "Setup complete. Launch with: pwsh -File .\\scripts\\run_gui_review.ps1"
}
finally {
  Pop-Location
}
