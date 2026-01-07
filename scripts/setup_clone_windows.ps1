<#
Clone distribution setup script (Windows).

Goal:
  - Verify required tools exist (Git, Python, CMake)
  - Create .venv if missing
  - Install dependencies
  - Build native module via CMake
  - Run a minimal smoke test (incl. official GUI stubbing test)

Usage:
  pwsh -File .\scripts\setup_clone_windows.ps1
  pwsh -File .\scripts\setup_clone_windows.ps1 -Toolchain msvc -Configuration Release
  pwsh -File .\scripts\setup_clone_windows.ps1 -SkipBuild
#>

[CmdletBinding()]
param(
  [ValidateSet('msvc','mingw')]
  [string]$Toolchain = 'msvc',

  [ValidateSet('Release','RelWithDebInfo','Debug')]
  [string]$Configuration = 'Release',

  [switch]$SkipBuild,
  [switch]$SkipSmokeTests,
  [switch]$NoInstall,

  # Optional: explicitly pick the base python command used to create venv
  [string]$Python = ''
)

$ErrorActionPreference = 'Stop'

function Info([string]$m) { Write-Host "[info] $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "[warn] $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Error $m; exit 1 }

# Ensure UTF-8 for console output and Python I/O regardless of Windows locale.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
Push-Location $repoRoot
try {
  Info "Repo root: $repoRoot"

  # --- Tool checks ---
  foreach ($t in @('git','cmake')) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
      Fail "Required tool not found on PATH: $t"
    }
  }

  # Pick base python to create venv
  $basePython = $null
  if (-not [string]::IsNullOrWhiteSpace($Python)) {
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
      Fail "-Python '$Python' not found on PATH."
    }
    $basePython = $Python
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = 'py'
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $basePython = 'python'
  } else {
    Fail "Python not found. Install Python 3.8+ and ensure it is on PATH (or install the Python Launcher 'py')."
  }

  # Validate python version (>= 3.8)
  $verOk = $false
  try {
    if ($basePython -eq 'py') {
      & py -3 -c "import sys; print(sys.version)" | Out-Null
      $verOk = $true
    } else {
      & $basePython -c "import sys; print(sys.version)" | Out-Null
      $verOk = $true
    }
  } catch {
    $verOk = $false
  }
  if (-not $verOk) {
    Fail "Failed to execute Python. Please confirm Python installation works from this terminal."
  }

  # --- Create venv ---
  $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
  if (-not (Test-Path $venvPython)) {
    Info "Creating venv: .venv"
    if ($basePython -eq 'py') {
      & py -3 -m venv .venv
    } else {
      & $basePython -m venv .venv
    }
  } else {
    Info "Reusing existing venv: .venv"
  }

  if (-not (Test-Path $venvPython)) {
    Fail "venv python not found at: $venvPython"
  }

  # --- Install deps ---
  if (-not $NoInstall) {
    Info "Upgrading pip/setuptools/wheel"
    & $venvPython -m pip install --upgrade pip setuptools wheel

    if (Test-Path 'requirements-dev.txt') {
      Info "Installing Python deps: requirements-dev.txt"
      & $venvPython -m pip install -r requirements-dev.txt
    } elseif (Test-Path 'requirements.txt') {
      Info "Installing Python deps: requirements.txt"
      & $venvPython -m pip install -r requirements.txt
    } else {
      Warn "No requirements*.txt found; skipping dependency install"
    }
  } else {
    Info "Skipping dependency install (-NoInstall)"
  }

  # --- Build native module ---
  if (-not $SkipBuild) {
    Info "Building native components via scripts/build.ps1 (Toolchain=$Toolchain, Configuration=$Configuration)"
    & (Join-Path $scriptDir 'build.ps1') -Toolchain $Toolchain -Config $Configuration
  } else {
    Info "Skipping build (-SkipBuild)"
  }

  # --- Smoke tests ---
  if (-not $SkipSmokeTests) {
    Info "Smoke test: import dm_toolkit + dm_ai_module"
    & $venvPython -c "import dm_toolkit; import dm_ai_module; print('dm_ai_module.IS_NATIVE=', getattr(dm_ai_module,'IS_NATIVE',None))"

    # Official GUI stubbing verification (headless safe)
    $stubRunner = Join-Path $repoRoot 'run_pytest_with_pyqt_stub.py'
    if (Test-Path $stubRunner) {
      Info "Smoke test: GUI stubbing test (headless)"
      & $venvPython $stubRunner -q python/tests/gui/test_gui_stubbing.py
    } else {
      Warn "Stub runner not found at: $stubRunner (skipping GUI stub test)"
    }
  } else {
    Info "Skipping smoke tests (-SkipSmokeTests)"
  }

  Info "Setup completed successfully."
  Info "Next: .\\scripts\\run_gui.ps1 (or run CI-local: .\\scripts\\run_ci_local.ps1)"
}
finally {
  Pop-Location
}
