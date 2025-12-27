<#
Development pipeline helper for this repo.
Usage (PowerShell, venv activated):
  .\scripts\dev_pipeline.ps1            # run full pipeline (install, lint, typecheck, tests)
  .\scripts\dev_pipeline.ps1 -RunGUI   # run GUI at end (will call run_gui_with_real_pyqt.ps1)
  .\scripts\dev_pipeline.ps1 -NoInstall -SkipTests
#>

param(
    [switch]$NoInstall,
    [switch]$SkipTests,
    [switch]$RunGUI
)

function FailExit([int]$code, [string]$msg) {
    Write-Error $msg
    exit $code
}

Write-Host "Starting dev pipeline..."

# ensure python available
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { FailExit 1 "No 'python' on PATH. Activate your virtualenv first." }

if (-not $NoInstall) {
    Write-Host "Installing dependencies from requirements*.txt..."
    & $python -m pip install -U pip setuptools wheel
    & $python -m pip install -r requirements.txt
    if (Test-Path requirements-dev.txt) {
        & $python -m pip install -r requirements-dev.txt
    }
    if ($LASTEXITCODE -ne 0) { FailExit $LASTEXITCODE "pip install failed" }
}

# Lint / Type-check
Write-Host "Running mypy..."
& $python -m mypy .
if ($LASTEXITCODE -ne 0) { Write-Warning "mypy returned non-zero (check types). Continuing." }

# Run tests
if (-not $SkipTests) {
    Write-Host "Running pytest..."
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { FailExit $LASTEXITCODE "pytest failed" }
}

Write-Host "Pipeline steps completed successfully."

if ($RunGUI) {
    Write-Host "Launching GUI with real PyQt6 (runs run_gui_with_real_pyqt.ps1)..."
    & .\scripts\run_gui_with_real_pyqt.ps1
}

exit 0
