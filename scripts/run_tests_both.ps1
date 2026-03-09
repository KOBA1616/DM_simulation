param(
  [string]$ExtraArgs = ""
)
# Run fallback then native, saving separate logs
if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
Write-Host "Running fallback tests..."
$env:DM_DISABLE_NATIVE = "1"
mkdir -Force reports\tests | Out-Null
pytest $ExtraArgs 2>&1 | Tee-Object -FilePath reports\tests\pytest_fallback_full.txt

Write-Host "Running native tests..."
Remove-Item Env:DM_DISABLE_NATIVE -ErrorAction SilentlyContinue
pytest $ExtraArgs 2>&1 | Tee-Object -FilePath reports\tests\pytest_native_full.txt

exit $LASTEXITCODE
