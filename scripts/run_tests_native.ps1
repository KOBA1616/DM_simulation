param(
  [string]$ExtraArgs = ""
)
# Run tests in native mode (no DM_DISABLE_NATIVE)
if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
Remove-Item Env:DM_DISABLE_NATIVE -ErrorAction SilentlyContinue
mkdir -Force reports\tests | Out-Null
pytest $ExtraArgs 2>&1 | Tee-Object -FilePath reports\tests\pytest_native_full.txt
exit $LASTEXITCODE
