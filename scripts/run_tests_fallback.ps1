param(
  [string]$ExtraArgs = ""
)
# Run tests in fallback mode (force Python fallback implementation)
if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
$env:DM_DISABLE_NATIVE = "1"
mkdir -Force reports\tests | Out-Null
pytest $ExtraArgs 2>&1 | Tee-Object -FilePath reports\tests\pytest_fallback_full.txt
exit $LASTEXITCODE
