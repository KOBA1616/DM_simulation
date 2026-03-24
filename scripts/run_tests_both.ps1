param(
  [string]$ExtraArgs = ""
)
# Run fallback then native, saving separate logs
if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
Write-Host "Running fallback tests..."
$env:DM_DISABLE_NATIVE = "1"

# Temporarily hide any compiled native extension files so the Python fallback
# implementation (dm_ai_module.py) is imported during fallback runs.
Write-Host "Temporarily renaming native extension files to enforce Python fallback..."
$nativeCandidates = Get-ChildItem -Path . -Include 'dm_ai_module*.pyd','dm_ai_module*.so' -Recurse -File -ErrorAction SilentlyContinue
$moved = @()
foreach ($f in $nativeCandidates) {
  try {
    $bak = $f.FullName + '.bak'
    Rename-Item -Path $f.FullName -NewName ($f.Name + '.bak') -ErrorAction Stop
    $moved += $bak
  } catch {
    # ignore failures (file locked etc.)
  }
}

mkdir -Force reports\tests | Out-Null
pytest $ExtraArgs 2>&1 | Tee-Object -FilePath reports\tests\pytest_fallback_full.txt

# Restore renamed native extension files
if ($moved.Count -gt 0) {
  Write-Host "Restoring native extension files..."
  foreach ($bakPath in $moved) {
    try {
      $orig = $bakPath.Substring(0, $bakPath.Length - 4)
      Rename-Item -Path $bakPath -NewName ([io.path]::GetFileName($orig)) -ErrorAction Stop
    } catch {
      # ignore restore failures
    }
  }
}

Write-Host "Running native tests..."
Remove-Item Env:DM_DISABLE_NATIVE -ErrorAction SilentlyContinue
pytest $ExtraArgs 2>&1 | Tee-Object -FilePath reports\tests\pytest_native_full.txt

exit $LASTEXITCODE
