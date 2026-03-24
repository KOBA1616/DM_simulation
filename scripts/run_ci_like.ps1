# CI-like test runner: run tests in native and fallback modes and save logs
param(
    [string]$LogDir = "reports/tests"
)
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Helper to temporarily rename native extensions to force fallback
function Hide-NativeExtension {
    $pattern = "dm_ai_module*.pyd"
    $files = Get-ChildItem -Path . -Recurse -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        Rename-Item -Path $f.FullName -NewName ($f.Name + ".disabled") -ErrorAction SilentlyContinue
    }
}

function Restore-NativeExtension {
    $pattern = "dm_ai_module*.pyd.disabled"
    $files = Get-ChildItem -Path . -Recurse -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $orig = $f.Name -replace '\.disabled$',''
        Rename-Item -Path $f.FullName -NewName $orig -ErrorAction SilentlyContinue
    }
}

Write-Host "Running fallback (DM_DISABLE_NATIVE=1) tests..."
Hide-NativeExtension
$env:DM_DISABLE_NATIVE = "1"
pytest -q 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "pytest_fallback_full.txt")
Restore-NativeExtension
Remove-Item Env:DM_DISABLE_NATIVE -ErrorAction SilentlyContinue

Write-Host "Running native tests..."
pytest -q 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "pytest_native_full.txt")

Write-Host "Logs saved to" $LogDir
