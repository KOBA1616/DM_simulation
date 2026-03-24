# Auto build and test script
# Usage: Open PowerShell in repository root and run:
#   .\.venv\Scripts\Activate.ps1; .\scripts\auto_build_and_test.ps1

$ErrorActionPreference = 'Stop'

Write-Host "Activating venv (if present)"
$venvActivate = Join-Path -Path $PWD -ChildPath ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
} else {
    Write-Host "Warning: .venv not found; continuing with system Python"
}

Write-Host "Running quick build script"
if (Test-Path .\scripts\quick_build.ps1) {
    & .\scripts\quick_build.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build script failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "No quick_build.ps1 found; skipping build step"
}

# Locate built .pyd and copy to repo root
Write-Host "Searching for built dm_ai_module*.pyd"
$pyd = Get-ChildItem -Path bin, build, 'build-ninja' -Filter 'dm_ai_module*.pyd' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $pyd) {
    Write-Host "Found pyd: $($pyd.FullName). Copying to repo root"
    Copy-Item $pyd.FullName -Destination . -Force
} else {
    Write-Host "No built pyd found in bin/build folders. Check build output."
}

# Ensure reports directory exists
$reportsDir = Join-Path $PWD 'reports\tests'
if (!(Test-Path $reportsDir)) { New-Item -ItemType Directory -Path $reportsDir -Force | Out-Null }

# Run pytest and capture output (direct call to avoid Start-Process redirect issues)
Write-Host "Running pytest (this may take a while)"
$pytestExe = 'pytest'
$pyOut = & $pytestExe tests/ -q 2>&1
$pyOut | Tee-Object -FilePath (Join-Path $reportsDir 'pytest_latest.txt')
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "Tests passed. Output written to reports/tests/pytest_latest.txt"
} else {
    Write-Error "Tests failed (exit code $exitCode). See reports/tests/pytest_latest.txt"
}

exit $exitCode
