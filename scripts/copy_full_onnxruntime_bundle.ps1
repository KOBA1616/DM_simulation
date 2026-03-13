<#
Copy-Full-OnnxRuntime-Bundle
Search the active virtualenv's site-packages for onnxruntime/onnxruntime_providers_shared and copy
all relevant DLL/PYD files into the repository root to satisfy native loader dependencies.

Usage:
  .\scripts\copy_full_onnxruntime_bundle.ps1 -DryRun
  .\scripts\copy_full_onnxruntime_bundle.ps1

Exit codes:
  0 = success
  2 = no onnxruntime package found
  1 = copy failed
#>

param(
    [switch]$DryRun
)

$venvPath = Join-Path (Get-Location) ".venv\Lib\site-packages"
Write-Host "Looking for onnxruntime under: $venvPath" -ForegroundColor Cyan
if (-not (Test-Path $venvPath)) {
    Write-Host "Virtualenv site-packages not found at: $venvPath" -ForegroundColor Yellow
    exit 2
}

$pkgDirs = Get-ChildItem -Path $venvPath -Directory -Filter "onnxruntime*" -ErrorAction SilentlyContinue
if (-not $pkgDirs -or $pkgDirs.Count -eq 0) {
    Write-Host "No onnxruntime package directory found under site-packages." -ForegroundColor Yellow
    exit 2
}

# Collect candidate files
$candidates = @()
foreach ($d in $pkgDirs) {
    # common locations: root, capi, onnxruntime.capi
    $patterns = @('*.dll','*.pyd')
    foreach ($p in $patterns) {
        $found = Get-ChildItem -Path $d.FullName -Recurse -Include $p -File -ErrorAction SilentlyContinue
        if ($found) { $candidates += $found }
    }
}

if (-not $candidates -or $candidates.Count -eq 0) {
    Write-Host "No DLL/PYD candidates found in onnxruntime package directories." -ForegroundColor Yellow
    exit 2
}

Write-Host "Found $($candidates.Count) candidate files:" -ForegroundColor Green
$candidates | ForEach-Object { Write-Host "  $($_.FullName)" }

$destRoot = (Get-Location).Path
if ($DryRun) { Write-Host "DryRun: would copy these files to $destRoot" -ForegroundColor Cyan; exit 0 }

$failed = $false
foreach ($f in $candidates) {
    $dest = Join-Path $destRoot $f.Name
    try {
        Copy-Item -Path $f.FullName -Destination $dest -Force
        Write-Host "Copied $($f.Name) -> $dest" -ForegroundColor Green
    } catch {
        Write-Host "Failed copying $($f.FullName): $_" -ForegroundColor Red
        $failed = $true
    }
}

if ($failed) { exit 1 } else { exit 0 }
