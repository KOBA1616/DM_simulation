<#
Fix-ORT-For-Tests
Search common build/output dirs for onnxruntime DLLs and copy the most-recent one
into the repo root so tests that load native extension pick the expected ORT DLL.

Usage:
  # dry-run to just list candidates
  .\scripts\fix_ort_for_tests.ps1 -DryRun

  # actually copy the newest found ORT DLL to repo root as onnxruntime.dll
  .\scripts\fix_ort_for_tests.ps1

Exit codes:
  0 = success (copied)
  2 = no candidate found
  1 = copy failed
#>

param(
    [switch]$DryRun
)

$roots = @(
    "build-ninja",
    "build-msvc",
    "build",
    "bin",
    "_deps",
    "."
) | Where-Object { Test-Path $_ }

Write-Host "Searching for onnxruntime DLL candidates under:" -ForegroundColor Cyan
$roots | ForEach-Object { Write-Host "  $_" }

$candidates = @()
foreach ($r in $roots) {
    try {
        $c = Get-ChildItem -Path $r -Filter "onnxruntime*.dll" -File -Recurse -ErrorAction SilentlyContinue
        if ($c) { $candidates += $c }
    } catch {
        # ignore
    }
}

if (-not $candidates -or $candidates.Count -eq 0) {
    Write-Host "No onnxruntime DLL found in searched locations." -ForegroundColor Yellow
    exit 2
}

# Prefer newest by LastWriteTime
$best = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "Found candidate:" $best.FullName -ForegroundColor Green
Write-Host "LastWriteTime:" $best.LastWriteTime

$dest = Join-Path -Path (Get-Location) -ChildPath "onnxruntime.dll"
if ($DryRun) {
    Write-Host "DryRun: Would copy:`n  $($best.FullName)`n -> $dest" -ForegroundColor Cyan
    exit 0
}

try {
    Copy-Item -Path $best.FullName -Destination $dest -Force
    Write-Host "Copied $($best.Name) -> $dest" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "Copy failed: $_" -ForegroundColor Red
    exit 1
}
