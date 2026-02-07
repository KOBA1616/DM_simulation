# Test script to reproduce SELECT_NUMBER issue
$ErrorActionPreference = "Stop"

Write-Host "=== SELECT_NUMBER Debug Test ===" -ForegroundColor Cyan

# Clean old logs
if (Test-Path logs/) {
    Remove-Item logs/*.txt -ErrorAction SilentlyContinue
    Write-Host "Cleaned old logs" -ForegroundColor Yellow
}

# Build first
Write-Host "`nBuilding..." -ForegroundColor Cyan
cmake --build build --config Release 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Build successful" -ForegroundColor Green

# Run GUI for a short time
Write-Host "`nRunning GUI (will auto-stop after 10 seconds)..." -ForegroundColor Cyan
$process = Start-Process -FilePath ".\scripts\run_gui.ps1" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 10
Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue

Write-Host "`nCollecting debug logs..." -ForegroundColor Cyan

# Display key logs
$logFiles = @(
    "logs/select_number_debug.txt",
    "logs/pending_filter_debug.txt",
    "logs/select_number_actions.txt",
    "logs/pending_effects_debug.txt"
)

foreach ($logFile in $logFiles) {
    if (Test-Path $logFile) {
        Write-Host "`n=== $logFile ===" -ForegroundColor Yellow
        Get-Content $logFile | Select-Object -Last 30
    } else {
       Write-Host "`n$logFile: NOT FOUND" -ForegroundColor Red
    }
}

Write-Host "`n=== Analysis Complete ===" -ForegroundColor Cyan
