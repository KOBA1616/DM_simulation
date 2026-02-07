# Complete clean rebuild script
Write-Host "=== Step 1: Stopping Python processes ===" -ForegroundColor Cyan
Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "`n=== Step 2: Removing build artifacts ===" -ForegroundColor Cyan
Remove-Item -Recurse -Force build-msvc -ErrorAction SilentlyContinue
Remove-Item -Force dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force bin\Release -ErrorAction SilentlyContinue
Write-Host "Cleanup complete" -ForegroundColor Green

Write-Host "`n=== Step 3: Running CMake configure ===" -ForegroundColor Cyan
cmake -B build-msvc -G "Visual Studio 17 2022" -A x64

Write-Host "`n=== Step 4: Building dm_ai_module ===" -ForegroundColor Cyan
cmake --build build-msvc --config Release --target dm_ai_module

Write-Host "`n=== Step 5: Checking build output ===" -ForegroundColor Cyan
if (Test-Path "bin\Release\dm_ai_module.cp312-win_amd64.pyd") {
    $pyd = Get-Item "bin\Release\dm_ai_module.cp312-win_amd64.pyd"
    Write-Host "Built successfully:" -ForegroundColor Green
    Write-Host "  Size: $($pyd.Length) bytes"
    Write-Host "  Time: $($pyd.LastWriteTime)"
    
    Copy-Item $pyd.FullName -Destination "." -Force
    Write-Host "  Copied to workspace root" -ForegroundColor Green
} else {
    Write-Host "Build FAILED - .pyd not found" -ForegroundColor Red
}

Write-Host "`n=== Build complete ===" -ForegroundColor Cyan
