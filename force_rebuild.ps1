# Force rebuild by touching source files
Write-Host "=== Forcing source file timestamps update ===" -ForegroundColor Cyan

# Touch source files
$files = @(
    "src\engine\game_instance.cpp",
    "src\engine\game_instance.hpp",
    "src\engine\actions\strategies\phase_strategies.cpp"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        (Get-Item $file).LastWriteTime = Get-Date
        Write-Host "  Touched: $file" -ForegroundColor Green
    }
}

# Stop Python
Write-Host "`n=== Stopping Python processes ===" -ForegroundColor Cyan
Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Remove old binaries 
Write-Host "`n=== Removing old binaries ===" -ForegroundColor Cyan
Remove-Item -Force dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Force bin\Release\dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build-msvc\Release\* -ErrorAction SilentlyContinue
Write-Host "  Cleanup complete" -ForegroundColor Green

# Build
Write-Host "`n=== Building ===" -ForegroundColor Cyan
cmake --build build-msvc --config Release --target dm_ai_module

# Check result
Write-Host "`n=== Checking result ===" -ForegroundColor Cyan
if (Test-Path "bin\Release\dm_ai_module.cp312-win_amd64.pyd") {
    $pyd = Get-Item "bin\Release\dm_ai_module.cp312-win_amd64.pyd"
    Write-Host "SUCCESS! Built .pyd file:" -ForegroundColor Green
    Write-Host "  Path: $($pyd.FullName)" -ForegroundColor White
    Write-Host "  Size: $($pyd.Length) bytes" -ForegroundColor White
    Write-Host "  Modified: $($pyd.LastWriteTime)" -ForegroundColor White
    
    Copy-Item $pyd.FullName -Destination "." -Force
    Write-Host "`nCopied to workspace root" -ForegroundColor Green
    
    # Test
    Write-Host "`n=== Quick test ===" -ForegroundColor Cyan
    python -c "import dm_ai_module; print('Module version check:', dm_ai_module.__file__)" 2>&1 | Select-Object -Last 5
} else {
    Write-Host "FAILED - .pyd file not found" -ForegroundColor Red
}
