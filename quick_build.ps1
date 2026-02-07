# Quick incremental build script for Phase 1.1 + Phase 2
Write-Host "=== Quick Build (Incremental) ===" -ForegroundColor Cyan
Write-Host ""

# Stop Python processes
Write-Host "Stopping Python processes..." -ForegroundColor Yellow
Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Build
Write-Host "Building..." -ForegroundColor Yellow
$buildStart = Get-Date

if (-not (Test-Path build-msvc)) {
    Write-Host "build-msvc not found, running CMake configure..." -ForegroundColor Yellow
    cmake -B build-msvc -G "Visual Studio 17 2022" -A x64
}

cd build-msvc
$buildOutput = cmake --build . --config Release --target dm_ai_module 2>&1
$buildEnd = Get-Date
$buildTime = ($buildEnd - $buildStart).TotalSeconds
cd ..

# Save log
$buildOutput | Out-File -FilePath build_quick.log -Encoding utf8

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build succeeded in $([math]::Round($buildTime, 1))s" -ForegroundColor Green
    
    if (Test-Path "bin\Release\dm_ai_module*.pyd") {
        $pyd = Get-Item "bin\Release\dm_ai_module*.pyd"
        Write-Host "   $($pyd.Name) ($([math]::Round($pyd.Length/1MB, 2)) MB)" -ForegroundColor Gray
        
        # Quick test
        Write-Host ""
        Write-Host "Quick test..." -ForegroundColor Yellow
        python -c "import dm_ai_module; gs = dm_ai_module.GameState(42); print('✓ Import OK'); print('✓ is_human_player:', hasattr(gs, 'is_human_player'))"
    } else {
        Write-Host "⚠️  .pyd file not found!" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Last 10 errors:" -ForegroundColor Yellow
    $buildOutput | Select-String "error" | Select-Object -Last 10
    Write-Host ""
    Write-Host "See build_quick.log for full log" -ForegroundColor Gray
    exit 1
}
