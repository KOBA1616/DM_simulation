#!/usr/bin/env pwsh
# Build and test Phase 1 + Phase 2 implementation

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Building Phase 1 + Phase 2: SimpleAI + Player Modes" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

# Step 1: Configure CMake if needed
if (-not (Test-Path "build-msvc\CMakeCache.txt")) {
    Write-Host "`n[1/4] Configuring CMake..." -ForegroundColor Cyan
    cmake -B build-msvc -G "Visual Studio 17 2022" -A x64 `
          -DCMAKE_BUILD_TYPE=Release `
          -DPython3_ROOT_DIR=$env:VIRTUAL_ENV
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ CMake configuration failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[1/4] CMake already configured, skipping..." -ForegroundColor Green
}

# Step 2: Build C++ module
Write-Host "`n[2/4] Building dm_ai_module..." -ForegroundColor Cyan
cmake --build build-msvc --config Release --target dm_ai_module
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build completed" -ForegroundColor Green

# Step 3: Copy module to site-packages
Write-Host "`n[3/4] Installing module to site-packages..." -ForegroundColor Cyan
$pydFile = Get-ChildItem -Path "build-msvc\Release\dm_ai_module.*.pyd" | Select-Object -First 1
if ($pydFile) {
    Copy-Item -Force $pydFile.FullName "$env:VIRTUAL_ENV\Lib\site-packages\"
    Write-Host "✅ Module installed: $($pydFile.Name)" -ForegroundColor Green
} else {
    Write-Host "❌ .pyd file not found" -ForegroundColor Red
    exit 1
}

# Step 4: Run tests
Write-Host "`n[4/4] Running tests..." -ForegroundColor Cyan
Write-Host "`n--- Phase 1 Test ---" -ForegroundColor Yellow
python test_phase1_simple_ai.py
$phase1Result = $LASTEXITCODE

Write-Host "`n--- Phase 2 Test ---" -ForegroundColor Yellow
python test_phase2_player_modes.py
$phase2Result = $LASTEXITCODE

# Summary
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "BUILD AND TEST SUMMARY" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

if ($phase1Result -eq 0) {
    Write-Host "✅ Phase 1 (SimpleAI): PASSED" -ForegroundColor Green
} else {
    Write-Host "❌ Phase 1 (SimpleAI): FAILED" -ForegroundColor Red
}

if ($phase2Result -eq 0) {
    Write-Host "✅ Phase 2 (Player Modes): PASSED" -ForegroundColor Green
} else {
    Write-Host "❌ Phase 2 (Player Modes): FAILED" -ForegroundColor Red
}

if ($phase1Result -eq 0 -and $phase2Result -eq 0) {
    Write-Host "`n🎉 All tests passed! Ready for Phase 3" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n⚠️  Some tests failed, please review above output" -ForegroundColor Yellow
    exit 1
}
