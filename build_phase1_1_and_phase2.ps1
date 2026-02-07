# Phase 1.1 + Phase 2 統合ビルドスクリプト
Write-Host "=== Phase 1.1 + Phase 2 統合ビルド ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Cleanup
Write-Host "[1/4] クリーンアップ..." -ForegroundColor Yellow
Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Remove-Item -Force dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Force bin\Release\dm_ai_module*.pyd -ErrorAction SilentlyContinue
Write-Host "  ✓ 古いバイナリ削除完了" -ForegroundColor Green

# Step 2: Touch source files
Write-Host ""
Write-Host "[2/4] ソースファイルのタイムスタンプ更新..." -ForegroundColor Yellow
$files = @(
    "src\bindings\bind_core.cpp",
    "src\core\game_state.cpp",
    "src\core\game_state.hpp",
    "src\engine\ai\simple_ai.cpp",
    "src\engine\ai\simple_ai.hpp",
    "src\engine\game_instance.cpp"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        (Get-Item $file).LastWriteTime = Get-Date
        Write-Host "  ✓ Touched: $file" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠ Not found: $file" -ForegroundColor Yellow
    }
}

# Step 3: Build
Write-Host ""
Write-Host "[3/4] ビルド実行..." -ForegroundColor Yellow

if (-not (Test-Path build-msvc)) {
    Write-Host "  CMake初期設定中..." -ForegroundColor Gray
    cmake -B build-msvc -G "Visual Studio 17 2022" -A x64 2>&1 | Out-Null
}

$buildStart = Get-Date
cmake --build build-msvc --config Release --target dm_ai_module 2>&1 | Out-Null
$buildEnd = Get-Date
$buildTime = ($buildEnd - $buildStart).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ ビルド成功 ($([math]::Round($buildTime, 1))秒)" -ForegroundColor Green
    
    # Check output
    if (Test-Path bin\Release\dm_ai_module*.pyd) {
        $pydFile = Get-Item bin\Release\dm_ai_module*.pyd
        Write-Host "  ✓ $($pydFile.Name) 生成完了 ($([math]::Round($pydFile.Length/1MB, 2)) MB)" -ForegroundColor Gray
    } else {
        Write-Host "  ❌ pydファイルが見つかりません" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ❌ ビルド失敗" -ForegroundColor Red
    Write-Host ""
    Write-Host "エラーログを確認してください：" -ForegroundColor Yellow
    cmake --build build-msvc --config Release --target dm_ai_module 2>&1 | Select-String "error" | Select-Object -First 10
    exit 1
}

# Step 4: Test
Write-Host ""
Write-Host "[4/4] 機能テスト..." -ForegroundColor Yellow

$testResult = python -c @"
import sys
sys.path.insert(0, '.')
import dm_ai_module

# Test 1: is_human_player method
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()
try:
    result = gs.is_human_player(0)
    print('✓ is_human_player() メソッド使用可能')
except AttributeError as e:
    print(f'✗ is_human_player() エラー: {e}')
    sys.exit(1)

# Test 2: SimpleAI with GameState parameter
deck = list(range(1,11))*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)

actions = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
if len(actions) > 0:
    ai = dm_ai_module.SimpleAI()
    try:
        idx = ai.select_action(actions, gs)
        print(f'✓ SimpleAI.select_action(actions, gs) 動作確認')
    except Exception as e:
        print(f'✗ SimpleAI エラー: {e}')
        sys.exit(1)

print('✓ すべての機能テスト合格')
"@ 2>&1

Write-Host $testResult
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ テスト失敗" -ForegroundColor Red
    exit 1
}

# Success
Write-Host ""
Write-Host "=== ビルド & テスト完了 ===" -ForegroundColor Green
Write-Host ""
Write-Host "変更内容:" -ForegroundColor White
Write-Host "  Phase 1.1: SimpleAIのフェーズ対応化" -ForegroundColor Gray
Write-Host "  Phase 2:   PlayerMode C++化 + is_human_player() バインディング" -ForegroundColor Gray
Write-Host ""
Write-Host "次のステップ: .\scripts\run_gui.ps1 でGUI起動" -ForegroundColor Cyan
