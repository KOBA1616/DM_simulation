# Phase 1.1 Build and Test Script
Write-Host "=== Phase 1.1: フェーズ対応SimpleAI ビルド & テスト ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean build
Write-Host "[1/4] Pythonプロセス停止..." -ForegroundColor Yellow
Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "[2/4] 古いバイナリ削除..." -ForegroundColor Yellow
Remove-Item -Force dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Force bin\Release\dm_ai_module*.pyd -ErrorAction SilentlyContinue

# Step 2: Build
Write-Host "[3/4] ビルド実行中..." -ForegroundColor Yellow
if (-not (Test-Path build-msvc)) {
    cmake -B build-msvc -G "Visual Studio 17 2022" -A x64
}

$buildOutput = cmake --build build-msvc --config Release --target dm_ai_module 2>&1
$buildSuccess = $LASTEXITCODE -eq 0

if ($buildSuccess) {
    Write-Host "✅ ビルド成功" -ForegroundColor Green
} else {
    Write-Host "❌ ビルド失敗" -ForegroundColor Red
    Write-Host "エラー詳細:" -ForegroundColor Red
    $buildOutput | Select-String "error" | Select-Object -Last 10
    exit 1
}

# Step 3: Test Phase 1.1
Write-Host "[4/4] Phase 1.1 テスト実行..." -ForegroundColor Yellow
Write-Host ""

$testCode = @"
import sys
sys.path.insert(0, '.')
import dm_ai_module

print('=== Phase 1.1 テスト: フェーズ対応優先度 ===')
print()

# Setup
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()
deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Start game
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)

print(f'現在のフェーズ: {gs.current_phase}')
print()

# Test MANA phase
if gs.current_phase == 2:  # Phase.MANA
    print('✅ MANAフェーズでテスト開始')
    actions = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
    print(f'  利用可能なアクション: {len(actions)}')
    
    for i, a in enumerate(actions[:5]):
        print(f'    #{i}: type={a.type} card_id={a.card_id}')
    
    # AI selection test
    ai = dm_ai_module.SimpleAI()
    idx = ai.select_action(actions, gs)
    if idx is not None:
        selected = actions[idx]
        print(f'  AIが選択: #{idx} type={selected.type} card_id={selected.card_id}')
        if selected.type == 3:  # PlayerIntent::MANA_CHARGE
            print('  ✅ MANAフェーズで正しくMANA_CHARGEを選択')
        else:
            print(f'  ❌ 予期しないアクション（type={selected.type}）')

# Advance to MAIN phase
gs.turn_stats.mana_charged_by_player[gs.active_player_id] = True
dm_ai_module.PhaseManager.advance_phase(gs, card_db)
print()
print(f'フェーズ進行後: {gs.current_phase}')

if gs.current_phase == 3:  # Phase.MAIN
    print('✅ MAINフェーズでテスト')
    actions = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
    print(f'  利用可能なアクション: {len(actions)}')
    
    for i, a in enumerate(actions[:5]):
        print(f'    #{i}: type={a.type} card_id={a.card_id}')
    
    ai = dm_ai_module.SimpleAI()
    idx = ai.select_action(actions, gs)
    if idx is not None:
        selected = actions[idx]
        print(f'  AIが選択: #{idx} type={selected.type}')
        if selected.type == 8:  # PlayerIntent::DECLARE_PLAY
            print('  ✅ MAINフェーズで正しくDECLARE_PLAYを優先')
        elif selected.type == 0:  # PlayerIntent::PASS
            print('  ⚠️  プレイ可能なカードがなくPASSを選択（正常）')

print()
print('=== Phase 1.1 テスト完了 ===')
"@

python -c $testCode 2>&1

# Summary
Write-Host ""
Write-Host "=== ビルド & テスト完了 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Phase 1.1の変更点:" -ForegroundColor White
Write-Host "  • SimpleAI::get_priority()がGameStateを受け取るように変更" -ForegroundColor Gray
Write-Host "  • MANAフェーズ: MANA_CHARGE優先度 40→90" -ForegroundColor Gray
Write-Host "  • ATTACKフェーズ: ATTACK優先度 60→85" -ForegroundColor Gray
Write-Host "  • BLOCKフェーズ: BLOCK優先度 85（DECLARE_BLOCKER削除）" -ForegroundColor Gray
Write-Host "  • 未定義アクション参照を修正" -ForegroundColor Gray
Write-Host ""
Write-Host "詳細: PHASE_AWARE_AI_DESIGN.md を参照" -ForegroundColor Cyan
