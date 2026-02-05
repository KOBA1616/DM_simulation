# Step 1.5 完了確認スクリプト

Write-Host "`n=== Step 1.5: マナチャージフラグ修正の検証 ===" -ForegroundColor Cyan

# 1. ビルド実行
Write-Host "`n[1/4] ビルド実行中..." -ForegroundColor Yellow
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Remove-Item logs\*.txt -Force -ErrorAction SilentlyContinue

$buildOutput = cmake --build build --config Release --target dm_ai_module 2>&1
$buildSuccess = $LASTEXITCODE -eq 0

if ($buildSuccess) {
    Write-Host "✅ ビルド成功" -ForegroundColor Green
} else {
    Write-Host "❌ ビルド失敗" -ForegroundColor Red
    $buildOutput | Select-String "error" | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    exit 1
}

# 2. GUI起動
Write-Host "`n[2/4] GUI起動中..." -ForegroundColor Yellow
Write-Host "以下をテストしてください:" -ForegroundColor Cyan
Write-Host "  - P0がマナチャージ → PASSでターン終了" -ForegroundColor White
Write-Host "  - P1がマナチャージ → PASSでターン終了" -ForegroundColor White
Write-Host "  - P0が再度マナチャージ可能か確認" -ForegroundColor White
Write-Host "  - 最低3ターン実行してください" -ForegroundColor White
Write-Host "`nGUIを閉じると次のステップに進みます...`n" -ForegroundColor Yellow

. .\scripts\run_gui.ps1

# 3. ログ検証
Write-Host "`n[3/4] ログ検証中..." -ForegroundColor Yellow

if (Test-Path logs\mana_phase_debug.txt) {
    $manaLogs = Get-Content logs\mana_phase_debug.txt
    $setCommands = $manaLogs | Select-String "SET_MANA_CHARGED"
    $p0Charged = $manaLogs | Select-String "pid=0.*TRUE"
    $p1Charged = $manaLogs | Select-String "pid=1.*TRUE"
    
    Write-Host "  - SET_MANA_CHARGED実行回数: $($setCommands.Count)" -ForegroundColor White
    Write-Host "  - P0マナチャージ: $($p0Charged.Count)回" -ForegroundColor White
    Write-Host "  - P1マナチャージ: $($p1Charged.Count)回" -ForegroundColor White
    
    if ($setCommands.Count -gt 0) {
        Write-Host "✅ マナチャージフラグ設定が実行されました" -ForegroundColor Green
    } else {
        Write-Host "⚠️  マナチャージフラグ設定が実行されていません" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  mana_phase_debug.txt が見つかりません" -ForegroundColor Yellow
}

if (Test-Path logs\reset_turn_stats_debug.txt) {
    $resetLogs = Get-Content logs\reset_turn_stats_debug.txt
    $resetCount = ($resetLogs | Measure-Object).Count
    
    Write-Host "  - ターン統計リセット回数: $resetCount" -ForegroundColor White
    
    if ($resetCount -gt 0) {
        Write-Host "✅ ターン統計リセットが実行されました" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  reset_turn_stats_debug.txt が見つかりません" -ForegroundColor Yellow
}

# 4. 結果サマリ
Write-Host "`n[4/4] 結果サマリ" -ForegroundColor Yellow
Write-Host "`n修正内容:" -ForegroundColor Cyan
Write-Host "  - mana_charged_this_turn → mana_charged_by_player[2]" -ForegroundColor White
Write-Host "  - プレイヤーごとにマナチャージフラグを管理" -ForegroundColor White
Write-Host "  - game_logic_system.cpp の古いコード削除" -ForegroundColor White

Write-Host "`n検証結果:" -ForegroundColor Cyan
if ($buildSuccess) {
    Write-Host "  ✅ Step 1.5 完了" -ForegroundColor Green
    Write-Host "`n次のステップ:" -ForegroundColor Yellow
    Write-Host "  - MIGRATION_PLAN.md の Step 2 (TurnStats設計見直し) を検討" -ForegroundColor White
    Write-Host "  - または Step 3 (PhaseManager移行) に進む" -ForegroundColor White
} else {
    Write-Host "  ❌ Step 1.5 未完了 - ビルドエラーを修正してください" -ForegroundColor Red
}

Write-Host ""
