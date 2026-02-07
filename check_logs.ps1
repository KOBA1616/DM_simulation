# デバッグログ確認スクリプト

Write-Host "=== pending_effects_debug.txt ===" -ForegroundColor Cyan
if (Test-Path logs\pending_effects_debug.txt) {
    Get-Content logs\pending_effects_debug.txt
} else {
    Write-Host "ファイルなし（まだpending effectsが処理されていません）" -ForegroundColor Yellow
}

Write-Host "`n=== active_indices_debug.txt ===" -ForegroundColor Cyan
if (Test-Path logs\active_indices_debug.txt) {
    Get-Content logs\active_indices_debug.txt
} else {
    Write-Host "ファイルなし（まだアクション生成されていません）" -ForegroundColor Yellow
}

Write-Host "`n=== trigger_ability_debug.txt ===" -ForegroundColor Cyan
if (Test-Path logs\trigger_ability_debug.txt) {
    Get-Content logs\trigger_ability_debug.txt
} else {
    Write-Host "ファイルなし（まだTRIGGER_ABILITYが処理されていません）" -ForegroundColor Yellow
}

Write-Host "`n=== logsディレクトリの全ファイル ===" -ForegroundColor Cyan
Get-ChildItem logs -ErrorAction SilentlyContinue | Select-Object Name, LastWriteTime, Length
