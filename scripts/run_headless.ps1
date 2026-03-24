# scripts/run_headless.ps1
# DM Engine ヘッドレスモード実行スクリプト
#
# 使い方:
#   .\scripts\run_headless.ps1                              # ai-vs-ai 1試合
#   .\scripts\run_headless.ps1 -Mode batch -Games 100       # バッチ 100試合
#   .\scripts\run_headless.ps1 -Mode human-vs-ai             # 人間 vs AI
#   .\scripts\run_headless.ps1 -Mode ai-vs-ai -Output out.json
#
# 再発防止: PyQt6 が不要な環境（CI・Docker 等）でもそのまま動作する。
param(
    [string]$Mode   = "ai-vs-ai",
    [int]   $Games  = 1,
    [string]$Output = "",
    [switch]$Verbose
)

$env:PYTHONUTF8  = "1"
$env:PYTHONPATH  = "."

# 仮想環境または既定 python を選択
$PythonExe = if (Test-Path ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} elseif (Test-Path ".\.venv\bin\python") {
    ".\.venv\bin\python"
} else {
    "python"
}

$cmdArgs = @("tools\run_headless.py", "--mode", $Mode)
if ($Games -gt 1)  { $cmdArgs += @("--games", "$Games") }
if ($Output)       { $cmdArgs += @("--output", $Output) }
if ($Verbose)      { $cmdArgs += "--verbose" }

Write-Host ">> DM Engine Headless  mode=$Mode  games=$Games" -ForegroundColor Cyan
& $PythonExe @cmdArgs
