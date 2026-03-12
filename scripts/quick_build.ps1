#!/usr/bin/env pwsh
# quick_build.ps1 - インクリメンタルビルド（Ninja デフォルト）
# 用途: 変更ファイルのみ再コンパイルする高速ビルド
# 使用例:
#   .\scripts\quick_build.ps1              # Release (Ninja 優先)
#   .\scripts\quick_build.ps1 -Config Debug
#   .\scripts\quick_build.ps1 -Toolchain mingw
#   .\scripts\quick_build.ps1 -Generator "Visual Studio 17 2022"  # VS 強制

param(
    [ValidateSet('msvc','mingw')]
    [string]$Toolchain = 'msvc',
    [string]$Config = "Release",
    [string]$Generator = "",
    [switch]$SkipAutoCleanup
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\build.ps1" -Toolchain $Toolchain -Config $Config -Generator $Generator -SkipAutoCleanup:$SkipAutoCleanup

# ビルド成果物が存在するか確認し、ログを保存する
$pydPath = Join-Path $scriptDir "..\bin\dm_ai_module.cp312-win_amd64.pyd"
$logDir = Join-Path $scriptDir "..\reports\build"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "build_latest.txt"
if (Test-Path $pydPath) {
    $ts = (Get-Item $pydPath).LastWriteTime
    $msg = "[BUILD OK] $ts"
    Write-Host $msg
    $msg | Out-File -FilePath $logFile -Encoding utf8

    # Archive old root-level build logs into timestamped folder for traceability
    $archiveRoot = Join-Path $scriptDir "..\archive\build_logs"
    New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null
    $tsString = Get-Date -Format "yyyyMMdd_HHmmss"
    $thisArchive = Join-Path $archiveRoot $tsString
    New-Item -ItemType Directory -Force -Path $thisArchive | Out-Null

    $rootFiles = @("..\build_out.txt", "..\build_out2.txt", "..\build_summary.txt")
    foreach ($f in $rootFiles) {
        $full = Join-Path $scriptDir $f
        if (Test-Path $full) {
            try {
                Move-Item -Path $full -Destination $thisArchive -Force
            } catch {
                # If move fails, copy as fallback
                Copy-Item -Path $full -Destination $thisArchive -Force
            }
        }
    }

    exit 0
} else {
    $msg = "[BUILD FAIL] PYD not found"
    Write-Error $msg
    $msg | Out-File -FilePath $logFile -Encoding utf8
    exit 1
}
