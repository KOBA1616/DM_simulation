#!/usr/bin/env pwsh
# rebuild_clean.ps1 - クリーンビルド（ビルドディレクトリを削除して再構成）
# 用途: CMakeLists.txt 変更時や依存関係の問題解決時に使用
# 使用例:
#   .\scripts\rebuild_clean.ps1
#   .\scripts\rebuild_clean.ps1 -Config Debug
#   .\scripts\rebuild_clean.ps1 -Toolchain mingw

param(
    [ValidateSet('msvc','mingw')]
    [string]$Toolchain = 'msvc',
    [string]$Config = "Release",
    [string]$Generator = "",
    [switch]$SkipAutoCleanup
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\build.ps1" -Toolchain $Toolchain -Config $Config -Generator $Generator -Clean -SkipAutoCleanup:$SkipAutoCleanup
