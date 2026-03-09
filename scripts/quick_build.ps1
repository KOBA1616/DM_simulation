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
