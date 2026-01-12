<#
GUI review launcher (Windows).

Purpose:
  - Launch the Card Editor quickly for UI review.
  - Does NOT install dependencies. Run setup_run_gui_review_windows.ps1 once first.

Usage:
  pwsh -File .\scripts\run_gui_review.ps1
  pwsh -File .\scripts\run_gui_review.ps1 -CardsJson data\cards.json
#>

[CmdletBinding()]
param(
  [string]$CardsJson = 'data\cards.json'
)

$ErrorActionPreference = 'Stop'

function Info([string]$m) { Write-Host "[info] $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Error $m; exit 1 }

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path

$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
  Fail "venv python not found: $pythonExe`nRun: pwsh -File .\\scripts\\setup_run_gui_review_windows.ps1`n(or: pwsh -File .\\scripts\\setup_gui_review_windows.ps1 to setup+launch)"
}

Push-Location $repoRoot
try {
  Info "Launching Card Editor for GUI review"
  & $pythonExe -m dm_toolkit.gui.editor.window $CardsJson
}
finally {
  Pop-Location
}
