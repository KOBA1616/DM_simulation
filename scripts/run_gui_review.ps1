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
  # Attempt native model load at startup to avoid runtime Python fallback.
  try {
    $modelPath = $env:DM_NATIVE_ONNX_PATH
    if (-not $modelPath) {
      $cand = Get-ChildItem -Path (Join-Path $repoRoot 'models') -Filter '*.onnx' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($cand) { $modelPath = $cand.FullName }
    }
    if ($modelPath) {
      Write-Host "Attempting native ONNX load: $modelPath"
      $pyScript = @"
import sys,traceback
try:
    import dm_ai_module
    if not hasattr(dm_ai_module, 'native_load_onnx'):
        print('native_load_onnx not available')
        sys.exit(3)
    ok = dm_ai_module.native_load_onnx(r'''$modelPath''')
    if not ok:
        print('native_load_onnx returned False')
        sys.exit(2)
    print('native_load_onnx success')
    sys.exit(0)
except Exception as e:
    print('native_load_onnx exception:', e)
    traceback.print_exc()
    sys.exit(4)
"@

      $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ([System.Guid]::NewGuid().ToString() + '.py'))
      Set-Content -Path $tmp -Value $pyScript -Encoding UTF8
      & $pythonExe $tmp
      $code = $LASTEXITCODE
      Remove-Item $tmp -ErrorAction SilentlyContinue
      if ($code -ne 0) {
        Write-Warning "Native model load returned code $code; continuing with Python fallback for review."
      } else {
        Write-Host "Native model loaded successfully."
      }
    }
  } catch {
    Write-Warning "Exception while attempting native model load: $_"
  }

  & $pythonExe -m dm_toolkit.gui.editor.window $CardsJson
}
finally {
  Pop-Location
}
