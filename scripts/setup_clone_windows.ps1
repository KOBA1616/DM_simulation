<#
Clone distribution setup script (Windows).

Goal:
  - Verify required tools exist (Git, Python, CMake)
  - Create .venv if missing
  - Install dependencies
  - Build native module via CMake
  - Run a minimal smoke test (incl. official GUI stubbing test)

Usage:
  pwsh -File .\scripts\setup_clone_windows.ps1
  pwsh -File .\scripts\setup_clone_windows.ps1 -Toolchain msvc -Configuration Release
  pwsh -File .\scripts\setup_clone_windows.ps1 -SkipBuild
#>

[CmdletBinding()]
param(
  [ValidateSet('msvc','mingw')]
  [string]$Toolchain = 'msvc',

  [ValidateSet('Release','RelWithDebInfo','Debug')]
  [string]$Configuration = 'Release',

  [switch]$SkipBuild,
  [switch]$SkipSmokeTests,
  [switch]$NoInstall,

  # Optional: attempt to install prerequisites (best-effort). Off by default.
  [switch]$InstallCMake,
  [switch]$InstallVSBuildTools,

  # Optional: explicitly pick the base python command used to create venv
  [string]$Python = ''
)

$ErrorActionPreference = 'Stop'

function Info([string]$m) { Write-Host "[info] $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "[warn] $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Error $m; exit 1 }

function Test-HasCommand([string]$name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Enable-MsvcDevEnvironment {
  # Best-effort: load MSVC environment variables into this PowerShell session.
  # This helps CMake/MSBuild find cl/link when VS is installed but env is not configured.
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) {
    return $false
  }

  try {
    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    $installPath = ($installPath | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($installPath)) {
      return $false
    }

    $vsDevCmd = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
      return $false
    }

    Info "Loading MSVC dev environment from: $vsDevCmd"
    $cmd = "\"$vsDevCmd\" -no_logo -arch=x64 -host_arch=x64 && set"
    $lines = & cmd.exe /c $cmd
    foreach ($line in $lines) {
      if ($line -match '^(?<k>[^=]+)=(?<v>.*)$') {
        $env:${matches.k} = $matches.v
      }
    }
    return $true
  } catch {
    return $false
  }
}

function Ensure-BuildToolchain {
  param([string]$toolchain)

  if ($toolchain -eq 'msvc') {
    # If cl.exe isn't available, try to load VS dev env. If that fails, provide clear guidance.
    if (-not (Test-HasCommand 'cl.exe')) {
      $loaded = Enable-MsvcDevEnvironment
      if (-not $loaded -and -not (Test-HasCommand 'cl.exe')) {
        Warn "MSVC build tools not detected (cl.exe not on PATH)."

        if ($InstallVSBuildTools) {
          if (Test-HasCommand 'winget') {
            Info "Attempting to install Visual Studio 2022 Build Tools via winget (best-effort)..."
            # Install Build Tools + VC workload (may require elevation / user interaction depending on policy)
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e --silent --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart"
            Warn "Build Tools installation requested. Re-run this script in a new terminal after install completes."
          } else {
            Warn "winget not found; cannot auto-install Build Tools. Please install manually."
          }
        }

        Fail (@(
          'MSVC toolchain selected but Visual Studio C++ build tools are missing.',
          'Install one of:',
          '  - Visual Studio 2022 Build Tools (recommended) + "Desktop development with C++"',
          '  - Visual Studio 2022 Community + C++ workload',
          'Then re-run: pwsh -File .\\scripts\\setup_clone_windows.ps1'
        ) -join "`n")
      }
    }
  }

  if ($toolchain -eq 'mingw') {
    if (-not (Test-HasCommand 'g++.exe') -and -not (Test-HasCommand 'g++')) {
      Fail (@(
        'MinGW toolchain selected but g++ was not found on PATH.',
        'Install MinGW/MSYS2 and add its bin directory to PATH, or run:',
        '  pwsh -File .\\scripts\\setup_mingw_env.ps1 -GccPath "C:\\Path\\To\\x86_64-w64-mingw32-gcc.exe"'
      ) -join "`n")
    }
  }
}

# Ensure UTF-8 for console output and Python I/O regardless of Windows locale.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
Push-Location $repoRoot
try {
  Info "Repo root: $repoRoot"

  # --- Tool checks ---
  if (-not (Test-HasCommand 'git')) { Fail 'Required tool not found on PATH: git' }
  if (-not (Test-HasCommand 'cmake')) {
    if ($InstallCMake -and (Test-HasCommand 'winget')) {
      Info 'Installing CMake via winget (best-effort)...'
      winget install --id Kitware.CMake -e --silent
      Warn 'CMake install requested. Open a new terminal and re-run this script if cmake is still not found.'
    }
    if (-not (Test-HasCommand 'cmake')) {
      Fail 'Required tool not found on PATH: cmake'
    }
  }

  # Pick base python to create venv
  $basePython = $null
  if (-not [string]::IsNullOrWhiteSpace($Python)) {
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
      Fail "-Python '$Python' not found on PATH."
    }
    $basePython = $Python
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = 'py'
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $basePython = 'python'
  } else {
    Fail "Python not found. Install Python 3.8+ and ensure it is on PATH (or install the Python Launcher 'py')."
  }

  # Validate python version (>= 3.8)
  $verOk = $false
  try {
    if ($basePython -eq 'py') {
      & py -3 -c "import sys; print(sys.version)" | Out-Null
      $verOk = $true
    } else {
      & $basePython -c "import sys; print(sys.version)" | Out-Null
      $verOk = $true
    }
  } catch {
    $verOk = $false
  }
  if (-not $verOk) {
    Fail "Failed to execute Python. Please confirm Python installation works from this terminal."
  }

  # --- Create venv ---
  $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
  if (-not (Test-Path $venvPython)) {
    Info "Creating venv: .venv"
    if ($basePython -eq 'py') {
      & py -3 -m venv .venv
    } else {
      & $basePython -m venv .venv
    }
  } else {
    Info "Reusing existing venv: .venv"
  }

  if (-not (Test-Path $venvPython)) {
    Fail "venv python not found at: $venvPython"
  }

  # --- Install deps ---
  if (-not $NoInstall) {
    Info "Upgrading pip/setuptools/wheel"
    & $venvPython -m pip install --upgrade pip setuptools wheel

    if (Test-Path 'requirements-dev.txt') {
      Info "Installing Python deps: requirements-dev.txt"
      & $venvPython -m pip install -r requirements-dev.txt
    } elseif (Test-Path 'requirements.txt') {
      Info "Installing Python deps: requirements.txt"
      & $venvPython -m pip install -r requirements.txt
    } else {
      Warn "No requirements*.txt found; skipping dependency install"
    }
  } else {
    Info "Skipping dependency install (-NoInstall)"
  }

  # --- Build native module ---
  if (-not $SkipBuild) {
    Ensure-BuildToolchain -toolchain $Toolchain
    Info "Building native components via scripts/build.ps1 (Toolchain=$Toolchain, Configuration=$Configuration)"
    & (Join-Path $scriptDir 'build.ps1') -Toolchain $Toolchain -Config $Configuration
  } else {
    Info "Skipping build (-SkipBuild)"
  }

  # --- Smoke tests ---
  if (-not $SkipSmokeTests) {
    Info "Smoke test: import dm_toolkit + dm_ai_module"
    & $venvPython -c "import dm_toolkit; import dm_ai_module; print('dm_ai_module.IS_NATIVE=', getattr(dm_ai_module,'IS_NATIVE',None))"

    # Official GUI stubbing verification (headless safe)
    $stubRunner = Join-Path $repoRoot 'run_pytest_with_pyqt_stub.py'
    if (Test-Path $stubRunner) {
      Info "Smoke test: GUI stubbing test (headless)"
      & $venvPython $stubRunner -q python/tests/gui/test_gui_stubbing.py
    } else {
      Warn "Stub runner not found at: $stubRunner (skipping GUI stub test)"
    }
  } else {
    Info "Skipping smoke tests (-SkipSmokeTests)"
  }

  Info "Setup completed successfully."
  Info "Next: .\\scripts\\run_gui.ps1 (or run CI-local: .\\scripts\\run_ci_local.ps1)"
}
finally {
  Pop-Location
}
