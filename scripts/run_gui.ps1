param(
    [Parameter(Position = 0)]
    [string]$Config = "Release",
    [ValidateSet('msvc', 'mingw')]
    [string]$Toolchain = 'msvc',
    [switch]$Build,
    [switch]$AllowFallback,
    [switch]$InstallPyQt,
    [switch]$Review
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDirName = if ($Toolchain -eq 'mingw') { 'build-mingw' } else { 'build-msvc' }
$buildDir = Join-Path $projectRoot $buildDirName

# If the expected build directory doesn't exist, try common alternatives
if (-not (Test-Path $buildDir)) {
    $altNames = @('build-ninja', 'build', 'build-msvc', 'build-mingw')
    foreach ($name in $altNames) {
        $candidate = Join-Path $projectRoot $name
        if (Test-Path $candidate) {
            Write-Host "Preferred build directory '$buildDir' not found; using detected directory: $candidate"
            $buildDir = $candidate
            break
        }
    }
}
function Get-ProjectPythonExe {
    param([string]$root)
    $venv = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }
    $sys = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($sys) { return $sys }
    return $null
}

function Test-NativeJsonLoader {
    param(
        [Parameter(Mandatory = $true)]
        [string]$pythonExe,
        [Parameter(Mandatory = $true)]
        [string]$projectRoot,
        [string]$nativeModulePath = ''
    )
    $prevOverride = $env:DM_AI_MODULE_NATIVE
    if ([string]::IsNullOrWhiteSpace($nativeModulePath)) {
        Remove-Item Env:DM_AI_MODULE_NATIVE -ErrorAction SilentlyContinue
    } else {
        $env:DM_AI_MODULE_NATIVE = $nativeModulePath
    }
    $probe = @"
import os
import dm_ai_module as dm
path = os.path.join(r'$projectRoot', 'data', 'cards.json')
db = dm.JsonLoader.load_cards(path)
raise SystemExit(0 if db else 2)
"@
    try {
        & $pythonExe -c $probe *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        if ([string]::IsNullOrWhiteSpace($prevOverride)) {
            Remove-Item Env:DM_AI_MODULE_NATIVE -ErrorAction SilentlyContinue
        } else {
            $env:DM_AI_MODULE_NATIVE = $prevOverride
        }
    }
}

function Get-NativePydCandidates {
    param(
        [Parameter(Mandatory = $true)]
        [string]$projectRoot,
        [Parameter(Mandatory = $true)]
        [string]$buildDir
    )
    $list = New-Object System.Collections.Generic.List[string]

    # 再発防止: 前回実行で残った DM_AI_MODULE_NATIVE が壊れた .pyd を指すと
    # 毎回 native 判定に失敗するため、ユーザー指定値も候補として再検証する。
    if (-not [string]::IsNullOrWhiteSpace($env:DM_AI_MODULE_NATIVE) -and (Test-Path $env:DM_AI_MODULE_NATIVE)) {
        $list.Add((Resolve-Path $env:DM_AI_MODULE_NATIVE).Path)
    }

    $searchDirs = @(
        (Join-Path $projectRoot 'bin'),
        (Join-Path $projectRoot 'bin\Release'),
        $buildDir,
        $projectRoot
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($dir in $searchDirs) {
        Get-ChildItem -Path $dir -Filter 'dm_ai_module*.pyd' -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { $list.Add($_.FullName) }
    }

    $seen = @{}
    $ordered = New-Object System.Collections.Generic.List[string]
    foreach ($p in $list) {
        $k = $p.ToLowerInvariant()
        if (-not $seen.ContainsKey($k)) {
            $seen[$k] = $true
            $ordered.Add($p)
        }
    }
    return $ordered
}

# Ensure Python output is UTF-8 regardless of Windows locale.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Prefer venv python if available.
$pythonExe = Get-ProjectPythonExe -root $projectRoot
if (-not $pythonExe) {
    Write-Error "Python executable not found. Activate a venv or ensure 'python' is on PATH."
    exit 1
}

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# GUI起動時にビルドを行う理由:
# - GUI はネイティブの C++ 拡張（dm_ai_module など）やプラットフォーム固有の DLL/.pyd
#   を読み込んで高速なAI処理や ONNX 推論、ネイティブリソースへのアクセスを行います。
# - これらのネイティブアーティファクトはビルド工程で生成されるため、開発環境でGUIを
#   正常に動作させるには事前にビルドしておくことが推奨されます。
# - ネイティブが無い状態では Python フォールバックが使われますが、機能制限や
#   AttributeError／不定動作（例: ヒープ破壊）が発生する可能性があるため、自動的な
#   ビルド実行オプション（-Build）を用意しています。フォールバックが必要なら
#   -AllowFallback を明示的に指定してください。
if ($Build) {
    Write-Host "Building project before launching GUI..."
    & "$scriptDir/build.ps1" -Config $Config -Toolchain $Toolchain
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed (build.ps1 returned $LASTEXITCODE). Aborting."
        exit $LASTEXITCODE
    }
}

# Optionally ensure PyQt6 is installed (installs into selected Python; venv preferred)
if ($InstallPyQt) {
    Write-Host "Installing PyQt6 into: $pythonExe"
    & $pythonExe -m pip install --upgrade pip setuptools wheel
    & $pythonExe -m pip install PyQt6
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyQt6 into $pythonExe. Aborting."
        exit 1
    }
}

# Review mode: launch the GUI review helper if requested
if ($Review) {
    $reviewScript = Join-Path $scriptDir 'run_gui_review.ps1'
    if (Test-Path $reviewScript) {
        Write-Host "Launching GUI review mode via run_gui_review.ps1"
        & $reviewScript
        exit $LASTEXITCODE
    } else {
        Write-Warning "run_gui_review.ps1 not found; falling back to standard GUI launcher."
    }
}

if (-not (Test-Path $buildDir)) {
    Write-Error "Build directory not found at $buildDir. This script no longer attempts automatic builds."
    Write-Host "Run the build script to create native artifacts: $scriptDir\build.ps1 -Config $Config -Toolchain $Toolchain"
    if (-not $AllowFallback) {
        Write-Error "Aborting to avoid running with Python fallback. Use -AllowFallback to override."
        exit 1
    } else {
        Write-Warning "Build directory missing; proceeding with Python fallback as requested by -AllowFallback."
    }
}

# Optional: ensure MinGW DLLs are present (only when explicitly configured)
$mingwBin = $env:DM_MINGW_BIN
if (-not [string]::IsNullOrWhiteSpace($mingwBin) -and (Test-Path $mingwBin)) {
    Write-Host "Copying MinGW DLLs from '$mingwBin' to build directory..."
    Get-ChildItem "$mingwBin\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        $dest = Join-Path $buildDir $_.Name
        if (-not (Test-Path $dest)) {
            Copy-Item $_.FullName -Destination $buildDir
        }
    }
}

$existing = @()
if ($env:PYTHONPATH) {
    $existing = $env:PYTHONPATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}
$env:PYTHONPATH = $projectRoot
if ($existing.Count -gt 0) {
    $env:PYTHONPATH = $env:PYTHONPATH + ';' + ($existing -join ';')
}

# 再発防止: native 不健全時の自動劣化運転を避けるため、
# -AllowFallback が無い限り起動を失敗させて根本修正を優先する。
if (-not $env:DM_DISABLE_NATIVE) {
    $selectedNative = $null
    try {
        $candidates = Get-NativePydCandidates -projectRoot $projectRoot -buildDir $buildDir
        foreach ($candidate in $candidates) {
            if (Test-NativeJsonLoader -pythonExe $pythonExe -projectRoot $projectRoot -nativeModulePath $candidate) {
                $selectedNative = $candidate
                break
            }
        }
    } catch {
        $selectedNative = $null
    }

    if ($selectedNative) {
        $env:DM_AI_MODULE_NATIVE = $selectedNative
        Remove-Item Env:DM_DISABLE_NATIVE -ErrorAction SilentlyContinue
        Write-Host "Using healthy native dm_ai_module: $selectedNative"
    } else {
        Remove-Item Env:DM_AI_MODULE_NATIVE -ErrorAction SilentlyContinue
        if ($AllowFallback) {
            $env:DM_DISABLE_NATIVE = '1'
            Write-Warning "No healthy native dm_ai_module found; switching to Python fallback (DM_DISABLE_NATIVE=1)."
        } else {
            # 再発防止: 自動フォールバックで不具合を隠さず、native 健全性の根本修正を優先する。
            Write-Error "No healthy native dm_ai_module found. Fix native build/loader first, or use -AllowFallback explicitly."
            exit 1
        }
    }
}

if ($env:DM_DISABLE_NATIVE -eq '1') {
    Write-Warning "Running GUI with Python fallback mode (native module disabled)."
} else {
    Write-Host "Using native dm_ai_module (preflight passed)"
}

# Centralized logging defaults for GUI runs. These can be overridden by the user's environment.
# - DM_CONSOLE_LOG_LEVEL: console verbosity (INFO|WARNING|ERROR)
# - DM_FILE_LOG_LEVEL: file handler verbosity (DEBUG|INFO|...)
# - DM_ROOT_LOG_LEVEL: root logger level
if (-not $env:DM_CONSOLE_LOG_LEVEL) { $env:DM_CONSOLE_LOG_LEVEL = 'WARNING' }
if (-not $env:DM_FILE_LOG_LEVEL) { $env:DM_FILE_LOG_LEVEL = 'DEBUG' }
if (-not $env:DM_ROOT_LOG_LEVEL) { $env:DM_ROOT_LOG_LEVEL = 'DEBUG' }
if (-not $env:DM_SILENT_LOGGERS) { $env:DM_SILENT_LOGGERS = 'EngineCompat,dm_ai_module,dm_toolkit' }

# Optional logging tuning defaults (can be overridden by the user)
# - DM_LOG_RATE_LIMIT_SECONDS: per-console-message rate-limit (0 disables)
# - DM_LOG_MAX_BYTES: rotating file max size (bytes)
# - DM_LOG_BACKUP_COUNT: rotating file backup count
# - DM_LOG_FORMAT: python logging Formatter format string
if (-not $env:DM_LOG_RATE_LIMIT_SECONDS) { $env:DM_LOG_RATE_LIMIT_SECONDS = '0.5' }
if (-not $env:DM_LOG_MAX_BYTES) { $env:DM_LOG_MAX_BYTES = '10485760' }
if (-not $env:DM_LOG_BACKUP_COUNT) { $env:DM_LOG_BACKUP_COUNT = '5' }
if (-not $env:DM_LOG_FORMAT) { $env:DM_LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s' }
if (-not $env:DM_LOG_CONSOLE_STDERR) { $env:DM_LOG_CONSOLE_STDERR = '1' }
# Optional: per-logger override example, format: Name=LEVEL,Name2=LEVEL
# Default sensible per-logger overrides for noisy subsystems
if (-not $env:DM_LOGGER_LEVELS) { $env:DM_LOGGER_LEVELS = 'dm_ai_module=WARNING,EngineCompat=WARNING,dm_toolkit=WARNING' }

# AI Configuration
# - DM_SELECT_NUMBER_MODE: How AI chooses for SELECT_NUMBER effects (draw cards, etc.)
#   Options: 'heuristic' (fast, game state analysis), 'evaluator' (HeuristicEvaluator),
#            'mcts' (simulation-based, slower but better)
if (-not $env:DM_SELECT_NUMBER_MODE) { $env:DM_SELECT_NUMBER_MODE = 'evaluator' }


Write-Host "Starting GUI..."
& $pythonExe -m dm_toolkit.gui.app
