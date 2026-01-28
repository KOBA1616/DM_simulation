param(
    [Parameter(Position = 0)]
    [string]$Config = "Release",
    [ValidateSet('msvc','mingw')]
    [string]$Toolchain = 'msvc',
    [switch]$Build,
    [switch]$AllowFallback
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDirName = if ($Toolchain -eq 'mingw') { 'build-mingw' } else { 'build-msvc' }
$buildDir = Join-Path $projectRoot $buildDirName

# Ensure Python output is UTF-8 regardless of Windows locale.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Prefer venv python if available.
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if ($Build) {
    Write-Host "Building project before launching GUI..."
    & "$scriptDir/build.ps1" -Config $Config -Toolchain $Toolchain
}

if (-not (Test-Path $buildDir)) {
    Write-Host "Build directory not found at $buildDir. Attempting automatic build..."
    try {
        & "$scriptDir/build.ps1" -Config $Config -Toolchain $Toolchain
    } catch {
        Write-Error "Automatic build failed: $_"
    }
    if (-not (Test-Path $buildDir)) {
        if (-not $AllowFallback) {
            Write-Error "Build directory still missing after automatic build. Aborting to avoid running with Python fallback."
            exit 1
        } else {
            Write-Warning "Build directory still missing; proceeding with Python fallback as requested by -AllowFallback."
        }
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

# Prefer native build artefact: search build dir for dm_ai_module*.pyd and set override
try {
    # Search preferred locations for native artefact. Include build dir and common bin/Release.
    $searchDirs = @($buildDir, (Join-Path $projectRoot 'bin\Release')) | Where-Object { $_ -and (Test-Path $_) }
    $pyd = $null
    foreach ($dir in $searchDirs) {
        $pyd = Get-ChildItem -Path $dir -Filter "dm_ai_module*.pyd" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($pyd) { break }
    }
    if ($pyd) {
        $full = $pyd.FullName
        Write-Host "Found native dm_ai_module at $full -- forcing loader override."
        $env:DM_AI_MODULE_NATIVE = $full
        # Ensure directory containing pyd is on PYTHONPATH first; avoid empty entries and duplicates
        $pydDir = Split-Path -Parent $full
        $current = @()
        if ($env:PYTHONPATH) { $current = $env:PYTHONPATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } }
        if ($current -notcontains $pydDir) {
            $env:PYTHONPATH = $pydDir + ';' + ($current -join ';')
        }
    }
} catch {
    # non-fatal
}

# Centralized logging defaults for GUI runs. These can be overridden by the user's environment.
# - DM_CONSOLE_LOG_LEVEL: console verbosity (INFO|WARNING|ERROR)
# - DM_FILE_LOG_LEVEL: file handler verbosity (DEBUG|INFO|...)
# - DM_ROOT_LOG_LEVEL: root logger level
if (-not $env:DM_CONSOLE_LOG_LEVEL) { $env:DM_CONSOLE_LOG_LEVEL = 'WARNING' }
if (-not $env:DM_FILE_LOG_LEVEL)    { $env:DM_FILE_LOG_LEVEL = 'DEBUG' }
if (-not $env:DM_ROOT_LOG_LEVEL)    { $env:DM_ROOT_LOG_LEVEL = 'DEBUG' }
if (-not $env:DM_SILENT_LOGGERS)    { $env:DM_SILENT_LOGGERS = 'EngineCompat,dm_ai_module,dm_toolkit' }

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


Write-Host "Starting GUI..."
& $pythonExe "$projectRoot/dm_toolkit/gui/app.py"
