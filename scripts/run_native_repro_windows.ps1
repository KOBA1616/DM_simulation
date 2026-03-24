param(
    [switch]$Build,
    [int]$RunCount = 100,
    [string]$LogDir = "logs/native_repro",
    [switch]$CollectEventLog,
    [string]$ProcDumpPath = "C:\\Tools\\procdump.exe",
    [switch]$DryRun
)

Write-Host "[run_native_repro_windows] Build=$Build RunCount=$RunCount LogDir=$LogDir CollectEventLog=$CollectEventLog ProcDumpPath=$ProcDumpPath DryRun=$DryRun"

function Ensure-Dir($p) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

# Basic checks
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Warning "cmake not found in PATH. Native build will fail unless CMake is available."
}
if (-not (Get-Command msbuild -ErrorAction SilentlyContinue) -and -not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    Write-Warning "MSVC toolchain not detected (msbuild/cl). Ensure Visual Studio build tools are installed." 
}

Ensure-Dir $LogDir

$buildDir = "build"
$nativeOut = "bin"

if ($DryRun) {
    Write-Host "[DryRun] Would perform (sketch):"
    if ($Build) {
        Write-Host "  - mkdir $buildDir"
        Write-Host "  - cd $buildDir; cmake -S .. -B . -G 'Ninja' -DCMAKE_BUILD_TYPE=Release"
        Write-Host "  - cmake --build . --config Release --target dm_ai_module"
        Write-Host "  - copy <built pyd/dll> to root for pytest"
    }
    Write-Host "  - Run: python -m pip install -r requirements.txt (if needed)"
    Write-Host "  - Run: .venv\Scripts\Activate.ps1; python scripts/repeat_native_loader.py --count $RunCount --log-dir $LogDir"
    if ($CollectEventLog) { Write-Host "  - Collect Windows Event Log (wevtutil epl Application $LogDir\Application.evtx)" }
    Write-Host "  - If proc dump desired: $ProcDumpPath -accepteula -ma -e 1 -f \"python.exe\" -- python -m pytest ..."
    exit 0
}

# Non-dry run: attempt build and run loop (best-effort; may fail if environment lacks toolchain)
if ($Build) {
    Ensure-Dir $buildDir
    Push-Location $buildDir
    try {
        & cmake -S .. -B . -G "Ninja" -DCMAKE_BUILD_TYPE=Release
        & cmake --build . --config Release --target dm_ai_module
    } catch {
        Write-Error "Build step failed: $_"
        Pop-Location
        exit 2
    }
    Pop-Location
    Write-Host "Build finished (check $buildDir)."
}

# Run repeat_native_loader under proc dump wrapper if available
for ($i = 1; $i -le $RunCount; $i++) {
    $runLog = Join-Path $LogDir "native_run_$i.log"
    $cmd = "${env:VIRTUAL_ENV}\Scripts\python.exe scripts/repeat_native_loader.py --count 1 --log-dir $LogDir"
    if (Test-Path $ProcDumpPath) {
        Write-Host "Running (procdump): $ProcDumpPath -accepteula -ma -e 1 -f \"python.exe\" -- $cmd > $runLog 2>&1"
        & $ProcDumpPath -accepteula -ma -e 1 -f "python.exe" -- ${env:VIRTUAL_ENV}\Scripts\python.exe scripts/repeat_native_loader.py --count 1 --log-dir $LogDir *> $runLog
    } else {
        Write-Host "Running: $cmd > $runLog"
        & ${env:VIRTUAL_ENV}\Scripts\python.exe scripts/repeat_native_loader.py --count 1 --log-dir $LogDir *> $runLog
    }
}

if ($CollectEventLog) {
    $evtx = Join-Path $LogDir "Application.evtx"
    Write-Host "Exporting Application event log to $evtx"
    & wevtutil epl Application $evtx
}

Write-Host "run_native_repro_windows completed. Logs: $LogDir"
