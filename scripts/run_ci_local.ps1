[CmdletBinding()]
param(
  [ValidateSet('Release','RelWithDebInfo','Debug')]
  [string]$Configuration = 'Release',

  [switch]$SkipBuild,
  [switch]$SkipCTest,
  [switch]$SkipPytest,
  [switch]$SkipMypy,

  [string]$LogDir = (Join-Path $PSScriptRoot '..\dumps\logs')
)

$ErrorActionPreference = 'Stop'

# Ensure Python and its console output are UTF-8 regardless of Windows locale.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Invoke-Step {
  param(
    [Parameter(Mandatory)] [string]$Name,
    [Parameter(Mandatory)] [scriptblock]$Script,
    [Parameter(Mandatory)] [string]$LogFile,
    [Parameter(Mandatory)] [string]$StepLogFile
  )

  Write-Host "==> $Name"

  $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  "[$timestamp] START $Name" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
  "[$timestamp] START $Name" | Out-File -FilePath $StepLogFile -Encoding UTF8 -Append

  $LASTEXITCODE = 0
  $output = & $Script 2>&1
  $exitCode = $LASTEXITCODE

  $output | Out-File -FilePath $LogFile -Encoding UTF8 -Append
  $output | Out-File -FilePath $StepLogFile -Encoding UTF8 -Append

  $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  "[$timestamp] END $Name (exit=$exitCode)" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
  "[$timestamp] END $Name (exit=$exitCode)" | Out-File -FilePath $StepLogFile -Encoding UTF8 -Append

  if ($exitCode -ne 0) {
    throw "$Name failed (exit=$exitCode). See log: $LogFile"
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
  $pythonExe = 'python'
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = (Join-Path $LogDir "ci_local_${runStamp}")
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$logFile = (Join-Path $runDir "ci_local_${runStamp}.log")
$metaFile = (Join-Path $runDir "00_meta.log")

"CI local run: $runStamp" | Out-File -FilePath $logFile -Encoding UTF8
"Repo: $repoRoot" | Out-File -FilePath $logFile -Encoding UTF8 -Append
"Python: $pythonExe" | Out-File -FilePath $logFile -Encoding UTF8 -Append
"Configuration: $Configuration" | Out-File -FilePath $logFile -Encoding UTF8 -Append

"CI local run: $runStamp" | Out-File -FilePath $metaFile -Encoding UTF8
"Repo: $repoRoot" | Out-File -FilePath $metaFile -Encoding UTF8 -Append
"Python: $pythonExe" | Out-File -FilePath $metaFile -Encoding UTF8 -Append
"Configuration: $Configuration" | Out-File -FilePath $metaFile -Encoding UTF8 -Append

Push-Location $repoRoot
try {
  if (-not $SkipBuild) {
    Invoke-Step -Name 'Install Python dev deps' -LogFile $logFile -StepLogFile (Join-Path $runDir '10_python_deps.log') -Script {
      & $pythonExe -m pip install --upgrade pip setuptools wheel
      if (Test-Path 'requirements-dev.txt') { & $pythonExe -m pip install -r requirements-dev.txt }
    }

    # Configure+build only if build dir missing or CMakeCache missing
    Invoke-Step -Name 'Configure CMake' -LogFile $logFile -StepLogFile (Join-Path $runDir '20_cmake_configure.log') -Script {
      if (-not (Test-Path 'build\CMakeCache.txt')) {
        cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=$Configuration
      } else {
        Write-Host 'CMake already configured: build/CMakeCache.txt exists'
      }
    }

    Invoke-Step -Name 'Build (CMake)' -LogFile $logFile -StepLogFile (Join-Path $runDir '30_cmake_build.log') -Script {
      cmake --build build --config $Configuration -- /m
    }
  }

  if (-not $SkipCTest) {
    Invoke-Step -Name 'Run C++ tests (ctest)' -LogFile $logFile -StepLogFile (Join-Path $runDir '40_ctest.log') -Script {
      if (Test-Path 'build\CTestTestfile.cmake') {
        Push-Location build
        try { ctest -V } finally { Pop-Location }
      } else {
        Write-Host 'No CTest file; skipping ctest'
      }
    }
  }

  if (-not $SkipPytest) {
    Invoke-Step -Name 'Run Python tests (pytest)' -LogFile $logFile -StepLogFile (Join-Path $runDir '50_pytest.log') -Script {
      if (Test-Path 'build\Release') { $env:PYTHONPATH = "$env:PYTHONPATH;$(Resolve-Path 'build\Release')" }
      if (Test-Path 'build') { $env:PYTHONPATH = "$env:PYTHONPATH;$(Resolve-Path 'build')" }
      & $pythonExe -m pytest -q
    }
  }

  if (-not $SkipMypy) {
    Invoke-Step -Name 'Run mypy' -LogFile $logFile -StepLogFile (Join-Path $runDir '60_mypy.log') -Script {
      & $pythonExe -m mypy --config-file mypy.ini dm_toolkit
    }
  }

  Write-Host "\nAll CI-local steps passed. Log dir: $runDir"
}
finally {
  Pop-Location
}
