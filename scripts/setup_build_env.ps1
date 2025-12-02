# PowerShell helper script to set up a persistent, usable build environment on Windows.
# - Installs CMake via winget if available
# - Adds common CMake and Python install locations to the USER PATH (persistent)
# - Installs Python dev requirements from requirements-dev.txt

param(
    [switch]$InstallCMake,
    [string]$PythonMajorMinor = "Python311"
)

function Write-Info($m) { Write-Host "[info] $m" -ForegroundColor Cyan }
function Write-Warn($m) { Write-Host "[warn] $m" -ForegroundColor Yellow }

Write-Info "Starting build environment setup script"

# 1) Optionally install CMake with winget
if ($InstallCMake) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "Installing CMake via winget..."
        winget install --id Kitware.CMake -e --silent
    } else {
        Write-Warn "winget not found. Please install CMake manually or enable winget."
    }
} else {
    Write-Info "Skipping CMake install (use -InstallCMake to auto-install with winget)"
}

# 2) Build list of paths to ensure in user PATH
$cmakePath = 'C:\Program Files\CMake\bin'
$pythonBase = Join-Path $env:LOCALAPPDATA "Programs\$PythonMajorMinor"
$pythonScripts = Join-Path $pythonBase 'Scripts'

$desired = @($cmakePath, $pythonBase, $pythonScripts)

# Get current user PATH
$userPath = [Environment]::GetEnvironmentVariable('Path','User')
if (-not $userPath) { $userPath = '' }

foreach ($p in $desired) {
    if (-not (Test-Path $p)) {
        Write-Warn "Path not found on disk: $p (will still add to PATH if requested)"
    }
    if ($userPath -notlike "*${p}*") {
        Write-Info "Adding to user PATH: $p"
        if ($userPath -eq '') { $userPath = $p } else { $userPath = "$userPath;$p" }
    } else {
        Write-Info "Already present in user PATH: $p"
    }
}

# Persist the new user PATH
try {
    setx Path $userPath | Out-Null
    Write-Info "User PATH updated. Note: open a new PowerShell session to see changes."
} catch {
    Write-Warn "Failed to set user PATH via setx: $_"
}

# 3) Install Python dev dependencies if requirements file exists
$req = Join-Path (Get-Location) 'requirements-dev.txt'
if (Test-Path $req) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Info "Installing Python dev dependencies from requirements-dev.txt"
        python -m pip install --upgrade pip
        python -m pip install -r $req
    } else {
        Write-Warn "Python not found in PATH. Install Python and re-run this script or add Python to PATH."
    }
} else {
    Write-Warn "requirements-dev.txt not found at $req"
}

Write-Info "Setup script finished. Open a new terminal to pick up updated PATH."