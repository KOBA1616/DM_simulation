# PowerShell script to initialize development environment variables
# Usage: . .\scripts\init_env.ps1

Write-Host "Initializing development environment..." -ForegroundColor Cyan

# 1. Setup GitHub CLI Path
$ghPath = "C:\Program Files\GitHub CLI"
if (Test-Path $ghPath) {
    if ($env:PATH -notlike "*$ghPath*") {
        $env:PATH = "$ghPath;$env:PATH"
        Write-Host "Added GitHub CLI to PATH." -ForegroundColor Green
    } else {
        Write-Host "GitHub CLI is already in PATH." -ForegroundColor Gray
    }
} else {
    Write-Warning "GitHub CLI not found at default location: $ghPath"
}

# 2. Verify Tools
function Test-Tool ($cmd, $name) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $version = & $cmd --version
        Write-Host "Found $name : $version" -ForegroundColor Green
        return $true
    } else {
        Write-Warning "$name ($cmd) not found."
        return $false
    }
}

Test-Tool "gh" "GitHub CLI"
Test-Tool "cmake" "CMake"
Test-Tool "python" "Python"
Test-Tool "git" "Git"

Write-Host "Environment initialization complete." -ForegroundColor Cyan
