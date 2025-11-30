<#
This script will add the specified MinGW bin directory to the user's PATH permanently.
It expects the variable $mingwGccPath to point to the gcc executable, for example:
  C:\Program Files (x86)\mingw64\bin\x86_64-w64-mingw32-gcc.exe

Usage (PowerShell, run as current user):
  .\scripts\setup_mingw_env.ps1 -GccPath "C:\Program Files (x86)\mingw64\bin\x86_64-w64-mingw32-gcc.exe"

This will prepend the bin directory to the user's PATH if not already present.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$GccPath
)

if (-not (Test-Path $GccPath)) {
    Write-Error "Provided path does not exist: $GccPath"
    exit 1
}

$binDir = Split-Path -Path $GccPath -Parent
Write-Host "Detected MinGW bin directory: $binDir"

$current = [Environment]::GetEnvironmentVariable("Path", "User")
if ($current -and $current.Split(';') -contains $binDir) {
    Write-Host "Bin directory already in user PATH. No changes made."
    exit 0
}

$newPath = $binDir + ";" + $current
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")
Write-Host "User PATH updated to include: $binDir"
Write-Host "You may need to restart your shell or log out/in for changes to take effect."

Write-Host "Also setting MINGW_GCC_PATH environment variable for current process and user."
[Environment]::SetEnvironmentVariable("MINGW_GCC_PATH", $GccPath, "User")
[Environment]::SetEnvironmentVariable("MINGW_GCC_PATH", $GccPath, "Process")

Write-Host "Done."
