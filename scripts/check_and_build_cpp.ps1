<#
.DESCRIPTION
  Check environment for C++ toolchain (MSVC / g++) and run CMake+build.

.SYNOPSIS
  .\scripts\check_and_build_cpp.ps1 [-EnableCppTests]

.PARAMETER EnableCppTests
  When provided, passes -DENABLE_CPP_TESTS=ON to CMake.

.NOTES
  This helper does not install toolchains. It prints platform-specific
  guidance when no compiler is found.
#>

param(
    [switch]$EnableCppTests
)

function Check-Command($name) {
    try { Get-Command $name -ErrorAction Stop > $null; return $true } catch { return $false }
}

Write-Host "Checking toolchain..."

$hasCMake = Check-Command cmake
$hasNinja = Check-Command ninja
$hasCl = Check-Command cl
$hasGpp = Check-Command g++
$hasClang = Check-Command clang++

if (-not $hasCMake) {
    Write-Error "cmake not found in PATH. Please install CMake and re-run."
    exit 1
}

if (-not ($hasCl -or $hasGpp -or $hasClang)) {
    Write-Warning "No C++ compiler found in PATH."
    Write-Host "Options to install a compiler on Windows:"
    Write-Host "  - Install Visual Studio (with 'Desktop development with C++') or Visual Studio Build Tools."
    Write-Host "  - Or install MSYS2/MinGW and add its mingw64/bin to PATH."
    Write-Host "After installing, re-open the shell where 'cl' or 'g++' is available."
    exit 2
}

$generator = 'Ninja'
if (-not $hasNinja) {
    Write-Host "ninja not found; falling back to default generator (may prompt interactively)."
    $generator = 'Default'
}

$buildDir = Join-Path -Path (Get-Location) -ChildPath 'build'
Write-Host "Configuring build in: $buildDir"

$cmakeArgs = @('-S', '.', '-B', 'build', '-G', $generator, '-DCMAKE_BUILD_TYPE=Release')
if ($EnableCppTests) { $cmakeArgs += '-DENABLE_CPP_TESTS=ON' }

Write-Host "Running: cmake $($cmakeArgs -join ' ')"
$proc = Start-Process -FilePath cmake -ArgumentList $cmakeArgs -NoNewWindow -Wait -PassThru
if ($proc.ExitCode -ne 0) { Write-Error "CMake configuration failed with exit code $($proc.ExitCode)"; exit $proc.ExitCode }

Write-Host "Building..."
$buildArgs = @('--build', 'build')
$proc2 = Start-Process -FilePath cmake -ArgumentList $buildArgs -NoNewWindow -Wait -PassThru
if ($proc2.ExitCode -ne 0) { Write-Error "Build failed with exit code $($proc2.ExitCode)"; exit $proc2.ExitCode }

Write-Host "Build completed. If you enabled C++ tests, you can run:"
Write-Host "  ctest --test-dir build -V"

Write-Host "If you need to rebuild the Python extension for the venv, run your usual quick_build script or ensure build artifacts are copied to the Python extension path."
