$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectRoot "build"

# Ensure build directory exists
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Ensure DLLs are present (Copy from MinGW if available)
# This is a workaround for "DLL load failed" on Windows
$mingwBin = "C:\Users\mediastation36\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20251118-ucrt-x86_64\bin"
if (Test-Path $mingwBin) {
    Write-Host "Copying MinGW DLLs to build directory..."
    $dlls = Get-ChildItem "$mingwBin\*.dll"
    foreach ($dll in $dlls) {
        $dest = Join-Path $buildDir $dll.Name
        if (-not (Test-Path $dest)) {
            Copy-Item $dll.FullName -Destination $buildDir
        }
    }
}

# Add build directory and python source directory to PYTHONPATH
$env:PYTHONPATH = "$buildDir;$projectRoot/python;$env:PYTHONPATH"

Write-Host "Starting GUI..."
python "$projectRoot/python/gui/app.py"
