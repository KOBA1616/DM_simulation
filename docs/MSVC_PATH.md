Detected Visual Studio installation:

- Installation root: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
- vcvars scripts:
  - C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
  - C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat

How to configure and build from PowerShell (recommended):

1) Start a VS developer environment (recommended):

    & "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

2) Or source vcvars and run CMake in a single cmd chain:

    cmd /c ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 && cmake -S . -B build -G Ninja -A x64 && cmake --build build --config Release -j 2"

Notes:
- If Ninja is not installed or not desired, use the Visual Studio generator instead:

    cmake -S . -B build -G "Visual Studio 17 2022" -A x64

- This file was created automatically by the migration helper. Update if you have a different Visual Studio installation.
