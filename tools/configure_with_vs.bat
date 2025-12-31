@echo off
REM Configure build using MSVC environment and Ninja
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cmake -S "C:\Users\ichirou\DM_simulation" -B "C:\Users\ichirou\DM_simulation\build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
