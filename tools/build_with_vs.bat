@echo off
REM Build using MSVC environment and Ninja
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cmake --build "C:\Users\ichirou\DM_simulation\build" --config RelWithDebInfo --parallel
