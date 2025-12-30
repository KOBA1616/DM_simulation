@echo off
REM Initialize VS developer environment for x64 and run CMake configure+build
pushd "%~dp0"
if exist build rmdir /s /q build
REM Use Ninja generator by default; do not invoke Visual Studio vcvars
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="C:\Users\ichirou\DM_simulation\.venv\Scripts\python.exe"
cmake --build build
popd
exit /b %ERRORLEVEL%
