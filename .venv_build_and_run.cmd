@echo off
REM Initialize VS developer environment for x64 and run CMake configure+build
pushd "%~dp0"
if exist build rmdir /s /q build
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="C:\Users\ichirou\DM_simulation\.venv\Scripts\python.exe" -DPYTHON_EXECUTABLE="C:\Users\ichirou\DM_simulation\.venv\Scripts\python.exe"
cmake --build build --config Release -- /m
popd
exit /b %ERRORLEVEL%
