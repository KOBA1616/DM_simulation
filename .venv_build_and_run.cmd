@echo off
REM Initialize VS developer environment for x64 and run CMake configure+build
pushd "%~dp0"
if exist build rmdir /s /q build
REM Use Ninja generator by default; do not invoke Visual Studio vcvars
@echo off
setlocal

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

cmake -S "%ROOT%" -B "%ROOT%build" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="%PY%"
cmake --build build
popd
exit /b %ERRORLEVEL%
