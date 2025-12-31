@echo off
REM CI build script: configure and build using MSVC+Ninja
SET ROOT=%~dp0\..
pushd %ROOT%
if "%1"=="" (
  set BUILD_TYPE=RelWithDebInfo
) else (
  set BUILD_TYPE=%1
)
echo Configuring (%BUILD_TYPE%)...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cmake -S "%CD%" -B "%CD%\build" -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 (
  echo CMake configure failed & popd & exit /b 1
)
echo Building...
cmake --build "%CD%\build" --config %BUILD_TYPE% --parallel
if errorlevel 1 (
  echo Build failed & popd & exit /b 1
)
echo Build succeeded
popd
