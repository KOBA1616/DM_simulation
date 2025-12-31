@echo off
REM Clean build artifacts and common temporary build folders
SET ROOT=%~dp0\..
pushd %ROOT%
echo Removing build directories...
if exist build rd /s /q build
if exist Debug rd /s /q Debug
if exist Release rd /s /q Release
if exist RelWithDebInfo rd /s /q RelWithDebInfo
if exist .vs rd /s /q .vs
if exist CMakeFiles rd /s /q CMakeFiles
if exist CMakeCache.txt del /f /q CMakeCache.txt
echo Clean complete
popd
