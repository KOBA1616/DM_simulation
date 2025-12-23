@echo off
setlocal

REM Check if python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python.
    exit /b 1
)

REM Add bin directory to PYTHONPATH
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\bin;%PYTHONPATH%"

echo Starting DM AI Simulator...
python "%PROJECT_ROOT%\dm_toolkit\gui\app.py" %*

endlocal
