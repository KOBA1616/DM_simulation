@echo off
REM Wrapper to run build.ps1 from CMD/VSCode terminals reliably
SETLOCAL ENABLEDELAYEDEXPANSION
REM Resolve script directory
SET "SCRIPT_DIR=%~dp0"
PUSHD "%SCRIPT_DIR%.."
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%SCRIPT_DIR%build.ps1' %*"
IF %ERRORLEVEL% NEQ 0 (
    echo Build script failed with exit code %ERRORLEVEL%
    EXIT /B %ERRORLEVEL%
)
POPD
ENDLOCAL
EXIT /B 0
