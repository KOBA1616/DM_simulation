@echo off
REM Run VS dev env then show cl location
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\vsdevcmd.bat"
where cl
echo RETURN_CODE=%ERRORLEVEL%
pause
