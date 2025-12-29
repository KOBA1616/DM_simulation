@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
echo --- PATH ---
echo %PATH%
echo --- where cmake ---
where cmake || echo CMAKE_NOT_FOUND
echo --- cmake --version ---
cmake --version 2>nul || echo CMAKE_VERSION_UNKNOWN
echo --- where nmake ---
where nmake || echo NMAKE_NOT_FOUND
echo --- where cl ---
where cl || echo CL_NOT_FOUND
pause
