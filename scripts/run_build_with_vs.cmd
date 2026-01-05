@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake -S "%~dp0.." -B "%~dp0..\build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "%~dp0..\build" --config Release
pause
