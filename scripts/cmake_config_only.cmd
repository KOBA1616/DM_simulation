@echo off
set LOG=%~dp0cmake_config_%RANDOM%.log
echo Running CMake configure > "%LOG%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cmake -S "%CD%" -B "%CD%\\build_vs_nmake" -G "NMake Makefiles" -DUSE_ONNXRUNTIME=ON -DORT_INCLUDE_DIR="C:\Users\ichirou\DM_simulation\build_mingw_onnx\onnxruntime-1.18.0\onnxruntime-win-x64-1.18.0\include" -DORT_LIB_DIR="C:\Users\ichirou\DM_simulation\build_mingw_onnx\onnxruntime-1.18.0\onnxruntime-win-x64-1.18.0\lib" > "%LOG%" 2>&1
echo RETURN=%ERRORLEVEL% >> "%LOG%"
type "%LOG%"
pause
