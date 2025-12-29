@echo off
REM Launch Developer Command Prompt (x64) then run CMake + build with NMake
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\LaunchDevCmd.bat" -arch=amd64
cmake -S "%~dp0" -B "%~dp0build_vs" -G "NMake Makefiles" -DUSE_ONNXRUNTIME=ON -DORT_INCLUDE_DIR="C:/Users/ichirou/DM_simulation/build_mingw_onnx/onnxruntime-1.18.0/onnxruntime-win-x64-1.18.0/include" -DORT_LIB_DIR="C:/Users/ichirou/DM_simulation/build_mingw_onnx/onnxruntime-1.18.0/onnxruntime-win-x64-1.18.0/lib"
if %errorlevel% neq 0 exit /b %errorlevel%
cmake --build "%~dp0build_vs" --config Release -- /m
exit /b %errorlevel%
