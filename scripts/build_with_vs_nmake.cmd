@echo off
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%t
set LOG=%TEMP%\build_vs_nmake_%TS%.log
set SRC_DIR=%~dp0..
set BUILD_DIR=%~dp0..\build_vs_nmake
echo Build started at %DATE% %TIME% > "%LOG%"

rem Initialize MSVC environment and append output to the log
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >> "%LOG%" 2>&1

echo ==== CMake Configure ==== | powershell -NoProfile -Command "Out-Host" >> "%LOG%"
powershell -NoProfile -Command "& { cmake -S '%SRC_DIR%' -B '%BUILD_DIR%' -G 'NMake Makefiles' -DUSE_ONNXRUNTIME=ON -DORT_INCLUDE_DIR='C:\Users\ichirou\DM_simulation\build_mingw_onnx\onnxruntime-1.18.0\onnxruntime-win-x64-1.18.0\include' -DORT_LIB_DIR='C:\Users\ichirou\DM_simulation\build_mingw_onnx\onnxruntime-1.18.0\onnxruntime-win-x64-1.18.0\lib' 2>&1 | Tee-Object -FilePath '%LOG%' -Append }"

echo ==== CMake Build ==== | powershell -NoProfile -Command "Out-Host" >> "%LOG%"
powershell -NoProfile -Command "& { cmake --build '%BUILD_DIR%' --config Release 2>&1 | Tee-Object -FilePath '%LOG%' -Append }"

echo RETURN=%ERRORLEVEL% >> "%LOG%"
type "%LOG%"
exit /b %ERRORLEVEL%
