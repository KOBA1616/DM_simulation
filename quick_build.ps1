cd build-msvc
cmake --build . --config Release --target dm_ai_module 2>&1 | Tee-Object ../build_quick.log
