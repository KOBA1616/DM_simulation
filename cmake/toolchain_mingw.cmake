# Toolchain file to use MinGW compiler located in a custom path.
# Usage: pass -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain_mingw.cmake when configuring.

if(NOT DEFINED ENV{MINGW_GCC_PATH})
    message(STATUS "MINGW_GCC_PATH not set. Edit this toolchain file or set the environment variable MINGW_GCC_PATH to the path of x86_64-w64-mingw32-gcc.exe")
else()
    set(MINGW_GCC_PATH "$ENV{MINGW_GCC_PATH}")
endif()

if(EXISTS "${MINGW_GCC_PATH}")
    # Derive the bin directory
    get_filename_component(MINGW_BIN_DIR ${MINGW_GCC_PATH} DIRECTORY)
    message(STATUS "Using MinGW GCC at: ${MINGW_GCC_PATH}")
    set(CMAKE_C_COMPILER "${MINGW_GCC_PATH}" CACHE STRING "C compiler" FORCE)
    # Try to infer g++ alongside gcc
    set(MINGW_GXX_PATH "${MINGW_BIN_DIR}/x86_64-w64-mingw32-g++.exe")
    if(EXISTS "${MINGW_GXX_PATH}")
        set(CMAKE_CXX_COMPILER "${MINGW_GXX_PATH}" CACHE STRING "C++ compiler" FORCE)
    else()
        message(WARNING "Could not find x86_64-w64-mingw32-g++.exe next to gcc; please set CMAKE_CXX_COMPILER manually if needed.")
    endif()
else()
    message(WARNING "Mingw gcc path '${MINGW_GCC_PATH}' does not exist. Please set environment variable MINGW_GCC_PATH or edit this file.")
endif()

set(CMAKE_SYSTEM_NAME Windows)
