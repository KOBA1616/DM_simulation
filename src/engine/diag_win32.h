#pragma once
#include <string>

// Declaration only; implementation is in diag_win32.cpp to avoid including <windows.h>
void diag_write_win32(const std::string &s);
