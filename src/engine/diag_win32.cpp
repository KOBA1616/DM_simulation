#include "diag_win32.h"
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>

void diag_write_win32(const std::string &s) {
    HANDLE h = CreateFileA("logs\\diag_win32.txt",
                           FILE_APPEND_DATA,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL,
                           OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL,
                           NULL);
    if (h == INVALID_HANDLE_VALUE) return;
    SetFilePointer(h, 0, NULL, FILE_END);
    DWORD written = 0;
    std::string out = s + "\n";
    WriteFile(h, out.c_str(), (DWORD)out.size(), &written, NULL);
    CloseHandle(h);
}
#else
#include <fstream>
void diag_write_win32(const std::string &s) {
    try {
        std::ofstream f("logs/diag_win32.txt", std::ios::app);
        if (f) { f << s << "\n"; f.close(); }
    } catch(...) {}
}
#endif
