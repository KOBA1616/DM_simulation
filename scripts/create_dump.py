import sys
import ctypes
import ctypes.wintypes as wintypes
import os

def write_minidump(pid, out_path):
    PROCESS_ALL_ACCESS = 0x1F0FFF
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    dbghelp = ctypes.WinDLL('Dbghelp', use_last_error=True)

    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    OpenProcess.restype = wintypes.HANDLE

    CreateFileW = kernel32.CreateFileW
    CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    CreateFileW.restype = wintypes.HANDLE

    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL

    MiniDumpWriteDump = dbghelp.MiniDumpWriteDump
    MiniDumpWriteDump.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID, wintypes.LPVOID]
    MiniDumpWriteDump.restype = wintypes.BOOL

    hProcess = OpenProcess(PROCESS_ALL_ACCESS, False, int(pid))
    if not hProcess:
        raise OSError('OpenProcess failed, pid=%s' % pid)

    # CREATE_ALWAYS = 2, GENERIC_WRITE = 0x40000000, FILE_SHARE_WRITE=0x2
    GENERIC_WRITE = 0x40000000
    CREATE_ALWAYS = 2
    FILE_ATTRIBUTE_NORMAL = 0x80
    FILE_SHARE_WRITE = 0x2

    hFile = CreateFileW(out_path, GENERIC_WRITE, FILE_SHARE_WRITE, None, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if hFile == wintypes.HANDLE(-1).value:
        CloseHandle(hProcess)
        raise OSError('CreateFile failed: ' + out_path)

    # MINIDUMP_TYPE flags
    MiniDumpWithFullMemory = 0x00000002
    MiniDumpWithHandleData = 0x00000004
    MiniDumpWithFullMemoryInfo = 0x00000800
    dump_type = MiniDumpWithFullMemory | MiniDumpWithHandleData | MiniDumpWithFullMemoryInfo

    ok = MiniDumpWriteDump(hProcess, int(pid), hFile, dump_type, None, None, None)
    CloseHandle(hFile)
    CloseHandle(hProcess)
    if not ok:
        err = ctypes.get_last_error()
        raise OSError('MiniDumpWriteDump failed, err=%d' % err)
    return out_path

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: create_dump.py <pid> <out.dmp>')
        sys.exit(2)
    pid = sys.argv[1]
    out = sys.argv[2]
    out = os.path.abspath(out)
    print('Writing dump for pid', pid, 'to', out)
    try:
        path = write_minidump(pid, out)
        print('Wrote dump:', path)
    except Exception as e:
        print('Failed to write dump:', e)
        sys.exit(1)
