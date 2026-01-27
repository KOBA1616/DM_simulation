import ctypes, os
p = os.path.join('native_prototypes','index_to_command','build','Release','index_to_command_native.cp312-win_amd64.pyd')
print('trying ctypes.WinDLL on', p)
try:
    dll = ctypes.WinDLL(p)
    print('ctypes loaded OK, attrs:', [a for a in dir(dll)[:20]])
except OSError as e:
    print('OSError:', e)
    try:
        FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
        buf = ctypes.create_unicode_buffer(1024)
        ctypes.windll.kernel32.FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM, None, e.winerror, 0, buf, len(buf), None)
        print('WinError code:', e.winerror)
        print('Message:', buf.value)
    except Exception:
        pass
