p = 'C:/Users/ichirou/DM_simulation/scripts/run_gui.ps1'
import io
b = open(p, 'rb').read()
# UTF-8 BOM
if b.startswith(b'\xef\xbb\xbf'):
    open(p, 'wb').write(b[3:])
    print('Removed BOM')
else:
    print('No BOM found')
