import sys
from pathlib import Path
p = Path('CMakeLists.txt')
if not p.exists():
    print('CMakeLists.txt not found')
    sys.exit(1)
raw = p.read_bytes()
text = None
for enc in ('utf-8','utf-16','utf-16-le','utf-16-be','latin-1'):
    try:
        text = raw.decode(enc)
        print('decoded with', enc)
        break
    except Exception:
        pass
if text is None:
    print('Failed to decode CMakeLists.txt')
    sys.exit(2)
p.write_text(text, encoding='utf-8')
print('Wrote CMakeLists.txt as UTF-8')
