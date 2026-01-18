import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print('GameResult values:')
for attr in dir(dm.GameResult):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(dm.GameResult, attr)
        print(attr, int(val))
    except Exception:
        pass

print('raw NONE:', dm.GameResult.NONE)
print('raw P1_WIN:', dm.GameResult.P1_WIN)
print('raw P2_WIN:', dm.GameResult.P2_WIN)
