import sys, pathlib
repo_root = str(pathlib.Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import dm_ai_module as dm
print('DM module path:', dm.__file__)
print('Has CommandEncoder:', hasattr(dm, 'CommandEncoder'))
print('CommandEncoder repr:', getattr(dm, 'CommandEncoder', None))
print('dir snippet:', [k for k in dir(dm) if 'Command' in k][:20])
