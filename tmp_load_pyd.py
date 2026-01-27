import importlib.util, importlib.machinery, os, traceback
search_root = os.path.join('native_prototypes','index_to_command','build')
print('search_root:', os.path.abspath(search_root))
found = []
for root, dirs, files in os.walk(search_root):
    for f in files:
        if f.startswith('index_to_command_native') and f.endswith('.pyd'):
            found.append(os.path.join(root, f))
print('candidates:', found)
if not found:
    print('No candidates')
else:
    path = sorted(found, key=lambda p: ('Release' not in p, p))[0]
    print('trying to load', path)
    try:
        name = 'index_to_command_native_local'
        loader = importlib.machinery.ExtensionFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        print('loaded attrs:', [a for a in dir(mod) if not a.startswith('__')])
    except Exception as e:
        print('failed to load:', e)
        traceback.print_exc()
