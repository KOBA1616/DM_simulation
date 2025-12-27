import importlib
m = importlib.import_module('dm_ai_module')
print('module file:', getattr(m,'__file__',None))
print('has ActionGenerator:', hasattr(m,'ActionGenerator'))
for name in ['ActionGenerator','GenericCardSystem','Action','ActionDef','EffectResolver']:
    print(name, '->', hasattr(m,name))
print('sample names containing Action:', [n for n in dir(m) if 'Action' in n][:50])
