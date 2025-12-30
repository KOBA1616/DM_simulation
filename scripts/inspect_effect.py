import dm_ai_module
import inspect
print('EffectDef obj:', dm_ai_module.EffectDef)
print('repr type:', type(dm_ai_module.EffectDef))
print('name:', getattr(dm_ai_module.EffectDef,'__name__',None))
try:
    print('init sig:', inspect.signature(dm_ai_module.EffectDef.__init__))
except Exception as e:
    print('init sig error:', e)
print('dir snippets:', [n for n in dir(dm_ai_module) if 'Effect' in n][:20])
