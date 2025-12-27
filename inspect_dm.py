import dm_ai_module, inspect
print('EffectDef:', dm_ai_module.EffectDef)
try:
    print('sig:', inspect.signature(dm_ai_module.EffectDef))
except Exception as e:
    print('sig-error', e)
print('FilterDef:', dm_ai_module.FilterDef)
print('Instruction.set_args sig:', inspect.signature(dm_ai_module.Instruction.set_args))
print('EffectDef init:', getattr(dm_ai_module.EffectDef, '__init__'))
