import dm_ai_module
print('module:', dm_ai_module)
print('PlayerIntent?', hasattr(dm_ai_module, 'PlayerIntent'))
print('PassiveType?', hasattr(dm_ai_module, 'PassiveType'))
print('EffectActionType?', hasattr(dm_ai_module, 'EffectActionType'))
print('SelfAttention?', hasattr(dm_ai_module, 'SelfAttention'))
print('CommandDef?', hasattr(dm_ai_module, 'CommandDef'))
print('sample attrs:', [n for n in dir(dm_ai_module) if 'Player' in n or 'Passive' in n or 'CommandDef' in n][:20])
