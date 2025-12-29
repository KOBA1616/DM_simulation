import dm_ai_module
from dm_ai_module import EffectDef, TriggerType, ConditionDef, ActionDef, EffectActionType

action = ActionDef()
action.type = EffectActionType.CAST_SPELL
try:
    action.filter.zones = ['HAND']
    action.filter.owner = 'SELF'
except Exception:
    pass

print('About to construct')
eff = EffectDef(TriggerType.NONE, ConditionDef(), [action])
print('Constructed:', type(eff), eff)
