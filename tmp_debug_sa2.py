import dm_ai_module
from dm_ai_module import GameState

s = GameState(100)
s.add_test_card_to_battle(0,1,10,False,True)
ci = s.players[0].battle[0]
print('ci.is_summon_sick:', getattr(ci,'is_summon_sick', None))
# create cdef as in earlier
cdef = dm_ai_module.CardDefinition(1, 'Vanilla', 'FIRE', [], ['Human'], 1, 1000, dm_ai_module.CardKeywords(), [])
card_db = {1: cdef}
print('cdef.keywords.speed_attacker:', getattr(getattr(cdef,'keywords',None),'speed_attacker',None))
print('ci.speed_attacker:', getattr(ci,'speed_attacker', None))
print("granted_keywords:", getattr(ci,'granted_keywords', None))
cond = not (cdef and getattr(getattr(cdef, 'keywords', None), 'speed_attacker', False)) and not getattr(ci, 'speed_attacker', False) and not ('SPEED_ATTACKER' in getattr(ci, 'granted_keywords', set()))
print('should continue?', cond)
