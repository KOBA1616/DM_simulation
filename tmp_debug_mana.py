import dm_ai_module
from dm_ai_module import GameState, CardDefinition, Civilization

state = GameState(100)
# Add Fire and Nature mana
state.add_card_to_mana(0, 1, 101)
state.add_card_to_mana(0, 2, 102)
# Create card def
cdef = CardDefinition()
cdef.id = 50
cdef.cost = 2
cdef.civilizations = [Civilization.FIRE, Civilization.NATURE]

# Card DB for mana types
mana_fire = CardDefinition()
mana_fire.id = 1
mana_fire.civilizations = [Civilization.FIRE]
mana_nature = CardDefinition()
mana_nature.id = 2
mana_nature.civilizations = [Civilization.NATURE]
card_db = {1: mana_fire, 2: mana_nature, 50: cdef}

print('Before:', [getattr(ci,'card_id',None) for ci in state.players[0].mana_zone])
print('ci civ names:', [[ (c.name if hasattr(c,'name') else str(c)) for c in getattr(ci,'civilizations',[]) ] for ci in state.players[0].mana_zone])
res = dm_ai_module.ManaSystem.auto_tap_mana(state, state.players[0], cdef, card_db)
print('auto_tap_mana returned', res)
print('After tapped states:', [getattr(ci,'is_tapped',None) for ci in state.players[0].mana_zone])
