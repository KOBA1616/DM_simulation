import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
import dm_ai_module
from dm_ai_module import GameState, LethalSolver

print('loaded dm_ai_module from', getattr(dm_ai_module,'__file__',None))
# create game and add attacker
g = GameState(100)
g.active_player_id = 0
# ensure player slots
try:
    g._ensure_player(0)
    g._ensure_player(1)
except Exception:
    pass
# add attacker
g.add_test_card_to_battle(0, 1, 100, False, False)
print('player0 battle count:', len(g.players[0].battle))
print('shield count p1:', len(getattr(g.players[1],'shield_zone',[])))
card_db = {}
kw = dm_ai_module.CardKeywords()
card_db[1] = dm_ai_module.CardDefinition(1, 'Van', 'FIRE', [], [], 2, 2000, kw, [])
card_db[1].type = dm_ai_module.CardType.CREATURE
print('is_lethal ->', LethalSolver.is_lethal(g, card_db))

# Scenario: 1 shield vs 2 attackers
g2 = GameState(100)
g2._ensure_player(0)
g2._ensure_player(1)
g2.add_card_to_shield(1,1,200)
g2.add_test_card_to_battle(0,1,100,False,False)
g2.add_test_card_to_battle(0,1,101,False,False)
print('shields p1:', len(getattr(g2.players[1],'shield_zone',[])))
print('attackers p0:', len(getattr(g2.players[0],'battle',[])))
print('is_lethal ->', LethalSolver.is_lethal(g2, card_db))
