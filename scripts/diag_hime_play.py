import sys
sys.path.insert(0, 'bin\\Release')
import dm_ai_module as dm

card_db = dm.JsonLoader.load_cards('data/cards.json')

# Himekatt id=6
card_id = 6
print('Testing card', card_id)

game = dm.GameInstance(99, card_db)
config = dm.ScenarioConfig()
config.my_hand_cards = [card_id]
config.my_mana_zone = [1,1,1,1]
config.my_shields = []
config.enemy_shield_count = 5

try:
    game.reset_with_scenario(config)
except Exception as e:
    print('reset_with_scenario error', e)

# Ensure MAIN
game.state.current_phase = dm.Phase.MAIN

print('Before play: pending_effects count=', len(getattr(game.state, 'pending_effects', [])))

from dm_toolkit import commands_v2 as commands
actions = commands.generate_legal_commands(game.state, card_db, strict=False)
print('Generated actions:', len(actions))
for i,a in enumerate(actions[:20]):
    print(i, repr(getattr(a, 'type', a)), getattr(a, 'card_id', None), getattr(a, 'source_instance_id', None))

# Find PLAY_CARD action
play_actions = [a for a in actions if getattr(a, 'type', None) == getattr(dm.ActionType, 'PLAY_CARD', None)]
print('play_actions:', len(play_actions))
if play_actions:
    game.resolve_action(play_actions[0])
    print('Played card')
else:
    print('No play action found')

print('After play: pending_effects count=', len(getattr(game.state, 'pending_effects', [])))

actions2 = commands.generate_legal_commands(game.state, card_db, strict=False)
print('Generated after actions:', len(actions2))
for i,a in enumerate(actions2[:20]):
    print(i, repr(getattr(a, 'type', a)), getattr(a, 'card_id', None), getattr(a, 'source_instance_id', None))
