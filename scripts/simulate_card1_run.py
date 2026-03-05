import pathlib
import sys

# Ensure project root is on sys.path so native dm_ai_module can be imported
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dm_ai_module as dm

ROOT = pathlib.Path('.')
CARDS_JSON = str(ROOT / 'data' / 'cards.json')

print('Loading DB...')
db = dm.JsonLoader.load_cards(CARDS_JSON)
print('Creating game...')
game = dm.GameInstance(42, db)
s = game.state

CARD_ID = 1
MANA_COST = 2

# Prepare decks and mana
s.set_deck(0, [CARD_ID] * 40)
s.set_deck(1, [1] * 40)
for i in range(MANA_COST):
    s.add_card_to_mana(0, CARD_ID, 9200 + i)
    
# Initialize game start (draw initial hands, set phases)
dm.PhaseManager.start_game(s, db)

# Advance to MAIN for player 0
for _ in range(30):
    if 'MAIN' in str(s.current_phase).upper() and s.active_player_id == 0:
        break
    legal = dm.IntentGenerator.generate_legal_commands(s, db)
    if legal:
        game.resolve_command(legal[0])
    else:
        dm.PhaseManager.next_phase(s, db)

print('Starting MAIN; hand size:', len(s.players[0].hand))

# Find PLAY
play = next((c for c in dm.IntentGenerator.generate_legal_commands(s, db) if 'PLAY' in str(getattr(c,'type','')).upper()), None)
print('PLAY cmd found:', bool(play))
if not play:
    raise SystemExit('No PLAY command')

print('Resolving PLAY')
game.resolve_command(play)
print('hand after play:', len(s.players[0].hand))

# Find RESOLVE
resolve = next((c for c in dm.IntentGenerator.generate_legal_commands(s, db) if 'RESOLVE' in str(getattr(c,'type','')).upper()), None)
print('RESOLVE found:', bool(resolve))
if resolve:
    print('Resolving RESOLVE')
    game.resolve_command(resolve)

# After RESOLVE, show legal commands
print('\nAfter RESOLVE: waiting_for_user_input=', s.waiting_for_user_input)
legal = dm.IntentGenerator.generate_legal_commands(s, db)
print('Legal commands:')
for c in legal:
    print(' -', str(getattr(c,'type','?')), ', attrs:', {k: getattr(c,k) for k in ['target_instance','instance_id'] if hasattr(c,k)})

# SELECT_NUMBER: choose 1 if available
sel_num = next((c for c in legal if 'SELECT_NUMBER' in str(getattr(c,'type','')).upper() and getattr(c,'target_instance',-1)==1), None)
if sel_num:
    print('\nChoosing SELECT_NUMBER=1')
    drawn = getattr(sel_num,'target_instance',0)
    game.resolve_command(sel_num)
    print('After SELECT_NUMBER: waiting=', s.waiting_for_user_input, 'hand=', len(s.players[0].hand))
else:
    print('\nNo SELECT_NUMBER found')

# Resolve SELECT_TARGETs
returned = 0
while True:
    legal = dm.IntentGenerator.generate_legal_commands(s, db)
    sel_targets = [c for c in legal if 'SELECT_TARGET' in str(getattr(c,'type','')).upper()]
    print('\nGenerated SELECT_TARGET count =', len(sel_targets))
    if not sel_targets:
        break
    tgt = sel_targets[0]
    print('Resolving SELECT_TARGET instance_id=', getattr(tgt,'instance_id',None))
    game.resolve_command(tgt)
    returned += 1
    print('After resolving one: waiting=', s.waiting_for_user_input, 'hand=', len(s.players[0].hand))
    if not s.waiting_for_user_input:
        break

print('\nTotal returned =', returned)
print('Final hand size =', len(s.players[0].hand))
print('Deck top ids (first 5) =', s.get_zone(0, dm.Zone.DECK)[:5])
print('Done')
