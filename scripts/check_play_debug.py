import pathlib, os
import dm_ai_module as dm
ROOT=pathlib.Path(__file__).resolve().parents[1]
CARDS=str(ROOT/'data'/'cards.json')
db=dm.JsonLoader.load_cards(CARDS)
# use seed 42
game=dm.GameInstance(42, db)
s=game.state
s.set_deck(0, [1]*40)
# replicate test setup
dm.PhaseManager.start_game(s, db)
print('phase=>', s.current_phase, 'active=>', s.active_player_id)
print('hand len p0=', len(s.players[0].hand))
cmds=dm.IntentGenerator.generate_legal_commands(s, db)
print('generated cmds count=', len(cmds))
for c in cmds:
    print('cmd.type=', getattr(c,'type', None), 'attrs=', {k: v for k,v in getattr(c,'__dict__', {}).items()})
