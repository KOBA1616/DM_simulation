from dm_toolkit.engine.compat import EngineCompat
import dm_ai_module as dm

CARD_DB = dm.JsonLoader.load_cards('data/cards.json')

inst = dm.GameInstance(0, CARD_DB)
# Try canonical start_game
try:
    inst.start_game()
except Exception:
    try:
        dm.PhaseManager.start_game(inst.state, CARD_DB)
    except Exception:
        try:
            inst.state.setup_test_duel()
        except Exception:
            pass

p0 = EngineCompat.get_player(inst.state, 0)
print('Before -> hand:', len(getattr(p0, 'hand', [])), 'mana:', len(getattr(p0, 'mana_zone', [])))

# Ensure at least one card in hand
if len(getattr(p0, 'hand', [])) == 0:
    try:
        inst.state.add_card_to_hand(0, 1)
    except Exception:
        # fallback manual
        try:
            c = type('C', (), {'instance_id': inst.state.get_next_instance_id()})()
            p0.hand.append(c)
        except Exception:
            pass

p0 = EngineCompat.get_player(inst.state, 0)
if len(getattr(p0, 'hand', [])) > 0:
    iid = getattr(p0.hand[0], 'instance_id', None)
else:
    iid = None

cmd = {'type': 'MANA_CHARGE', 'instance_id': iid}
print('Executing command:', cmd)
EngineCompat.ExecuteCommand(inst.state, cmd, CARD_DB)

p0_after = EngineCompat.get_player(inst.state, 0)
print('After  -> hand:', len(getattr(p0_after, 'hand', [])), 'mana:', len(getattr(p0_after, 'mana_zone', [])))
print('Done')
