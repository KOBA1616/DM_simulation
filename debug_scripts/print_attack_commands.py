from __future__ import annotations
import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import dm_ai_module as dm
print('dm_ai_module file=', getattr(dm, '__file__', None))

def _make_game():
    db = dm.CardDatabase()
    db[1] = dm.CardDefinition(1, "TestCard", "NONE", [], 0, 1000, dm.CardKeywords(), [])
    game = dm.GameInstance(0, db)
    game.state.set_deck(0, [1] * 40)
    game.state.set_deck(1, [1] * 40)
    game.start_game()
    for _ in range(10):
        if "ATTACK" in str(game.state.current_phase).upper():
            break
        dm.PhaseManager.next_phase(game.state, db)
    return game, db

if __name__ == '__main__':
    game, db = _make_game()
    attacker_iid = 9011
    target_iid = 9012
    game.state.add_test_card_to_battle(0, 1, attacker_iid, False, False)
    game.state.add_test_card_to_battle(1, 1, target_iid, False, False)
    p = dm.PassiveEffect()
    p.type = dm.PassiveType.ALLOW_ATTACK_UNTAPPED
    p.specific_targets = [attacker_iid]
    p.controller = 0
    game.state.add_passive_effect(p)
    print('passive_count=', game.state.get_passive_effect_count())
    for i, eff in enumerate(game.state.passive_effects):
        print('eff', i, eff.type, getattr(eff, 'specific_targets', None))
    print('active_player=', game.state.active_player_id)
    for pid in [0,1]:
        bz = game.state.players[pid].battle_zone
        print(f'player {pid} battle_zone ids=', [c.instance_id for c in bz])
        for c in bz:
            print('  card', c.instance_id, 'card_id', c.card_id, 'is_tapped', c.is_tapped, 'sickness', getattr(c,'summoning_sickness',None), 'turn_played', getattr(c,'turn_played',None))
    legal = dm.IntentGenerator.generate_legal_commands(game.state, db)
    print('legal count=', len(legal))
    for c in legal:
        print('CMD type=', c.type, 'inst=', getattr(c,'instance_id',None), 'tgt=', getattr(c,'target_instance',None), 'slot=', getattr(c,'slot_index',None))
