import json
import pathlib
import dm_ai_module


def load_trace(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_state_to_gamestate(step):
    s = dm_ai_module.GameState(0)
    for pid, p in enumerate(step.get('players', [])):
        for c in p.get('deck', []): s.players[pid].deck.append(dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0))))
        for c in p.get('hand', []): s.players[pid].hand.append(dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0))))
        for c in p.get('battle_zone', []): s.players[pid].battle_zone.append(dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0))))
        for c in p.get('shield_zone', []): s.players[pid].shield_zone.append(dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0))))
        for c in p.get('graveyard', []): s.players[pid].graveyard.append(dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0))))
    return s


def test_graveyard_trace_reduces_prob():
    trace_path = pathlib.Path(__file__).parent / 'data' / 'trace_graveyard.json'
    trace = load_trace(trace_path)
    ids = [9, 10, 11]
    b = dm_ai_module.ParametricBelief()
    b.initialize_ids(ids)
    vec0 = b.get_vector()

    prev = None
    v_after: list[float] | None = None
    for step in trace:
        s = apply_state_to_gamestate(step)
        if prev is None:
            b.update(s)
        else:
            b.update_with_prev(prev, s)
        prev = s
        v_after = b.get_vector()

    assert v_after is not None
    assert abs(sum(v_after) - 1.0) < 1e-6
    assert vec0[0] >= v_after[0]
