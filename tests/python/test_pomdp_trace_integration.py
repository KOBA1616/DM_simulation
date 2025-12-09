import json
import pathlib
import dm_ai_module


def load_trace(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_state_to_gamestate(step):
    # create a minimal GameState and populate zones per the trace step
    s = dm_ai_module.GameState(0)
    for pid, p in enumerate(step.get('players', [])):
        # deck
        for c in p.get('deck', []):
            ci = dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0)))
            s.players[pid].deck.append(ci)
        # hand
        for c in p.get('hand', []):
            ci = dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0)))
            s.players[pid].hand.append(ci)
        # battle_zone
        for c in p.get('battle_zone', []):
            ci = dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0)))
            s.players[pid].battle_zone.append(ci)
        # shield_zone
        for c in p.get('shield_zone', []):
            ci = dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0)))
            s.players[pid].shield_zone.append(ci)
        # graveyard
        for c in p.get('graveyard', []):
            ci = dm_ai_module.CardInstance(int(c['card_id']), int(c.get('instance_id', 0)))
            s.players[pid].graveyard.append(ci)
    return s


def test_trace_monotonic_penalty():
    trace_path = pathlib.Path(__file__).parent / 'data' / 'example_trace.json'
    trace = load_trace(trace_path)

    # use three candidate card ids for a simple belief vector
    ids = [1, 2, 3]
    b = dm_ai_module.ParametricBelief()
    b.initialize_ids(ids)
    vec0 = b.get_vector()
    assert abs(sum(vec0) - 1.0) < 1e-6

    # apply step 1 (card seen in deck), then step 2 (seen in hand)
    v_after_deck = None
    v_after_hand = None

    for step in trace:
        s = apply_state_to_gamestate(step)
        b.update(s)
        v = b.get_vector()
        if step['step'] == 1:
            v_after_deck = v
        if step['step'] == 2:
            v_after_hand = v

    assert v_after_deck is not None and v_after_hand is not None
    # ensure normalization
    assert abs(sum(v_after_deck) - 1.0) < 1e-6
    assert abs(sum(v_after_hand) - 1.0) < 1e-6

    # monotonic: initial >= after_deck >= after_hand for the observed card (id 1 at index 0)
    assert vec0[0] + 1e-9 >= v_after_deck[0]
    assert v_after_deck[0] + 1e-9 >= v_after_hand[0]
