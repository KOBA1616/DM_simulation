import sys
import types

from dm_toolkit import commands


def test_play_heuristic_with_mocked_dm_ai_module():
    # Prepare a fake dm_ai_module to avoid loading native extensions.
    fake = types.SimpleNamespace()

    class ActionType:
        PASS = 'PASS'
        MANA_CHARGE = 'MANA_CHARGE'
        PLAY_CARD = 'PLAY_CARD'

    class Action:
        def __init__(self):
            self.type = None
            self.card_id = None
            self.source_instance_id = -1

        def __repr__(self):
            return f"<Action type={self.type}>"

    # dm_ai_module.generate_commands returns no native commands to force Python fallback.
    fake.generate_commands = staticmethod(lambda state, card_db=None: [])

    class PhaseManager:
        @staticmethod
        def next_phase(state, card_db=None):
            return None

    fake.Action = Action
    fake.ActionType = ActionType
    fake.generate_commands = fake.generate_commands
    fake.PhaseManager = PhaseManager

    sys.modules['dm_ai_module'] = fake

    # Build a minimal fake game state with two cards in hand.
    class FakePhase:
        def __init__(self, name):
            self.name = name

    class FakeCardInstance:
        def __init__(self, card_id, instance_id=1):
            self.card_id = card_id
            self.instance_id = instance_id

    class FakePlayer:
        def __init__(self, hand, mana_zone):
            self.hand = hand
            self.mana_zone = mana_zone

    class FakeState:
        def __init__(self):
            self.active_player_id = 0
            self.current_phase = FakePhase('Phase.MAIN')
            self.players = [FakePlayer([FakeCardInstance('c1', 10), FakeCardInstance('c2', 11)], [types.SimpleNamespace(is_tapped=False)]),
                            FakePlayer([], [])]

    state = FakeState()

    # card_db as Python dict: c1 affordable, c2 too expensive
    card_db = {'c1': {'cost': 1}, 'c2': {'cost': 99}}

    cmds = commands.generate_legal_commands(state, card_db)

    # Ensure at least one returned command corresponds to a PLAY_CARD from fallback
    found_play = False
    for w in cmds:
        underlying = getattr(w, '_action', None)
        if underlying is None:
            continue
        t = getattr(underlying, 'type', None)
        if t == ActionType.PLAY_CARD:
            found_play = True
            break

    assert found_play, f"Expected PLAY_CARD in fallback commands, got: {cmds}"
