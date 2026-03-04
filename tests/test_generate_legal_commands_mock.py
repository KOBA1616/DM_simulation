import os
import sys
import types

from dm_toolkit import commands

# 再発防止: sys.modules['dm_ai_module'] を差し替えたら必ず復元すること。
# 復元しないと後続テストが偽のモジュールを使用してクラッシュする（テスト汚染）。
# 再発防止: ActionGenerator は削除済み。IntentGenerator を使用すること。
# 再発防止: PLAY_CARD は CommandType に存在しない。PLAY_FROM_ZONE を使用すること。

def test_play_heuristic_with_mocked_dm_ai_module():
    # Prepare a fake dm_ai_module to avoid loading native extensions.
    fake = types.SimpleNamespace()

    class CommandType:
        PASS = 'PASS'
        MANA_CHARGE = 'MANA_CHARGE'
        PLAY_FROM_ZONE = 'PLAY_FROM_ZONE'

    # IntentGenerator returns no native actions to force Python fallback.
    class IntentGenerator:
        @staticmethod
        def generate_legal_commands(state, card_db=None):
            return []

    class PhaseManager:
        @staticmethod
        def next_phase(state, card_db=None):
            return None

    fake.CommandType = CommandType
    fake.IntentGenerator = IntentGenerator
    fake.PhaseManager = PhaseManager

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

    # sys.modules を差し替えて Python フォールバックを強制し、テスト後に必ず復元する
    original_module = sys.modules.get('dm_ai_module')
    original_native = os.environ.get('DM_DISABLE_NATIVE')
    try:
        sys.modules['dm_ai_module'] = fake
        os.environ['DM_DISABLE_NATIVE'] = '1'  # Python フォールバックを有効化

        cmds = commands.generate_legal_commands(state, card_db)

        # Ensure at least one returned command corresponds to PLAY_FROM_ZONE from fallback
        # Python フォールバックは dict {'type': 'PLAY_FROM_ZONE', ...} を _action として返す
        found_play = False
        for w in cmds:
            underlying = getattr(w, '_action', None)
            if underlying is None:
                underlying = w.to_dict() if hasattr(w, 'to_dict') else None
            if underlying is None:
                continue
            t = underlying.get('type') if isinstance(underlying, dict) else getattr(underlying, 'type', None)
            if t == 'PLAY_FROM_ZONE':
                found_play = True
                break

        assert found_play, f"Expected PLAY_FROM_ZONE in fallback commands, got: {cmds}"
    finally:
        # 再発防止: 必ず元のモジュールに復元する
        if original_module is not None:
            sys.modules['dm_ai_module'] = original_module
        else:
            sys.modules.pop('dm_ai_module', None)
        if original_native is None:
            os.environ.pop('DM_DISABLE_NATIVE', None)
        else:
            os.environ['DM_DISABLE_NATIVE'] = original_native
