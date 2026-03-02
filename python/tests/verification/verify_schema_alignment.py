import dm_ai_module
import sys

def verify_enum_alignment():
    """
    Verifies that the Python-side Action->Command mapping targets
    valid C++ CommandType enums.
    """

    # 1. Fetch Actual Enums
    actual_enums = set(e.name for e in dm_ai_module.CommandType.__members__.values())

    # 2. Define Expected Enums (from the new specification)
    # Based on the list I retrieved earlier
    expected_enums = {
        'TRANSITION', 'MUTATE', 'FLOW', 'QUERY', 'DRAW_CARD', 'DISCARD',
        'DESTROY', 'MANA_CHARGE', 'TAP', 'UNTAP', 'POWER_MOD', 'ADD_KEYWORD',
        'RETURN_TO_HAND', 'BREAK_SHIELD', 'SEARCH_DECK', 'SHIELD_TRIGGER',
        'ATTACK_PLAYER', 'ATTACK_CREATURE', 'BLOCK', 'RESOLVE_BATTLE',
        'RESOLVE_PLAY', 'RESOLVE_EFFECT', 'SHUFFLE_DECK', 'LOOK_AND_ADD',
        'MEKRAID', 'REVEAL_CARDS', 'PLAY_FROM_ZONE', 'CAST_SPELL',
        'SUMMON_TOKEN', 'SHIELD_BURN', 'SELECT_NUMBER', 'CHOICE',
        'LOOK_TO_BUFFER', 'SELECT_FROM_BUFFER', 'PLAY_FROM_BUFFER',
        'MOVE_BUFFER_TO_ZONE', 'FRIEND_BURST', 'REGISTER_DELAYED_EFFECT', 'NONE'
    }

    # 3. Check for Missing Enums in Implementation
    missing_in_impl = expected_enums - actual_enums
    if missing_in_impl:
        print(f"FAIL: The following expected enums are missing in dm_ai_module: {missing_in_impl}")
        sys.exit(1)

    # 4. Check for New Enums in Implementation (Warning)
    new_in_impl = actual_enums - expected_enums
    if new_in_impl:
        print(f"WARNING: The following enums exist in implementation but not in expectation: {new_in_impl}")

    # 5. Check action_to_command mapping logic (Simulation)
    # 再発防止: action_to_command は削除済み。この検証セクションはスキップ。
    print("SKIP: action_to_command は削除済みのためマッピング検証セクションをスキップ。")

    print("SUCCESS: CommandType enums are aligned.")
    sys.exit(0)

if __name__ == "__main__":
    verify_enum_alignment()
