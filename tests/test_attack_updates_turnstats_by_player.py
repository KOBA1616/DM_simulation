from pathlib import Path


def test_commands_updates_attacked_by_player_present():
    p = Path("src/engine/infrastructure/commands/definitions/commands.cpp")
    src = p.read_text(encoding="utf-8")
    assert 'attacked_this_turn_by_player' in src, "commands.cpp に attacked_this_turn_by_player の更新がありません"
    assert 'previous_turn_stats = state.turn_stats' in src, "commands.cpp に previous_turn_stats の保存ロジックがありません"
