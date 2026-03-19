from pathlib import Path


def test_set_attack_source_increments_attacked_stat():
    p = Path("src/engine/infrastructure/commands/definitions/commands.cpp")
    src = p.read_text(encoding="utf-8")

    # Ensure SET_ATTACK_SOURCE handling exists
    idx = src.find("case FlowType::SET_ATTACK_SOURCE:")
    assert idx != -1, "SET_ATTACK_SOURCE の処理が commands.cpp に見つかりません"

    # Ensure attacked_this_turn is updated within the handler
    handler_block = src[idx: idx + 800]
    assert "attacked_this_turn" in handler_block, "SET_ATTACK_SOURCE ハンドラに attacked_this_turn の更新がありません"
