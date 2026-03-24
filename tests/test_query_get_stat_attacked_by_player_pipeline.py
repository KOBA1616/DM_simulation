from pathlib import Path


def test_card_stats_has_player_attacked_array():
    p = Path("src/core/card_stats.hpp")
    src = p.read_text(encoding="utf-8")
    assert "attacked_this_turn_by_player" in src, "card_stats.hpp に attacked_this_turn_by_player がありません"


def test_pipeline_supports_my_opp_attacked_stat():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert 'my_stat == "ATTACKED_THIS_TURN"' in src or 'MY_ATTACKED_THIS_TURN' in src, "pipeline に MY_ または ATTACKED_THIS_TURN の MY_ ハンドラがありません"
    assert 'opp_stat == "ATTACKED_THIS_TURN"' in src or 'OPPONENT_ATTACKED_THIS_TURN' in src, "pipeline に OPPONENT_ の ATTACKED ハンドラがありません"
