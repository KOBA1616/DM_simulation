from pathlib import Path


def test_pipeline_executor_contains_attacked_handlers():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "ATTACKED_THIS_TURN" in src, "pipeline_executor.cpp に ATTACKED_THIS_TURN の扱いがありません"
    assert "attacked_this_turn" in src, "pipeline_executor.cpp に attacked_this_turn 参照がありません"
