from pathlib import Path


SPEC_PATH = Path("docs/cost_reduction_spec.md")


def test_cost_reduction_spec_documents_current_contracts() -> None:
    src = SPEC_PATH.read_text(encoding="utf-8")

    # Composition order contract aligned with implementation.
    assert "PASSIVE -> STATIC -> ACTIVE" in src

    # Rollout/rollback contract for STAT_SCALED.
    assert "STAT_SCALED_ENABLED=1|0" in src

    # Conflict severity contract aligned with editor save behavior.
    assert "detect_passive_static_conflicts" in src
    assert "ERROR" in src and "WARNING" in src

    # Formula parity contract.
    assert "reduction = min(max_reduction, max(0, stat_value - min_stat + 1) * per_value)" in src
