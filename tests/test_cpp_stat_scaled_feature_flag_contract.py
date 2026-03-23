from pathlib import Path


CPP_PATH = Path("src/engine/systems/effects/continuous_effect_system.cpp")


def test_cpp_stat_scaled_has_feature_flag_guard() -> None:
    src = CPP_PATH.read_text(encoding="utf-8")

    # RED/GREEN contract: C++ runtime must support staged rollback for STAT_SCALED.
    assert 'STAT_SCALED_ENABLED' in src
    assert 'is_stat_scaled_enabled()' in src
    assert 'mod_def.value_mode == "STAT_SCALED" && is_stat_scaled_enabled()' in src
