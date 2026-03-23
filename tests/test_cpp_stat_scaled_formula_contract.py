from pathlib import Path


CPP_PATH = Path("src/engine/systems/effects/continuous_effect_system.cpp")


def test_cpp_stat_scaled_branch_has_contract_formula():
    src = CPP_PATH.read_text(encoding="utf-8")

    # Contract: C++ must implement STAT_SCALED branch.
    assert 'value_mode == "STAT_SCALED"' in src

    # Contract formula parity with Python implementation.
    assert 'std::max(0, stat_val - min_stat + 1) * per_value' in src
    assert 'mod_def.max_reduction.has_value()' in src
    assert 'std::min(calculated, mod_def.max_reduction.value())' in src


def test_cpp_stat_scaled_accepts_canonical_stat_aliases():
    src = CPP_PATH.read_text(encoding="utf-8")

    # Regression guard: editor/python tests commonly use upper snake case aliases.
    required_aliases = [
        '"SUMMON_COUNT_THIS_TURN"',
        '"CREATURES_PLAYED"',
    ]
    missing = [a for a in required_aliases if a not in src]
    assert not missing, f"C++ STAT_SCALED alias mapping missing: {missing}"
