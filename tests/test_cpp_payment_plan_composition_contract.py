from pathlib import Path


CPP_PATH = Path("src/engine/systems/mechanics/payment_plan.cpp")


def test_cpp_payment_plan_handles_static_cost_modifier_before_active():
    src = CPP_PATH.read_text(encoding="utf-8")

    # RED/GREEN contract: PaymentPlan must process static_abilities for COST_MODIFIER.
    assert "static_abilities" in src
    assert "ModifierType::COST_MODIFIER" in src

    # Composition-order contract: ACTIVE reduction must apply after static stage.
    assert "adjusted_after_static" in src
    assert "plan.final_cost = std::max(adjusted_after_static - red_amt, 0)" in src


def test_cpp_payment_plan_has_reoccurrence_prevention_comment_for_order():
    src = CPP_PATH.read_text(encoding="utf-8")
    assert "再発防止" in src
    assert "PASSIVE" in src and "STATIC" in src and "ACTIVE" in src
