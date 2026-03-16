#include <gtest/gtest.h>
#include "engine/systems/mechanics/payment_plan.hpp"
#include "core/card_json_types.hpp"

using namespace dm::engine;
using namespace dm::core;

TEST(PaymentPlanTest, PassiveReductionSum) {
    CardDefinition def;
    def.cost = 5;

    CostReductionDef cr1;
    cr1.type = ReductionType::PASSIVE;
    cr1.reduction_amount = 2;
    cr1.id = "p1";
    cr1.name = "p1";

    CostReductionDef cr2;
    cr2.type = ReductionType::PASSIVE;
    cr2.reduction_amount = 1;
    cr2.id = "p2";
    cr2.name = "p2";

    def.cost_reductions.push_back(cr1);
    def.cost_reductions.push_back(cr2);

    PaymentPlan plan = evaluate_cost(def, 1, std::nullopt, std::nullopt);
    EXPECT_EQ(plan.total_passive_reduction, 3);
    EXPECT_EQ(plan.adjusted_after_passive, 2);
    EXPECT_EQ(plan.final_cost, 2);
}

TEST(PaymentPlanTest, ActiveReductionApplied) {
    CardDefinition def;
    def.cost = 6;

    CostReductionDef passive;
    passive.type = ReductionType::PASSIVE;
    passive.reduction_amount = 1;
    passive.id = "p1";
    passive.name = "p1";

    CostReductionDef active;
    active.type = ReductionType::ACTIVE_PAYMENT;
    active.reduction_amount = 4; // simplified prototype semantics
    active.min_mana_cost = 1;
    active.id = "a1";
    active.name = "a1";

    def.cost_reductions.push_back(passive);
    def.cost_reductions.push_back(active);

    PaymentPlan plan = evaluate_cost(def, 1, std::optional<std::string>("a1"), std::optional<int>(1));
    EXPECT_EQ(plan.total_passive_reduction, 1);
    EXPECT_EQ(plan.active_reduction_amount, 4);
    EXPECT_EQ(plan.final_cost, 1); // 6 -1 -4 =1, floor enforces >=1
}
