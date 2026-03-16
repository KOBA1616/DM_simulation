#pragma once
#include <string>
#include <vector>
#include <optional>
#include "core/card_def.hpp"

namespace dm::engine {

    struct PaymentPlan {
        int base_cost = 0;
        int adjusted_after_passive = 0;
        int total_passive_reduction = 0;
        std::vector<std::string> passive_ids; // use CostReductionDef.id when available (fall back to name)
        std::optional<std::string> active_reduction_id;
        std::optional<int> active_units;
        int active_reduction_amount = 0;
        int final_cost = 0;
    };

    // Prototype evaluator: produce a PaymentPlan from CardDefinition.
    // This is a conservative, engine-side prototype that mirrors the Python prototype
    // implemented in dm_toolkit.payment.evaluate_cost. Detailed semantics (unit costs,
    // reduction_per_unit, amount semantics) must be aligned with JSON schema.
    PaymentPlan evaluate_cost(const dm::core::CardDefinition& card_def, int units = 1, const std::optional<std::string>& active_name = std::nullopt, const std::optional<int>& active_units = std::nullopt);

}
