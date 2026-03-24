#include "payment_plan.hpp"
#include <algorithm>

namespace dm::engine {

    static int compute_static_cost_modifier_reduction(const dm::core::ModifierDef& mod_def) {
        if (mod_def.type != dm::core::ModifierType::COST_MODIFIER) return 0;

        // Legacy/default mode: FIXED uses `value` directly.
        if (mod_def.value_mode.empty() || mod_def.value_mode == "FIXED") {
            return std::max(0, mod_def.value);
        }

        // STAT_SCALED mode: PaymentPlan prototype has no runtime GameState, so stat_value defaults to 0.
        // This keeps the formula contract aligned while avoiding context guessing.
        if (mod_def.value_mode == "STAT_SCALED") {
            const int stat_value = 0;
            const int per_value = mod_def.per_value;
            const int min_stat = mod_def.min_stat;

            if (per_value <= 0) return 0;
            int calculated = std::max(0, stat_value - min_stat + 1) * per_value;
            if (mod_def.max_reduction.has_value()) {
                calculated = std::min(calculated, mod_def.max_reduction.value());
            }
            return std::max(0, calculated);
        }

        return 0;
    }

    PaymentPlan evaluate_cost(const dm::core::CardDefinition& card_def, int units, const std::optional<std::string>& active_name, const std::optional<int>& active_units) {
        PaymentPlan plan;
        plan.base_cost = card_def.cost;

        int total_passive = 0;
        std::optional<int> floor_min;

        for (const auto& cr : card_def.cost_reductions) {
            if (cr.type != dm::core::ReductionType::PASSIVE) continue;
            // conservative: use reduction_amount field
            total_passive += cr.reduction_amount;
            if (cr.min_mana_cost > 0) {
                if (!floor_min.has_value()) floor_min = cr.min_mana_cost;
                else floor_min = std::max(floor_min.value(), cr.min_mana_cost);
            }
            // Prefer id if present, otherwise fall back to name for legacy data
            if (!cr.id.empty()) plan.passive_ids.push_back(cr.id);
            else if (!cr.name.empty()) plan.passive_ids.push_back(cr.name);
        }

        int adjusted = std::max(plan.base_cost - total_passive, 0);
        if (floor_min.has_value()) adjusted = std::max(adjusted, floor_min.value());

        plan.adjusted_after_passive = adjusted;
        plan.total_passive_reduction = total_passive;

        // 再発防止: 合成順序は PASSIVE -> STATIC -> ACTIVE を厳守する。
        // ここでは STATIC(COST_MODIFIER) を PASSIVE 後に適用し、ACTIVE は最後に適用する。
        int static_total_reduction = 0;
        for (const auto& mod_def : card_def.static_abilities) {
            static_total_reduction += compute_static_cost_modifier_reduction(mod_def);
        }
        int adjusted_after_static = std::max(plan.adjusted_after_passive - static_total_reduction, 0);

        // Active: look up by name (temporary until id exists)
        if (active_name.has_value()) {
            int red_amt = 0;
            std::optional<int> active_floor;
            std::optional<std::string> matched_id;
            for (const auto& cr : card_def.cost_reductions) {
                if (cr.type != dm::core::ReductionType::ACTIVE_PAYMENT) continue;
                // match by id first, then by name for backward compatibility
                if ((!cr.id.empty() && cr.id == active_name.value()) || (!cr.name.empty() && cr.name == active_name.value())) {
                    red_amt = cr.reduction_amount; // conservative
                    if (cr.min_mana_cost > 0) active_floor = cr.min_mana_cost;
                    matched_id = !cr.id.empty() ? std::optional<std::string>(cr.id) : std::optional<std::string>(cr.name);
                    break;
                }
            }
            plan.active_reduction_amount = red_amt;
            plan.final_cost = std::max(adjusted_after_static - red_amt, 0);
            if (active_floor.has_value()) plan.final_cost = std::max(plan.final_cost, active_floor.value());
            plan.active_reduction_id = matched_id ? matched_id : active_name; // prefer canonical id when found
            if (active_units.has_value()) plan.active_units = active_units;
        } else {
            plan.final_cost = adjusted_after_static;
        }

        return plan;
    }

}
