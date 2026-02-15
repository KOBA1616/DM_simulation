#include "cost_payment_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "mana_system.hpp"
#include <algorithm>

namespace dm::engine {

    using namespace dm::core;

    int CostPaymentSystem::calculate_max_units(const GameState& state,
                                               PlayerID player_id,
                                               const CostReductionDef& reduction,
                                               const std::map<CardID, CardDefinition>& card_db) {
        if (reduction.type != ReductionType::ACTIVE_PAYMENT) {
            return 0;
        }

        const auto& unit_cost = reduction.unit_cost;
        if (unit_cost.type == CostType::TAP_CARD) {
            // Count cards matching filter that are NOT tapped
            int count = 0;
            // Zones: BATTLE_ZONE is primary for Hyper Energy
            // We need to iterate over zones specified in the filter
             // FilterDef defaults: owner=SELF if not specified for cost payment usually?
             // Or explicitly defined. Let's assume FilterDef defines it.
             // If filter.owner is missing, for costs, it defaults to SELF usually.

            // We create a modified filter that forces is_tapped = false for counting valid tap targets
            FilterDef check_filter = unit_cost.filter;
            check_filter.is_tapped = false;

            // Use TargetUtils to find candidates
            // Wait, TargetUtils::collect_valid_targets might return all valid ones.
            // But we just need a count.

            // To use TargetUtils, we need to adapt since it works with EffectDef/Action usually.
            // Or we can manually iterate zones if simple.
            // Let's manually iterate for safety and control.

            const Player& player = state.players[player_id];

            // Handle Battle Zone
            bool check_bz = false;
             for (const auto& z : unit_cost.filter.zones) {
                 if (z == "BATTLE_ZONE") check_bz = true;
             }

            if (check_bz) {
                for (const auto& instance : player.battle_zone) {
                    // Must be untapped to pay tap cost
                    if (instance.is_tapped) continue;

                    // Check filter using TargetUtils
                    // We need CardData/CardDefinition.
                    if (card_db.find(instance.card_id) == card_db.end()) continue;
                    const auto& def = card_db.at(instance.card_id);

                    // We need to construct a CardData wrapper or use TargetUtils::is_valid_target taking CardDefinition
                    // TargetUtils::is_valid_target accepts (instance, def, filter, game_state)
                    // Signature: is_valid_target(instance, def, filter, state, source_controller, card_controller, ignore_passives)

                    // Identify controller.
                    // For cost payment, the source controller is the player paying (player_id).
                    // The card controller is who controls the potential target (usually same player for Hyper Energy).
                    // We check instance controller.

                    PlayerID card_owner = player_id; // Default assumption for owned zone
                    if (instance.instance_id >= 0 && (size_t)instance.instance_id < state.card_owner_map.size()) {
                        card_owner = state.get_card_owner(instance.instance_id);
                    }

                    if (TargetUtils::is_valid_target(instance, def, unit_cost.filter, state, player_id, card_owner, false)) {
                        count++;
                    }
                }
            }

            // Handle other zones if needed (Mana Zone tap?)

            return count;
        }

        // Other active payment types not yet fully implemented
        return 0;
    }

    int CostPaymentSystem::calculate_potential_reduction(const GameState& state,
                                                         PlayerID player_id,
                                                         const CostReductionDef& reduction,
                                                         const std::map<CardID, CardDefinition>& card_db) {
        if (reduction.type == ReductionType::PASSIVE) {
             // Not implemented here, handled by ManaSystem usually, or could be moved here.
             // For now focus on ACTIVE_PAYMENT
             return 0;
        }

        int units = calculate_max_units(state, player_id, reduction, card_db);
        if (reduction.max_units != -1) {
            units = std::min(units, reduction.max_units);
        }

        return units * reduction.reduction_amount;
    }

    bool CostPaymentSystem::can_pay_cost(const GameState& state,
                                         PlayerID player_id,
                                         const CardDefinition& card,
                                         const std::map<CardID, CardDefinition>& card_db) {

        // 1. Get Base Cost
        int cost = card.cost;
        (void)cost;

        // 2. Apply Standard (Passive) Reductions via ManaSystem (existing logic)
        // We use ManaSystem::get_adjusted_cost to get the cost after passive modifiers.
        // Assuming ManaSystem is accessible or we replicate the logic.
        // Actually, ManaSystem::get_adjusted_cost requires GameState.
        // It accounts for "active_modifiers" (CostModifier) in GameState.
        // It does NOT account for the card's OWN internal active reductions (like Hyper Energy).

        int adjusted_cost = ManaSystem::get_adjusted_cost(state, state.players[player_id], card);

        // 3. Check for Active Reductions (e.g. Hyper Energy)
        int potential_active_reduction = 0;
        int min_mana_cost_limit = 0; // Default lower bound is 0 or 1?
        (void)min_mana_cost_limit;
        // ManaSystem enforces min cost 1 usually, unless g_zero etc.
        // Hyper Energy sets min_mana_cost to 0 in spec example.

        for (const auto& reduction : card.cost_reductions) {
            if (reduction.type == ReductionType::ACTIVE_PAYMENT) {
                int r = calculate_potential_reduction(state, player_id, reduction, card_db);
                if (r > 0) {
                     potential_active_reduction += r;
                     // Logic for multiple active reductions? Usually add up.
                     // Logic for min cost? Use the reduction's min_cost.
                     // If multiple reductions have different min costs, logic gets complex.
                     // Assume simplest case: apply reduction, clamp at min_mana_cost.
                     // But we perform the clamp at the end.
                     // Spec says: "min_mana_cost = 0" in example.
                     // If standard rules say min 1, but this says min 0.

                     // We should track the lowest allowed minimum.
                     // If any reduction allows going to 0, we allow it?
                     // Let's assume strict constraint: if reduction.min_mana_cost is 0, we can go to 0.
                }
            }
        }

        // 4. Determine final cost range
        // Lowest possible cost user can achieve
        int min_achievable_cost = adjusted_cost - potential_active_reduction;
        if (min_achievable_cost < 0) min_achievable_cost = 0;

        // 5. Check if we have enough mana to pay this minimum
        // ManaSystem::get_usable_mana_count
        int available_mana = ManaSystem::get_usable_mana_count(state, player_id, card.civilizations, card_db);

        return available_mana >= min_achievable_cost;
    }

    int CostPaymentSystem::execute_payment(GameState& state,
                                           PlayerID player_id,
                                           const CostReductionDef& reduction,
                                           int units,
                                           const std::map<CardID, CardDefinition>& card_db) {
        if (reduction.type != ReductionType::ACTIVE_PAYMENT || units <= 0) {
            return 0;
        }

        const auto& unit_cost = reduction.unit_cost;
        int units_paid = 0;

        if (unit_cost.type == CostType::TAP_CARD) {
            Player& player = state.players[player_id];

            // Collect valid candidates using the same logic as calculate_max_units
            std::vector<int> candidates;
            bool check_bz = false;
            for (const auto& z : unit_cost.filter.zones) {
                if (z == "BATTLE_ZONE") check_bz = true;
            }

            if (check_bz) {
                // We use indices to safely modify later, but battle_zone might change if events trigger?
                // Tapping usually doesn't trigger "On Tap" immediate destroy/move effects that invalidate iterators in standard loop,
                // but safe practice is to collect IDs/indices first.
                for (size_t i = 0; i < player.battle_zone.size(); ++i) {
                    const auto& instance = player.battle_zone[i];
                    if (instance.is_tapped) continue;

                    if (card_db.find(instance.card_id) == card_db.end()) continue;
                    const auto& def = card_db.at(instance.card_id);

                    PlayerID card_owner = player_id;
                    if (instance.instance_id >= 0 && (size_t)instance.instance_id < state.card_owner_map.size()) {
                        card_owner = state.get_card_owner(instance.instance_id);
                    }

                    if (TargetUtils::is_valid_target(instance, def, unit_cost.filter, state, player_id, card_owner, false)) {
                        candidates.push_back((int)i);
                    }
                }
            }

            // Greedy execution: Tap the first 'units' candidates.
            // A more complex system would allow user selection if needed.
            // For Hyper Energy, it's usually "Any N creatures".
            for (int idx : candidates) {
                if (units_paid >= units) break;
                player.battle_zone[idx].is_tapped = true;
                units_paid++;
            }
        }

        return units_paid * reduction.reduction_amount;
    }

}
