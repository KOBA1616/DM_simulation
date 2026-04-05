#include "cost_payment_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "mana_system.hpp"
#include "payment_plan.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
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

                    // Check filter using dm::engine::utils::TargetUtils
                    if (card_db.find(instance.card_id) == card_db.end()) continue;
                    const auto& def = card_db.at(instance.card_id);

                    PlayerID card_owner = player_id; // Default assumption for owned zone
                    if (instance.instance_id >= 0 && (size_t)instance.instance_id < state.card_owner_map.size()) {
                        card_owner = state.get_card_owner(instance.instance_id);
                    }

                    if (dm::engine::utils::TargetUtils::is_valid_target(instance, def, unit_cost.filter, state, player_id, card_owner, false)) {
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

        // 1. Get base cost adjusted by passive modifiers
        int adjusted_cost = ManaSystem::get_adjusted_cost(state, state.players[player_id], card);

        // 2. Consider ACTIVE_PAYMENT reductions.
        // 再発防止: legal command 生成中に evaluate_cost() を呼ぶ経路で AV が再発したため、
        // ここでは ACTIVE_PAYMENT を直接計算し、検証用プロトタイプ評価には依存しない。
        int min_achievable_cost = adjusted_cost;

        for (const auto& reduction : card.cost_reductions) {
            if (reduction.type != ReductionType::ACTIVE_PAYMENT) continue;

            int max_units = calculate_max_units(state, player_id, reduction, card_db);
            if (max_units <= 0) continue;

            if (reduction.max_units != -1) {
                max_units = std::min(max_units, reduction.max_units);
            }
            if (max_units <= 0) continue;

            int candidate_cost = adjusted_cost - (max_units * reduction.reduction_amount);
            if (reduction.min_mana_cost > 0) {
                candidate_cost = std::max(candidate_cost, reduction.min_mana_cost);
            }
            candidate_cost = std::max(candidate_cost, 0);

            if (candidate_cost < min_achievable_cost) {
                min_achievable_cost = candidate_cost;
            }
        }

        if (min_achievable_cost < 0) min_achievable_cost = 0;

        // 3. Check available usable mana
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

                    if (dm::engine::utils::TargetUtils::is_valid_target(instance, def, unit_cost.filter, state, player_id, card_owner, false)) {
                        candidates.push_back((int)i);
                    }
                }
            }

            // Greedy execution: Tap the first 'units' candidates.
            // A more complex system would allow user selection if needed.
            // For Hyper Energy, it's usually "Any N creatures".
            for (int idx : candidates) {
                if (units_paid >= units) break;
                // Use command execution to record mutation and keep history/undo consistent
                int iid = player.battle_zone[idx].instance_id;
                auto tap_cmd = std::make_shared<game_command::MutateCommand>(iid, game_command::MutateCommand::MutationType::TAP);
                state.execute_command(std::move(tap_cmd));
                units_paid++;
            }
        }

        return units_paid * reduction.reduction_amount;
    }

}
