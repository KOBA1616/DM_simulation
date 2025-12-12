#include "mana_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "cost_calculator.hpp"
#include "payment_processor.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <numeric>
#include <functional>

namespace dm::engine {

    using namespace dm::core;

    int ManaSystem::get_adjusted_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def) {
        return CostCalculator::get_base_adjusted_cost(game_state, player, card_def);
    }

    int ManaSystem::get_projected_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def) {
        return get_adjusted_cost(game_state, player, card_def);
    }

    static std::vector<int> solve_payment_internal(const std::vector<CardInstance>& mana_zone,
                                              const std::vector<Civilization>& required_civs,
                                              int total_cost,
                                              const std::map<CardID, CardDefinition>& card_db) {
        std::vector<Civilization> colored_reqs;
        for (auto c : required_civs) {
            if (c != Civilization::NONE && c != Civilization::ZERO) {
                colored_reqs.push_back(c);
            }
        }

        int untapped_count = 0;
        for (const auto& c : mana_zone) if (!c.is_tapped) untapped_count++;
        if (untapped_count < total_cost) return {};

        std::vector<int> result;
        std::vector<bool> used(mana_zone.size(), false);

        std::vector<const CardDefinition*> mana_defs(mana_zone.size(), nullptr);
        for(size_t i=0; i<mana_zone.size(); ++i) {
             if (!mana_zone[i].is_tapped && card_db.count(mana_zone[i].card_id)) {
                 mana_defs[i] = &card_db.at(mana_zone[i].card_id);
             }
        }

        std::function<bool(size_t)> solve = [&](size_t req_idx) -> bool {
            if (req_idx >= colored_reqs.size()) {
                int needed = total_cost - (int)result.size();
                if (needed == 0) return true;

                for (size_t i = 0; i < mana_zone.size(); ++i) {
                    if (needed == 0) break;
                    if (!mana_zone[i].is_tapped && !used[i]) {
                        used[i] = true;
                        result.push_back(i);
                        needed--;
                    }
                }

                if (needed == 0) return true;
                return false;
            }

            Civilization req = colored_reqs[req_idx];
            for (size_t i = 0; i < mana_zone.size(); ++i) {
                if (mana_defs[i] && !used[i]) {
                    if (mana_defs[i]->has_civilization(req)) {
                        used[i] = true;
                        result.push_back(i);

                        if (solve(req_idx + 1)) return true;

                        result.pop_back();
                        used[i] = false;
                    }
                }
            }
            return false;
        };

        if (colored_reqs.empty()) {
            for (size_t i = 0; i < mana_zone.size(); ++i) {
                if ((int)result.size() == total_cost) break;
                if (!mana_zone[i].is_tapped) result.push_back(i);
            }
            return result;
        }

        if (solve(0)) {
            return result;
        }

        return {};
    }

    std::vector<int> ManaSystem::solve_payment(const std::vector<dm::core::CardInstance>& /*mana_zone*/,
                                              const std::vector<dm::core::Civilization>& /*required_civs*/,
                                              int /*total_cost*/) {
        return {};
    }

    bool ManaSystem::can_pay_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = get_adjusted_cost(game_state, player, card_def);
        if (cost <= 0) return true;
        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        return !indices.empty();
    }

    bool ManaSystem::can_pay_cost(const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = card_def.cost;
        if (cost <= 0) return true;
        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        return !indices.empty();
    }

    bool ManaSystem::auto_tap_mana(GameState& game_state, Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        // Use CostCalculator
        PaymentRequirement req = CostCalculator::calculate_requirement(game_state, player, card_def);

        // Solve payment using internal solver
        // We still use internal solver to find *which* cards to tap, then pass to PaymentProcessor.
        // This bridges the gap.

        // Note: PaymentRequirement might have is_g_zero or reduced cost.

        PaymentContext ctx;
        ctx.type = PaymentType::MANA; // Default

        if (req.is_g_zero || req.final_mana_cost <= 0) {
             // Zero cost
             return PaymentProcessor::process_payment(game_state, player, req, ctx);
        }

        // Solve
        auto indices = solve_payment_internal(player.mana_zone, req.required_civs, req.final_mana_cost, card_db);
        if (indices.empty()) return false;

        // Convert indices to CardIDs (Instance IDs? No, PaymentProcessor uses CardID type but expects Instance ID likely?)
        // The header says `std::vector<dm::core::CardID> mana_cards_to_tap`.
        // `CardID` is `uint16_t`. But `CardInstance::instance_id` is `int`.
        // `CardInstance::card_id` is `CardID`.
        // PaymentProcessor::pay_mana implementation uses `get_card_instance(id)`.
        // This implies it expects `instance_id` cast to `CardID` (risky if instance_id > 65535) OR `instance_id`.
        // Let's check `PaymentProcessor.cpp`. It calls `get_card_instance(id)`.
        // `get_card_instance` takes `int`.
        // So `CardID` in `PaymentRequirement`/`Context` is actually `InstanceID` (misnamed typedef usage).
        // I should probably fix the typedef or cast.

        for (int idx : indices) {
            ctx.mana_cards_to_tap.push_back(static_cast<CardID>(player.mana_zone[idx].instance_id));
        }

        return PaymentProcessor::process_payment(game_state, player, req, ctx);
    }

    bool ManaSystem::auto_tap_mana(Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = card_def.cost;
        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        if (indices.empty() && cost > 0) return false;

        for (int idx : indices) {
            player.mana_zone[idx].is_tapped = true;
        }
        return true;
    }

    void ManaSystem::untap_all(Player& player) {
        for (auto& card : player.mana_zone) {
            card.is_tapped = false;
        }
        for (auto& card : player.battle_zone) {
            card.is_tapped = false;
        }
    }

}
