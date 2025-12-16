#include "mana_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/game_command/commands.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <numeric>
#include <functional>

namespace dm::engine {

    using namespace dm::core;

    int ManaSystem::get_adjusted_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def) {
        int cost = card_def.cost;

        if (cost <= 0) return 0;

        for (const auto& mod : game_state.active_modifiers) {
            if (mod.controller != player.id) continue;

            CardInstance dummy_inst;
            dummy_inst.card_id = card_def.id;

            if (TargetUtils::is_valid_target(dummy_inst, card_def, mod.condition_filter, game_state, player.id, player.id)) {
                cost -= mod.reduction_amount;
            }
        }

        if (cost < 1) cost = 1;
        if (card_def.cost > 0 && cost < 1) cost = 1;

        return cost;
    }

    // Internal helper with DB access
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

        // Pre-fetch defs for untapped cards
        std::vector<const CardDefinition*> mana_defs(mana_zone.size(), nullptr);
        for(size_t i=0; i<mana_zone.size(); ++i) {
             if (!mana_zone[i].is_tapped && card_db.count(mana_zone[i].card_id)) {
                 mana_defs[i] = &card_db.at(mana_zone[i].card_id);
             }
        }

        // Recursive lambda
        std::function<bool(size_t)> solve = [&](size_t req_idx) -> bool {
            if (req_idx >= colored_reqs.size()) {
                // Requirements met. Now fill remaining cost with any unused untapped cards.
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
                return false; // Not enough filler
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
            // Just fill
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

    // Stub for the hpp declaration
    std::vector<int> ManaSystem::solve_payment(const std::vector<dm::core::CardInstance>& /*mana_zone*/,
                                              const std::vector<dm::core::Civilization>& /*required_civs*/,
                                              int /*total_cost*/) {
        return {};
    }

    bool ManaSystem::can_pay_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = get_adjusted_cost(game_state, player, card_def);
        if (cost <= 0) return true; // Free after reductions
        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        return !indices.empty();
    }

    bool ManaSystem::can_pay_cost(const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = card_def.cost;
        if (cost <= 0) return true; // Zero or negative cost is always payable without mana
        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        return !indices.empty();
    }

    bool ManaSystem::auto_tap_mana(GameState& game_state, Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = get_adjusted_cost(game_state, player, card_def);

        if (cost <= 0) {
            game_state.turn_stats.played_without_mana = true;
            return true;
        }

        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        if (indices.empty()) return false;

        // Apply tap via Command
        for (int idx : indices) {
            // Must find instance_id
            int iid = player.mana_zone[idx].instance_id;
            auto cmd = std::make_unique<game_command::MutateCommand>(iid, game_command::MutateCommand::MutationType::TAP);
            game_state.execute_command(std::move(cmd));
        }

        // Paid mana exists, so not "played without mana" for this path

        return true;
    }

    bool ManaSystem::auto_tap_mana(GameState& game_state, Player& player, const CardDefinition& card_def, int cost_override, const std::map<CardID, CardDefinition>& card_db) {
        int cost = cost_override;

        if (cost <= 0) {
            game_state.turn_stats.played_without_mana = true;
            return true;
        }

        auto indices = solve_payment_internal(player.mana_zone, card_def.civilizations, cost, card_db);
        if (indices.empty()) return false;

        for (int idx : indices) {
            // Must find instance_id
            int iid = player.mana_zone[idx].instance_id;
            auto cmd = std::make_unique<game_command::MutateCommand>(iid, game_command::MutateCommand::MutationType::TAP);
            game_state.execute_command(std::move(cmd));
        }

        return true;
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

    void ManaSystem::untap_all(GameState& game_state, Player& player) {
        // Collect instances to untap to avoid iterator invalidation issues if any
        std::vector<int> to_untap;

        for (const auto& card : player.mana_zone) {
            if (card.is_tapped) to_untap.push_back(card.instance_id);
        }
        for (const auto& card : player.battle_zone) {
            if (card.is_tapped) to_untap.push_back(card.instance_id);
        }

        for (int iid : to_untap) {
             auto cmd = std::make_unique<game_command::MutateCommand>(iid, game_command::MutateCommand::MutationType::UNTAP);
             game_state.execute_command(std::move(cmd));
        }
    }

    void ManaSystem::untap_all(Player& player) {
        for (auto& card : player.mana_zone) {
            card.is_tapped = false;
        }
        for (auto& card : player.battle_zone) {
            card.is_tapped = false;
        }
    }

    int ManaSystem::get_projected_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def) {
        return get_adjusted_cost(game_state, player, card_def);
    }

    int ManaSystem::get_usable_mana_count(const GameState& game_state, PlayerID player_id, const std::vector<Civilization>& required_civs, const std::map<CardID, CardDefinition>& card_db) {
        const auto& player = game_state.players[player_id];
        int count = 0;
        for (const auto& card : player.mana_zone) {
            if (!card.is_tapped) count++;
        }

        // Perform strict check if there are required civilizations
        // If we have N untapped cards, can they satisfy required_civs?
        // Note: this function returns 'count' (max mana), but implies valid mana.
        // If requirements cannot be met even using ALL untapped cards, then usable mana is effectively 0 for this card.
        // (Or more accurately, we return 0 to indicate "cannot play").
        // We reuse solve_payment_internal logic: can we pay 'count' (using all untapped) given requirements?
        // Wait, solve_payment_internal checks if we can pay EXACTLY cost.
        // Here we just want to know if we meet requirements.
        // Actually, if we use solve_payment_internal with cost=0? No.
        // We want to check if the SET of untapped cards covers 'required_civs'.

        std::vector<Civilization> colored_reqs;
        for (auto c : required_civs) {
            if (c != Civilization::NONE && c != Civilization::ZERO) {
                colored_reqs.push_back(c);
            }
        }

        if (colored_reqs.empty()) return count;

        // Check if we can satisfy colored_reqs using untapped cards
        // We can reuse solve_payment_internal logic but with cost = colored_reqs.size() (minimal payment)
        // If we can pay the minimal cost satisfying colors, then we definitely have valid mana.
        // But wait, get_usable_mana_count returns the TOTAL amount available.
        // So: return count IF minimal requirements are met. Else 0.

        int min_cost = (int)colored_reqs.size();
        if (min_cost > count) return 0; // Not enough cards to cover colors

        auto indices = solve_payment_internal(player.mana_zone, required_civs, min_cost, card_db);
        if (indices.empty()) return 0;

        return count;
    }

}
