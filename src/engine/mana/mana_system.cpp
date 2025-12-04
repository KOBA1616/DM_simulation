#include "mana_system.hpp"
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../card_system/target_utils.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    int ManaSystem::get_adjusted_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def) {
        int cost = card_def.cost;

        // If base cost is 0, it stays 0 (e.g. dummy test card)
        if (cost <= 0) return 0;

        for (const auto& mod : game_state.active_modifiers) {
            if (mod.controller != player.id) continue;

            // Use TargetUtils with a dummy CardInstance (since Cost is checked for CardDefinition)
            // Or adapt TargetUtils to work without instance for static checks?
            // TargetUtils::is_valid_target takes CardInstance & CardDefinition.
            // Some checks (is_tapped) require instance.
            // Cost modifiers usually don't depend on tapped state of the card being played (since it's in hand/mana).
            // We pass a dummy instance.
            CardInstance dummy_inst;
            dummy_inst.card_id = card_def.id;
            // The instance ID and other state are irrelevant for cost check usually.

            // source_controller is player.id
            // card_controller is also player.id (it's our card we want to play)
            if (TargetUtils::is_valid_target(dummy_inst, card_def, mod.condition_filter, player.id, player.id)) {
                cost -= mod.reduction_amount;
            }
        }

        if (cost < 1) cost = 1; // Minimum cost is 1 (except if base was 0, handled above? No, wait.)

        // Re-checking logic:
        // Regular cost reduction rules say min cost is 1.
        // But if the card ITSELF has cost 0 (like a token or dummy), it should be 0.
        // Also G-Zero makes it 0.
        // Since we don't have G-Zero fully here yet, let's respect base cost.
        if (card_def.cost > 0 && cost < 1) cost = 1;

        return cost;
    }

    bool ManaSystem::can_pay_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = get_adjusted_cost(game_state, player, card_def);

        int available_mana = 0;
        Civilization available_civs = Civilization::NONE;

        for (const auto& card : player.mana_zone) {
            if (!card.is_tapped) {
                available_mana++;
                // Look up civilization from DB
                if (card_db.count(card.card_id)) {
                    available_civs = available_civs | card_db.at(card.card_id).civilization;
                }
            }
        }

        // Check cost
        if (available_mana < cost) {
            return false;
        }

        // Check civilization
        if (card_def.civilization != Civilization::ZERO) {
             if ((available_civs & card_def.civilization) == Civilization::NONE) {
                 return false;
             }
        }

        return true;
    }

    bool ManaSystem::can_pay_cost(const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        // Fallback for tests or legacy calls: assume no modifiers, so cost is unmodified
        int cost = card_def.cost;

        int available_mana = 0;
        Civilization available_civs = Civilization::NONE;

        for (const auto& card : player.mana_zone) {
            if (!card.is_tapped) {
                available_mana++;
                if (card_db.count(card.card_id)) {
                    available_civs = available_civs | card_db.at(card.card_id).civilization;
                }
            }
        }

        if (available_mana < cost) return false;

        if (card_def.civilization != Civilization::ZERO) {
             if ((available_civs & card_def.civilization) == Civilization::NONE) {
                 return false;
             }
        }

        return true;
    }

    bool ManaSystem::auto_tap_mana(GameState& game_state, Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        if (!can_pay_cost(game_state, player, card_def, card_db)) {
            return false;
        }

        int cost_remaining = get_adjusted_cost(game_state, player, card_def);
        int paid_mana = 0;
        bool civ_requirement_met = (card_def.civilization == Civilization::ZERO);

        // First pass: Try to tap a card that satisfies the civilization requirement
        if (!civ_requirement_met) {
            for (auto& card : player.mana_zone) {
                if (!card.is_tapped) {
                    const auto& mana_card_def = card_db.at(card.card_id);
                    if ((mana_card_def.civilization & card_def.civilization) != Civilization::NONE) {
                        card.is_tapped = true;
                        cost_remaining--;
                        paid_mana++;
                        civ_requirement_met = true;
                        break;
                    }
                }
            }
        }

        // Second pass: Tap remaining necessary mana
        if (cost_remaining > 0) {
            for (auto& card : player.mana_zone) {
                if (cost_remaining == 0) break;
                if (!card.is_tapped) {
                    card.is_tapped = true;
                    cost_remaining--;
                    paid_mana++;
                }
            }
        }

        // Check if played without mana (Meta Counter logic)
        // If paid_mana is 0, it means it was free (e.g. G-Zero, or heavily reduced to 0 but minimum cost is 1 so this branch handles purely 0 payment)
        // Note: get_adjusted_cost enforces minimum 1 unless it's a special mechanic not using cost (like G-Zero).
        // However, if cost_remaining was 0 initially (due to some other effect not yet implemented, or G-Zero overriding logic), paid_mana would be 0.
        // Currently get_adjusted_cost ensures >= 1.
        // But if we have G-Zero in the future, cost_remaining could be 0.
        // The requirement says: "If paid_mana (tapped mana) is 0, set played_without_mana = true".
        // Even if cost was reduced to 1, paid_mana would be 1.
        // So this only triggers if truly 0 mana was tapped.
        if (paid_mana == 0) {
            game_state.turn_stats.played_without_mana = true;
        }

        // Decrement turns_remaining for one-shot modifiers (turns_remaining == 1)
        // Wait, "This turn only" modifiers usually last the whole turn (turns_remaining = 1),
        // they don't get consumed by one usage unless specified (e.g. Fairy Gift).
        // The spec said "turn_limit".
        // Fairy Gift: "The next creature you summon this turn costs 3 less". This is usage based.
        // Cocco Lupia: "Your dragons cost 2 less". This is continuous.

        // If we have usage-based modifiers, we need to handle them here.
        // For now, let's assume modifiers are continuous for the turn (Cocco Lupia style).
        // If we implement Fairy Gift, we need a "usage count" in CostModifier.

        return true;
    }

    bool ManaSystem::auto_tap_mana(Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        // Legacy call - passes through to logic with no game state (no modifiers)
        if (!can_pay_cost(player, card_def, card_db)) {
            return false;
        }

        int cost_remaining = card_def.cost;
        bool civ_requirement_met = (card_def.civilization == Civilization::ZERO);

        if (!civ_requirement_met) {
            for (auto& card : player.mana_zone) {
                if (!card.is_tapped) {
                    const auto& mana_card_def = card_db.at(card.card_id);
                    if ((mana_card_def.civilization & card_def.civilization) != Civilization::NONE) {
                        card.is_tapped = true;
                        cost_remaining--;
                        civ_requirement_met = true;
                        break;
                    }
                }
            }
        }

        if (cost_remaining > 0) {
            for (auto& card : player.mana_zone) {
                if (cost_remaining == 0) break;
                if (!card.is_tapped) {
                    card.is_tapped = true;
                    cost_remaining--;
                }
            }
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

    int ManaSystem::get_projected_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def) {
        // [PLAN-002] Virtual Cost Calculation
        // For now, this is identical to get_adjusted_cost which handles Base -> Passive Modifiers.
        // Future: Add Active Reduction logic here (e.g., if card has Hyper Energy, check if it CAN be paid by tapping creatures).
        // For G-Zero, if condition met, return 0. (Not yet implemented in get_adjusted_cost, should be added here).

        // TODO: G-Zero Logic
        // if (card_def.keywords.g_zero && check_g_zero_condition(game_state, player, card_def)) return 0;

        return get_adjusted_cost(game_state, player, card_def);
    }

}
