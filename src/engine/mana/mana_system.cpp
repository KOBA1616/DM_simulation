#include "mana_system.hpp"
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../card_system/target_utils.hpp"
#include <algorithm>
#include <iostream>
#include <set>

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

    bool ManaSystem::can_pay_cost(const GameState& game_state, const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = get_adjusted_cost(game_state, player, card_def);

        int available_mana = 0;
        Civilization available_civs = Civilization::NONE;

        for (const auto& card : player.mana_zone) {
            if (!card.is_tapped) {
                available_mana++;
                if (card_db.count(card.card_id)) {
                    const auto& m_def = card_db.at(card.card_id);
                    for (auto civ : m_def.civilizations) {
                        available_civs = available_civs | civ;
                    }
                }
            }
        }

        if (available_mana < cost) return false;

        bool is_colorless = card_def.has_civilization(Civilization::ZERO) || card_def.civilizations.empty();
        if (!is_colorless) {
             bool match = false;
             for (auto req_civ : card_def.civilizations) {
                 if ((available_civs & req_civ) != Civilization::NONE) {
                     match = true;
                     break;
                 }
             }
             if (!match) return false;
        }

        return true;
    }

    bool ManaSystem::can_pay_cost(const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        int cost = card_def.cost;

        int available_mana = 0;
        Civilization available_civs = Civilization::NONE;

        for (const auto& card : player.mana_zone) {
            if (!card.is_tapped) {
                available_mana++;
                if (card_db.count(card.card_id)) {
                    const auto& m_def = card_db.at(card.card_id);
                    for (auto civ : m_def.civilizations) {
                        available_civs = available_civs | civ;
                    }
                }
            }
        }

        if (available_mana < cost) return false;

        bool is_colorless = card_def.has_civilization(Civilization::ZERO) || card_def.civilizations.empty();
        if (!is_colorless) {
             bool match = false;
             for (auto req_civ : card_def.civilizations) {
                 if ((available_civs & req_civ) != Civilization::NONE) {
                     match = true;
                     break;
                 }
             }
             if (!match) return false;
        }

        return true;
    }

    bool ManaSystem::auto_tap_mana(GameState& game_state, Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        if (!can_pay_cost(game_state, player, card_def, card_db)) return false;

        int cost_remaining = get_adjusted_cost(game_state, player, card_def);
        int paid_mana = 0;

        bool is_colorless = card_def.has_civilization(Civilization::ZERO) || card_def.civilizations.empty();
        bool civ_requirement_met = is_colorless;

        // First pass: Try to tap a card that satisfies the civilization requirement
        if (!civ_requirement_met && cost_remaining > 0) {
            for (auto& card : player.mana_zone) {
                if (!card.is_tapped) {
                    const auto& mana_card_def = card_db.at(card.card_id);
                    bool useful = false;
                    for (auto req_civ : card_def.civilizations) {
                        for (auto mc : mana_card_def.civilizations) {
                            if (mc == req_civ) {
                                useful = true;
                                break;
                            }
                        }
                        if (useful) break;
                    }

                    if (useful) {
                        card.is_tapped = true;
                        if (cost_remaining > 0) cost_remaining--;
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

        // If card had a base cost > 0, but we paid 0 mana, set flag.
        // This covers 0 cost cards (cost == 0) not setting the flag,
        // and cards reduced to 0 (though normal reduction minimum is 1, G-Zero skips payment).
        // Wait, auto_tap_mana is called by PAY_COST.
        // If cost_remaining was 0 at start (e.g. G-Zero or base cost 0), loops didn't run, paid_mana = 0.
        // If base cost was > 0, we want to flag it.
        // If base cost was 0, we do NOT want to flag it (e.g. tokens, dummy).
        // The check `paid_mana == 0 && card_def.cost > 0` seems correct for "Played Without Mana".
        // HOWEVER, get_adjusted_cost returns minimum 1 for normal reductions.
        // So `cost_remaining` is at least 1 unless base cost <= 0.
        // Unless G-Zero logic bypasses this function entirely?
        // G-Zero usually sets cost to 0 via specific handling or skips auto_tap_mana?
        // If `ActionType::PAY_COST` calls `auto_tap_mana`, it relies on `get_adjusted_cost`.
        // If `G_ZERO` is active, `ActionGenerator` might produce a `PAY_COST` with target_slot_index or similar?
        // Actually, `EffectResolver` calls `auto_tap_mana`.
        // If `EffectResolver` sees a G-Zero flag or special cost, it might handle it.

        // In the failing test `test_meta_counter_trigger_and_resolution`, the card `self.zero_card_id` has `cost = 0`.
        // `get_adjusted_cost` returns 0 for cost 0.
        // `auto_tap_mana` sees `cost_remaining` 0. Loops skip. `paid_mana` 0.
        // `if (paid_mana == 0 && card_def.cost > 0)` -> 0 > 0 is False.
        // So `played_without_mana` is NOT set.
        // The test expects it to be TRUE?
        // "Player 0 plays 0-cost card"
        // Meta Counter condition: "Opponent played a card without paying mana?"
        // Usually, 0-cost cards (like from G-Zero or naturally 0) COUNT for "played without mana" in the OCG rules?
        // Let's check the rules memory or requirement.
        // Memory: "The ManaSystem::auto_tap_mana logic was updated to prevent tapping mana for civilization requirements if the card's remaining cost is 0, ensuring played_without_mana is correctly set for 0-cost cards."
        // This implies it SHOULD be set.
        // So my check `card_def.cost > 0` is preventing it.
        // I should remove `&& card_def.cost > 0`?
        // But if I play a dummy card (token) or start of game setup?
        // Tokens are "Summoned", not "Played".
        // If a card is PLAYED and cost was 0, it is "Played without mana".
        // So yes, I should allow cost 0.

        if (paid_mana == 0) {
            game_state.turn_stats.played_without_mana = true;
        }

        return true;
    }

    bool ManaSystem::auto_tap_mana(Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        if (!can_pay_cost(player, card_def, card_db)) return false;

        int cost_remaining = card_def.cost;
        bool is_colorless = card_def.has_civilization(Civilization::ZERO) || card_def.civilizations.empty();
        bool civ_requirement_met = is_colorless;

        if (!civ_requirement_met && cost_remaining > 0) {
            for (auto& card : player.mana_zone) {
                if (!card.is_tapped) {
                    const auto& mana_card_def = card_db.at(card.card_id);
                     bool useful = false;
                    for (auto req_civ : card_def.civilizations) {
                        for (auto mc : mana_card_def.civilizations) {
                            if (mc == req_civ) {
                                useful = true;
                                break;
                            }
                        }
                        if (useful) break;
                    }

                    if (useful) {
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
        return get_adjusted_cost(game_state, player, card_def);
    }

}
