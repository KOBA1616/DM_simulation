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

        if (paid_mana == 0 && card_def.cost > 0) {
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
