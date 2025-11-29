#include "mana_system.hpp"
#include <algorithm>

namespace dm::engine {

    using namespace dm::core;

    bool ManaSystem::can_pay_cost(const Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
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
        if (available_mana < card_def.cost) {
            return false;
        }

        // Check civilization (unless card is colorless/Zero, assuming Zero requires no civ or Zero civ? 
        // Standard DM rules: Zero civ cards usually don't require specific mana color, but let's assume standard rules for colored cards)
        if (card_def.civilization != Civilization::ZERO) {
             if ((available_civs & card_def.civilization) == Civilization::NONE) {
                 return false;
             }
        }

        return true;
    }

    bool ManaSystem::auto_tap_mana(Player& player, const CardDefinition& card_def, const std::map<CardID, CardDefinition>& card_db) {
        if (!can_pay_cost(player, card_def, card_db)) {
            return false;
        }

        int cost_remaining = card_def.cost;
        bool civ_requirement_met = (card_def.civilization == Civilization::ZERO);

        // First pass: Try to tap a card that satisfies the civilization requirement
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

        // If we still haven't met the civ requirement (shouldn't happen if can_pay_cost was true), return false
        // But can_pay_cost checked it, so we assume it's possible.
        
        // Second pass: Tap remaining necessary mana
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

}
