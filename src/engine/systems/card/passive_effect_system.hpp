#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "target_utils.hpp"

namespace dm::engine {
    class PassiveEffectSystem {
    public:
        static PassiveEffectSystem& instance() {
            static PassiveEffectSystem instance;
            return instance;
        }

        int get_power_buff(const dm::core::GameState& game_state, const dm::core::CardInstance& creature, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            int buff = 0;
            for (const auto& eff : game_state.passive_effects) {
                if (eff.type == dm::core::PassiveType::POWER_MODIFIER) {
                    if (TargetUtils::is_valid_target(creature, card_db.at(creature.card_id), eff.target_filter, game_state, eff.controller, game_state.card_owner_map[creature.instance_id])) {
                        buff += eff.value;
                    }
                }
            }
            return buff;
        }

        bool check_restriction(const dm::core::GameState& game_state, const dm::core::CardInstance& card, dm::core::PassiveType restriction_type, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            for (const auto& eff : game_state.passive_effects) {
                if (eff.type == restriction_type) {
                    // Check if filter matches the card
                    // We need card definition
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        // Is the card affected?
                        // Restriction target_filter defines WHO is restricted.
                        // e.g. "Opponent creatures cannot attack".
                        // filter owner: OPPONENT.
                        // We check validity.

                        // We need to know who controls the card to check filter owner.
                        // Assuming we can get it from map or context?
                        // We can use game_state.card_owner_map.
                        dm::core::PlayerID controller = 0;
                        if (card.instance_id < (int)game_state.card_owner_map.size()) {
                            controller = game_state.card_owner_map[card.instance_id];
                        }

                        if (TargetUtils::is_valid_target(card, def, eff.target_filter, game_state, eff.controller, controller)) {
                            return true; // Restricted
                        }
                    }
                }
                // Lock Spells by Cost
                if (restriction_type == dm::core::PassiveType::LOCK_SPELL_BY_COST && eff.type == dm::core::PassiveType::LOCK_SPELL_BY_COST) {
                     // Check if card matches filter (e.g. CardType SPELL)
                     if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        // Also check cost
                        if (def.cost == eff.value) {
                             // Check filter for controller etc.
                             dm::core::PlayerID controller = 0;
                            if (card.instance_id < (int)game_state.card_owner_map.size()) {
                                controller = game_state.card_owner_map[card.instance_id];
                            }
                            if (TargetUtils::is_valid_target(card, def, eff.target_filter, game_state, eff.controller, controller)) {
                                return true;
                            }
                        }
                     }
                }
            }
            return false;
        }
    };
}
