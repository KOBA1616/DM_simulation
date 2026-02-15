#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/systems/rules/condition_system.hpp"

namespace dm::engine {
    class PassiveEffectSystem {
    public:
        static PassiveEffectSystem& instance() {
            static PassiveEffectSystem instance;
            return instance;
        }

        int get_power_buff(const dm::core::GameState& game_state, const dm::core::CardInstance& creature, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            int buff = 0;
            // Hack: We need a non-const GameState for ConditionSystem because some evaluators might (though shouldn't) mutate or just signature mismatch.
            // But evaluate_def takes non-const GameState&.
            // This is a design flaw in ConditionSystem signature if it's purely read-only.
            // However, for now we cast away constness because calculating power should not change state.
            dm::core::GameState& state_ref = const_cast<dm::core::GameState&>(game_state);

            for (const auto& eff : game_state.passive_effects) {
                if (eff.type == dm::core::PassiveType::POWER_MODIFIER) {
                    // Check condition if present
                    if (!eff.condition.type.empty() && eff.condition.type != "NONE") {
                         if (!dm::engine::rules::ConditionSystem::instance().evaluate_def(state_ref, eff.condition, eff.source_instance_id, card_db, {})) {
                             continue;
                         }
                    }

                    if (dm::engine::utils::TargetUtils::is_valid_target(creature, card_db.at(creature.card_id), eff.target_filter, game_state, eff.controller, game_state.get_card_owner(creature.instance_id))) {
                        buff += eff.value;
                    }
                }
            }
            return buff;
        }

        bool check_restriction(const dm::core::GameState& game_state, const dm::core::CardInstance& card, dm::core::PassiveType restriction_type, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            dm::core::GameState& state_ref = const_cast<dm::core::GameState&>(game_state);

            for (const auto& eff : game_state.passive_effects) {
                if (eff.type == restriction_type) {
                    // Check condition first
                    if (!eff.condition.type.empty() && eff.condition.type != "NONE") {
                         if (!dm::engine::rules::ConditionSystem::instance().evaluate_def(state_ref, eff.condition, eff.source_instance_id, card_db, {})) {
                             continue;
                         }
                    }

                    bool restricted = false;

                    // Check specific targets first
                    if (eff.specific_targets.has_value() && !eff.specific_targets->empty()) {
                        for (int id : *eff.specific_targets) {
                            if (id == card.instance_id) {
                                restricted = true;
                                break;
                            }
                        }
                    } else {
                        // Check filter
                        if (card_db.count(card.card_id)) {
                            const auto& def = card_db.at(card.card_id);
                            dm::core::PlayerID controller = 0;
                            if (card.instance_id < (int)game_state.card_owner_map.size()) {
                                controller = game_state.get_card_owner(card.instance_id);
                            }

                            if (dm::engine::utils::TargetUtils::is_valid_target(card, def, eff.target_filter, game_state, eff.controller, controller)) {
                                restricted = true;
                            }
                        }
                    }

                    if (restricted) {
                        return true;
                    }
                }
                // Lock Spells by Cost
                if (restriction_type == dm::core::PassiveType::LOCK_SPELL_BY_COST && eff.type == dm::core::PassiveType::LOCK_SPELL_BY_COST) {
                     // Check condition first
                     if (!eff.condition.type.empty() && eff.condition.type != "NONE") {
                         if (!dm::engine::rules::ConditionSystem::instance().evaluate_def(state_ref, eff.condition, eff.source_instance_id, card_db, {})) {
                             continue;
                         }
                     }

                     if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (def.cost == eff.value) {
                             dm::core::PlayerID controller = 0;
                            if (card.instance_id < (int)game_state.card_owner_map.size()) {
                                controller = game_state.get_card_owner(card.instance_id);
                            }
                            if (dm::engine::utils::TargetUtils::is_valid_target(card, def, eff.target_filter, game_state, eff.controller, controller)) {
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
