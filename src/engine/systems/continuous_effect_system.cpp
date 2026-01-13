#include "engine/systems/continuous_effect_system.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "core/modifiers.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine::systems {

    using namespace core;

    void ContinuousEffectSystem::recalculate(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        // 1. Remove existing static-sourced modifiers
        auto& passive_effects = state.passive_effects;
        passive_effects.erase(std::remove_if(passive_effects.begin(), passive_effects.end(),
            [](const PassiveEffect& eff) { return eff.is_source_static; }), passive_effects.end());

        auto& active_modifiers = state.active_modifiers;
        active_modifiers.erase(std::remove_if(active_modifiers.begin(), active_modifiers.end(),
            [](const CostModifier& mod) { return mod.is_source_static; }), active_modifiers.end());

        // 2. Iterate all cards in Battle Zone for both players
        //    (Static abilities are primarily active in Battle Zone)
        for (const auto& player : state.players) {
            for (const auto& card : player.battle_zone) {
                if (!card_db.count(card.card_id)) continue;
                const auto& def = card_db.at(card.card_id);

                for (const auto& mod_def : def.static_abilities) {
                    // Check Condition (Source condition, e.g. "If I am tapped")
                    if (mod_def.condition.type != "NONE" && !mod_def.condition.type.empty()) {
                        // ConditionSystem usually requires context.
                        // We pass the card itself as source.
                        // Use instance() to access singleton
                        if (!ConditionSystem::instance().evaluate_def(state, mod_def.condition, card.instance_id, card_db, {})) {
                            continue;
                        }
                    }

                    // Apply Modifier based on Type
                    if (mod_def.type == ModifierType::COST_MODIFIER) {
                        CostModifier cm;
                        cm.reduction_amount = mod_def.value;
                        cm.condition_filter = mod_def.filter; // Target filter (e.g. "Fire Birds")
                        cm.turns_remaining = -1; // Indefinite
                        cm.source_instance_id = card.instance_id;
                        cm.controller = player.id;
                        cm.is_source_static = true;

                        state.active_modifiers.push_back(cm);
                    }
                    else {
                        // Passive Effects (Power, Keywords)
                        PassiveEffect pe;
                        pe.is_source_static = true;
                        pe.source_instance_id = card.instance_id;
                        pe.controller = player.id;
                        pe.turns_remaining = -1;

                        // Map Type
                        switch (mod_def.type) {
                            case ModifierType::POWER_MODIFIER:
                                pe.type = PassiveType::POWER_MODIFIER;
                                break;
                            case ModifierType::GRANT_KEYWORD:
                                pe.type = PassiveType::KEYWORD_GRANT;
                                break;
                            case ModifierType::SET_KEYWORD:
                                // Not standard supported yet in PassiveType, maybe treat as Grant?
                                pe.type = PassiveType::KEYWORD_GRANT;
                                break;
                            case ModifierType::FORCE_ATTACK:
                                pe.type = PassiveType::FORCE_ATTACK;
                                break;
                            default:
                                continue; // Skip unknown
                        }

                        pe.value = mod_def.value;
                        pe.str_value = mod_def.str_val;
                        pe.target_filter = mod_def.filter;
                        pe.condition = mod_def.condition; // Store condition?

                        // Handle Empty Filter -> Assume SELF (Safety fallback)
                        // Must verify all fields are empty to ensure we don't accidentally override valid filters (e.g. "All Dragons")
                        bool is_empty = pe.target_filter.zones.empty() &&
                                        pe.target_filter.races.empty() &&
                                        pe.target_filter.civilizations.empty() &&
                                        pe.target_filter.types.empty() &&
                                        !pe.target_filter.owner.has_value() &&
                                        !pe.target_filter.min_cost.has_value() &&
                                        !pe.target_filter.max_cost.has_value() &&
                                        !pe.target_filter.min_power.has_value() &&
                                        !pe.target_filter.max_power.has_value() &&
                                        !pe.target_filter.is_tapped.has_value() &&
                                        !pe.target_filter.is_blocker.has_value() &&
                                        !pe.target_filter.is_evolution.has_value() &&
                                        pe.target_filter.and_conditions.empty();

                        if (is_empty) {
                            // Explicitly target SELF in BATTLE_ZONE
                            pe.target_filter.owner = "SELF";
                            pe.target_filter.zones = {"BATTLE_ZONE"};
                        }

                        state.passive_effects.push_back(pe);
                    }
                }
            }
        }
    }

}
