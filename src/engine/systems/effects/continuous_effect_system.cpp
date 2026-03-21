#include "continuous_effect_system.hpp"
#include "passive_effect_system.hpp"
#include "engine/systems/rules/condition_system.hpp"
#include "engine/utils/target_utils.hpp"
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

                // Diagnostic: report static abilities present on the definition
                try {
                    std::cerr << "ContinuousEffectSystem::recalculate player=" << player.id
                              << " card_instance=" << card.instance_id
                              << " card_id=" << def.id
                              << " static_count=" << def.static_abilities.size() << std::endl;
                } catch (...) {}

                // 再発防止: 「能力を無視する」効果が適用中のカードは静的能力を供給しない。
                if (PassiveEffectSystem::instance().check_restriction(
                        state, card, PassiveType::IGNORE_ABILITIES, card_db)) {
                    continue;
                }

                for (const auto& mod_def : def.static_abilities) {
                    try {
                        std::cerr << "  examine_static_mod: type=" << static_cast<int>(mod_def.type)
                                  << " value=" << mod_def.value
                                  << " value_mode='" << mod_def.value_mode << "'"
                                  << " stat_key='" << mod_def.stat_key << "' per_value=" << mod_def.per_value
                                  << " min_stat=" << mod_def.min_stat;
                        if (mod_def.max_reduction.has_value()) std::cerr << " max_reduction=" << mod_def.max_reduction.value();
                        std::cerr << std::endl;
                    } catch (...) {}
                    // Check Condition (Source condition, e.g. "If I am tapped")
                    if (mod_def.condition.type != "NONE" && !mod_def.condition.type.empty()) {
                        // dm::engine::rules::ConditionSystem usually requires context.
                        // We pass the card itself as source.
                        // Use instance() to access singleton
                        if (!dm::engine::rules::ConditionSystem::instance().evaluate_def(state, mod_def.condition, card.instance_id, card_db, {})) {
                            continue;
                        }
                    }

                    // Apply Modifier based on Type
                    if (mod_def.type == ModifierType::COST_MODIFIER) {
                        CostModifier cm;
                        // Default: legacy FIXED behavior
                        int reduction_amount = mod_def.value;

                        // STAT_SCALED: compute from game state statistics when configured
                        if (!mod_def.value_mode.empty() && mod_def.value_mode == "STAT_SCALED") {
                            int stat_val = 0;
                            const std::string& key = mod_def.stat_key;

                            // Minimal mapping of known stat keys to GameState fields.
                            // Extend this switch as more keys are needed.
                            if (key == "summon_count_this_turn" || key == "SUMMON_COUNT") {
                                stat_val = state.turn_stats.summon_count_this_turn;
                            } else if (key == "creatures_played_this_turn" || key == "CREATURES_PLAYED") {
                                stat_val = state.turn_stats.creatures_played_this_turn;
                            } else if (key == "attacked_this_turn" || key == "ATTACKED") {
                                // Use controller-specific counter when available
                                if (player.id >= 0 && player.id < 2) stat_val = state.turn_stats.attacked_this_turn_by_player[player.id];
                            } else if (key == "summon_count_by_player") {
                                if (player.id >= 0 && player.id < 2) stat_val = state.turn_stats.summon_count_this_turn; // fallback
                            } else {
                                // Unknown key: try some generic fallbacks
                                stat_val = 0;
                            }

                            int per_value = mod_def.per_value;
                            int min_stat = mod_def.min_stat;
                            int calculated = std::max(0, stat_val - min_stat + 1) * per_value;
                            if (mod_def.max_reduction.has_value()) {
                                calculated = std::min(calculated, mod_def.max_reduction.value());
                            }
                            if (calculated > 0) reduction_amount = calculated;
                            else reduction_amount = 0;
                        }

                        cm.reduction_amount = reduction_amount;
                        cm.condition_filter = mod_def.filter; // Target filter (e.g. "Fire Birds")
                        cm.turns_remaining = -1; // Indefinite
                        cm.source_instance_id = card.instance_id;
                        cm.controller = player.id;
                        cm.is_source_static = true;

                        state.active_modifiers.push_back(cm);
                        try {
                            std::cerr << "    pushed_active_mod: controller=" << cm.controller
                                      << " source_instance=" << cm.source_instance_id
                                      << " reduction=" << cm.reduction_amount
                                      << " active_mods_len=" << state.active_modifiers.size()
                                      << std::endl;
                        } catch(...) {}
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
                            case ModifierType::ADD_RESTRICTION: {
                                const std::string& kind = mod_def.str_val;

                                // Restriction kind is stored in str_val (GUI writes both mutation_kind and str_val).
                                // NOTE: SPELL_RESTRICTION is treated the same as TARGET_RESTRICTION in the engine for now.
                                if (kind == "TARGET_RESTRICTION" || kind == "SPELL_RESTRICTION" || kind == "TARGET_THIS_CANNOT_SELECT") {
                                    pe.type = PassiveType::CANNOT_BE_SELECTED;
                                } else if (kind == "TARGET_THIS_FORCE_SELECT") {
                                    pe.type = PassiveType::FORCE_SELECTION;
                                } else {
                                    continue; // Unknown restriction kind
                                }
                                break;
                            }
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

        // SUPER_SOUL_X: 下に置かれたカードの静的能力をトップクリーチャーに伝播する
        // 再発防止: super_soul_x フラグを持つ underlying_cards のみ対象。
        //   eff の source_instance_id はトップクリーチャーの ID を使う（エフェクト追跡のため）。
        for (const auto& player : state.players) {
            for (const auto& top_card : player.battle_zone) {
                for (const auto& under : top_card.underlying_cards) {
                    if (!card_db.count(under.card_id)) continue;
                    const auto& under_def = card_db.at(under.card_id);
                    if (!under_def.keywords.super_soul_x) continue;

                    for (const auto& mod_def : under_def.static_abilities) {
                        if (mod_def.condition.type != "NONE" && !mod_def.condition.type.empty()) {
                            if (!dm::engine::rules::ConditionSystem::instance().evaluate_def(
                                    state, mod_def.condition, top_card.instance_id, card_db, {})) {
                                continue;
                            }
                        }

                        if (mod_def.type == ModifierType::COST_MODIFIER) {
                            CostModifier cm;
                            cm.reduction_amount = mod_def.value;
                            cm.condition_filter = mod_def.filter;
                            cm.turns_remaining = -1;
                            cm.source_instance_id = top_card.instance_id;
                            cm.controller = player.id;
                            cm.is_source_static = true;
                            state.active_modifiers.push_back(cm);
                            try {
                                std::cerr << "    pushed_active_mod(super_soul): controller=" << cm.controller
                                          << " source_instance=" << cm.source_instance_id
                                          << " reduction=" << cm.reduction_amount
                                          << " active_mods_len=" << state.active_modifiers.size()
                                          << std::endl;
                            } catch(...) {}
                        } else {
                            PassiveEffect pe;
                            pe.is_source_static = true;
                            pe.source_instance_id = top_card.instance_id;
                            pe.controller = player.id;
                            pe.turns_remaining = -1;

                            switch (mod_def.type) {
                                case ModifierType::POWER_MODIFIER:
                                    pe.type = PassiveType::POWER_MODIFIER;
                                    break;
                                case ModifierType::GRANT_KEYWORD:
                                case ModifierType::SET_KEYWORD:
                                    pe.type = PassiveType::KEYWORD_GRANT;
                                    break;
                                case ModifierType::FORCE_ATTACK:
                                    pe.type = PassiveType::FORCE_ATTACK;
                                    break;
                                case ModifierType::ADD_RESTRICTION: {
                                    const std::string& kind = mod_def.str_val;
                                    if (kind == "TARGET_RESTRICTION" || kind == "SPELL_RESTRICTION" || kind == "TARGET_THIS_CANNOT_SELECT") {
                                        pe.type = PassiveType::CANNOT_BE_SELECTED;
                                    } else if (kind == "TARGET_THIS_FORCE_SELECT") {
                                        pe.type = PassiveType::FORCE_SELECTION;
                                    } else {
                                        continue;
                                    }
                                    break;
                                }
                                default:
                                    continue;
                            }

                            pe.value = mod_def.value;
                            pe.str_value = mod_def.str_val;
                            pe.target_filter = mod_def.filter;
                            pe.condition = mod_def.condition;

                            // 空フィルタ → トップクリーチャー自身を対象とする
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
                                pe.specific_targets = std::vector<int>{top_card.instance_id};
                            }

                            state.passive_effects.push_back(pe);
                        }
                    }
                }
            }
        }
    }

}
