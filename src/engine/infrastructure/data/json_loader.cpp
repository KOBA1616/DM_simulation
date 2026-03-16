#include "json_loader.hpp"
#include "card_registry.hpp"
#include "core/card_json_types.hpp"
#include "keyword_expander.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace dm::engine::infrastructure {

    using namespace dm::core;

    // Helper to convert Legacy ActionDef to New CommandDef
    static CommandDef convert_legacy_action(const ActionDef& act) {
        CommandDef cmd;
        cmd.optional = act.optional;
        cmd.amount = act.value1;
        cmd.str_param = act.str_val;
        cmd.target_filter = act.filter;
        cmd.target_group = act.scope;

        cmd.input_value_key = act.input_value_key;
        cmd.output_value_key = act.output_value_key;
        cmd.input_value_usage = act.input_value_usage;

        // Map Legacy Scope to Zone/Target if necessary
        if (act.source_zone.empty() == false) cmd.from_zone = act.source_zone;
        if (act.destination_zone.empty() == false) cmd.to_zone = act.destination_zone;

        switch (act.type) {
            case EffectPrimitive::DRAW_CARD:
                cmd.type = CommandType::DRAW_CARD;
                break;
            case EffectPrimitive::ADD_MANA:
                cmd.type = CommandType::BOOST_MANA;
                break;
            case EffectPrimitive::DESTROY:
                cmd.type = CommandType::DESTROY;
                break;
            case EffectPrimitive::RETURN_TO_HAND:
                cmd.type = CommandType::RETURN_TO_HAND;
                break;
            case EffectPrimitive::TAP:
                cmd.type = CommandType::TAP;
                break;
            case EffectPrimitive::UNTAP:
                cmd.type = CommandType::UNTAP;
                break;
            case EffectPrimitive::MODIFY_POWER:
                cmd.type = CommandType::POWER_MOD;
                break;
            case EffectPrimitive::BREAK_SHIELD:
                cmd.type = CommandType::BREAK_SHIELD;
                break;
            case EffectPrimitive::DISCARD:
                cmd.type = CommandType::DISCARD;
                break;
            case EffectPrimitive::SEARCH_DECK:
                cmd.type = CommandType::SEARCH_DECK;
                break;
            case EffectPrimitive::GRANT_KEYWORD:
                cmd.type = CommandType::ADD_KEYWORD;
                break;
            case EffectPrimitive::SEND_TO_MANA:
                cmd.type = CommandType::SEND_TO_MANA;
                break;
            case EffectPrimitive::MOVE_CARD:
                cmd.type = CommandType::TRANSITION;
                break;
            case EffectPrimitive::LOOK_AND_ADD:
                cmd.type = CommandType::LOOK_AND_ADD;
                break;
            case EffectPrimitive::SUMMON_TOKEN:
                cmd.type = CommandType::SUMMON_TOKEN;
                break;
            case EffectPrimitive::SEARCH_DECK_BOTTOM:
                cmd.type = CommandType::SEARCH_DECK_BOTTOM;
                break;
            case EffectPrimitive::MEKRAID:
                cmd.type = CommandType::MEKRAID;
                break;
            case EffectPrimitive::PLAY_FROM_ZONE:
                cmd.type = CommandType::PLAY_FROM_ZONE;
                break;
            case EffectPrimitive::COST_REFERENCE:
                cmd.type = CommandType::QUERY;
                break;
            case EffectPrimitive::LOOK_TO_BUFFER:
                cmd.type = CommandType::LOOK_TO_BUFFER;
                break;
            case EffectPrimitive::REVEAL_TO_BUFFER:
                cmd.type = CommandType::REVEAL_TO_BUFFER;
                break;
            case EffectPrimitive::SELECT_FROM_BUFFER:
                cmd.type = CommandType::SELECT_FROM_BUFFER;
                break;
            case EffectPrimitive::PLAY_FROM_BUFFER:
                cmd.type = CommandType::PLAY_FROM_BUFFER;
                break;
            case EffectPrimitive::MOVE_BUFFER_TO_ZONE:
                cmd.type = CommandType::MOVE_BUFFER_TO_ZONE;
                break;
            case EffectPrimitive::REVOLUTION_CHANGE:
                cmd.type = CommandType::MUTATE;
                break;
            case EffectPrimitive::COUNT_CARDS:
                cmd.type = CommandType::QUERY;
                break;
            case EffectPrimitive::GET_GAME_STAT:
                cmd.type = CommandType::QUERY;
                break;
            case EffectPrimitive::APPLY_MODIFIER:
                cmd.type = CommandType::MUTATE;
                break;
            case EffectPrimitive::REVEAL_CARDS:
                cmd.type = CommandType::REVEAL_CARDS;
                break;
            case EffectPrimitive::REGISTER_DELAYED_EFFECT:
                cmd.type = CommandType::REGISTER_DELAYED_EFFECT;
                break;
            case EffectPrimitive::RESET_INSTANCE:
                cmd.type = CommandType::MUTATE;
                break;
            case EffectPrimitive::SHUFFLE_DECK:
                cmd.type = CommandType::SHUFFLE_DECK;
                break;
            case EffectPrimitive::ADD_SHIELD:
                cmd.type = CommandType::ADD_SHIELD;
                break;
            case EffectPrimitive::SEND_SHIELD_TO_GRAVE:
                cmd.type = CommandType::SHIELD_BURN;
                break;
            case EffectPrimitive::SEND_TO_DECK_BOTTOM:
                cmd.type = CommandType::SEND_TO_DECK_BOTTOM;
                break;
            case EffectPrimitive::MOVE_TO_UNDER_CARD:
                cmd.type = CommandType::TRANSITION;
                break;
            case EffectPrimitive::SELECT_NUMBER:
                cmd.type = CommandType::SELECT_NUMBER;
                break;
            case EffectPrimitive::FRIEND_BURST:
                cmd.type = CommandType::FRIEND_BURST;
                break;
            case EffectPrimitive::CAST_SPELL:
                cmd.type = CommandType::CAST_SPELL;
                break;
            case EffectPrimitive::PUT_CREATURE:
                cmd.type = CommandType::PLAY_FROM_ZONE;
                break;
            case EffectPrimitive::SELECT_OPTION:
                cmd.type = CommandType::CHOICE;
                break;
            case EffectPrimitive::RESOLVE_BATTLE:
                cmd.type = CommandType::RESOLVE_BATTLE;
                break;
            // Map other types as needed or log warning
            default:
                // Fallback: If no mapping, we might lose functionality for now.
                // This is expected during migration.
                cmd.type = CommandType::NONE;
                break;
        }
        return cmd;
    }

    static int read_schema_version_or_default(const nlohmann::json& item) {
        if (item.contains("schema_version") && item.at("schema_version").is_number_integer()) {
            return item.at("schema_version").get<int>();
        }
        return 1;
    }

    static bool has_legacy_actions(const CardData& data) {
        for (const auto& eff : data.effects) {
            if (!eff.actions.empty()) {
                return true;
            }
        }
        for (const auto& eff : data.metamorph_abilities) {
            if (!eff.actions.empty()) {
                return true;
            }
        }
        return false;
    }

    static bool has_legacy_actions_in_json(const nlohmann::json& item) {
        auto has_actions_in = [](const nlohmann::json& arr) {
            if (!arr.is_array()) {
                return false;
            }
            for (const auto& eff : arr) {
                if (!eff.is_object()) {
                    continue;
                }
                if (eff.contains("actions") && eff.at("actions").is_array() && !eff.at("actions").empty()) {
                    return true;
                }
            }
            return false;
        };

        if (item.contains("effects") && has_actions_in(item.at("effects"))) {
            return true;
        }
        if (item.contains("metamorph_abilities") && has_actions_in(item.at("metamorph_abilities"))) {
            return true;
        }
        return false;
    }

    // Helper to convert CardData (JSON) to CardDefinition (Engine)
    // Forward declare to allow recursive calls
    static CardDefinition convert_to_def(const CardData& data);

    static CardDefinition convert_to_def(const CardData& data) {
        CardDefinition def;
        def.id = static_cast<CardID>(data.id);
        def.name = data.name;

        // Civilization mapping
        def.civilizations = data.civilizations;

        // Type mapping
        def.type = data.type; // Direct assignment, now Enum

        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

        // Copy effects
        def.effects.clear();
        def.effects.reserve(data.effects.size());

        for (const auto& eff : data.effects) {
            EffectDef engine_eff = eff;

            // Legacy Conversion: If commands are empty but actions exist
            if (engine_eff.commands.empty() && !engine_eff.actions.empty()) {
                engine_eff.commands.reserve(engine_eff.actions.size());
                for (const auto& act : engine_eff.actions) {
                    CommandDef cmd = convert_legacy_action(act);
                    if (cmd.type != CommandType::NONE) {
                        engine_eff.commands.push_back(cmd);
                    }
                }
            }
            def.effects.push_back(engine_eff);
        }

        // Metamorph Abilities
        def.metamorph_abilities.clear();
        def.metamorph_abilities.reserve(data.metamorph_abilities.size());
        for (const auto& eff : data.metamorph_abilities) {
             EffectDef engine_eff = eff;
             if (engine_eff.commands.empty() && !engine_eff.actions.empty()) {
                engine_eff.commands.reserve(engine_eff.actions.size());
                for (const auto& act : engine_eff.actions) {
                    CommandDef cmd = convert_legacy_action(act);
                    if (cmd.type != CommandType::NONE) {
                        engine_eff.commands.push_back(cmd);
                    }
                }
            }
            def.metamorph_abilities.push_back(engine_eff);
        }

        def.cost_reductions = data.cost_reductions;

        // Back-compat: ensure each cost_reduction has a stable `id` (used as canonical identifier).
        // Note: `name` is considered display-only and MUST NOT be relied upon as an identifier
        // by engine logic. Older code paths used `name` as identifier; we preserve `id` and
        // avoid overwriting `name` here to keep display labels intact.
        for (size_t i = 0; i < def.cost_reductions.size(); ++i) {
            auto &cr = def.cost_reductions[i];
            // Generate stable id if missing (new schema uses `id`)
            if (cr.id.empty()) {
                std::ostringstream ss;
                ss << "cr_" << def.id << "_" << i;
                cr.id = ss.str();
            }
            // Do NOT auto-fill or overwrite `name` here; leave it for editors/UI to manage.
        }

        // Evolution Condition
        if (data.evolution_condition.has_value()) {
            def.evolution_condition = data.evolution_condition;
        }

        // Revolution Change
        if (data.revolution_change_condition.has_value()) {
            def.revolution_change_condition = data.revolution_change_condition;
            def.keywords.revolution_change = true;
        }

        // Reaction Abilities
        def.reaction_abilities = data.reaction_abilities;

        // Twinpact Spell Side (Recursive)
        if (data.spell_side) {
            def.spell_side = std::make_shared<CardDefinition>(convert_to_def(*data.spell_side));
        }

        // Keywords from explicit 'keywords' block (e.g. S-Trigger)
        if (data.keywords.has_value()) {
            const auto& kws = *data.keywords;
            if (kws.count("shield_trigger") && kws.at("shield_trigger")) def.keywords.shield_trigger = true;
            if (kws.count("blocker") && kws.at("blocker")) def.keywords.blocker = true;
            if (kws.count("speed_attacker") && kws.at("speed_attacker")) def.keywords.speed_attacker = true;
            if (kws.count("slayer") && kws.at("slayer")) def.keywords.slayer = true;
            if (kws.count("double_breaker") && kws.at("double_breaker")) def.keywords.double_breaker = true;
            if (kws.count("triple_breaker") && kws.at("triple_breaker")) def.keywords.triple_breaker = true;
            if (kws.count("mach_fighter") && kws.at("mach_fighter")) def.keywords.mach_fighter = true;
            if (kws.count("evolution") && kws.at("evolution")) def.keywords.evolution = true;
            if (kws.count("g_strike") && kws.at("g_strike")) def.keywords.g_strike = true;
            if (kws.count("just_diver") && kws.at("just_diver")) def.keywords.just_diver = true;
            if (kws.count("shield_burn") && kws.at("shield_burn")) def.keywords.shield_burn = true;
            if (kws.count("untap_in") && kws.at("untap_in")) def.keywords.untap_in = true;
            if (kws.count("unblockable") && kws.at("unblockable")) def.keywords.unblockable = true;
            if (kws.count("must_be_chosen") && kws.at("must_be_chosen")) def.keywords.must_be_chosen = true;

             // Meta Counter Play (e.g. Oriot Judgement)
            if (kws.count("meta_counter_play") && kws.at("meta_counter_play")) {
                 def.keywords.meta_counter_play = true;
                 // Assuming implicit effect generation is handled elsewhere or by specific keywords
            }

            if (kws.count("hyper_energy") && kws.at("hyper_energy")) def.keywords.hyper_energy = true;
            if (kws.count("super_soul_x") && kws.at("super_soul_x")) def.keywords.super_soul_x = true;

            // Expand complex keywords (Friend Burst, Mega Last Burst, etc.)
            dm::engine::infrastructure::KeywordExpander::expand_keywords(data, def);
        }

        // Keywords mapping from effects
        for (const auto& eff : data.effects) {
            // 1. Trigger Flags
            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.shield_trigger = true;
            if (eff.trigger == TriggerType::ON_PLAY) def.keywords.cip = true;
            if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.at_attack = true;
            if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.destruction = true;

            // 2. Passive Keywords (Blocker, Speed Attacker, etc.)
            if (eff.trigger == TriggerType::PASSIVE_CONST) {
                // Passive keywords are derived from migrated `commands`.
                // Legacy `actions` are converted into `commands` at load time
                // (`convert_legacy_action`) so runtime logic should rely on
                // `commands` only.
                for (const auto& cmd : eff.commands) {
                    if (cmd.str_val == "BLOCKER") def.keywords.blocker = true;
                    if (cmd.str_val == "SPEED_ATTACKER") def.keywords.speed_attacker = true;
                    if (cmd.str_val == "SLAYER") def.keywords.slayer = true;
                    if (cmd.str_val == "DOUBLE_BREAKER") def.keywords.double_breaker = true;
                    if (cmd.str_val == "TRIPLE_BREAKER") def.keywords.triple_breaker = true;
                    if (cmd.str_val == "POWER_ATTACKER") {
                        def.keywords.power_attacker = true;
                        def.power_attacker_bonus = cmd.amount;
                    }
                    if (cmd.str_val == "EVOLUTION") def.keywords.evolution = true;
                    if (cmd.str_val == "MACH_FIGHTER") def.keywords.mach_fighter = true;
                    if (cmd.str_val == "G_STRIKE") def.keywords.g_strike = true;
                    if (cmd.str_val == "JUST_DIVER") def.keywords.just_diver = true;
                    if (cmd.str_val == "HYPER_ENERGY") def.keywords.hyper_energy = true;
                    if (cmd.str_val == "META_COUNTER") def.keywords.meta_counter_play = true;
                    if (cmd.str_val == "SHIELD_BURN") def.keywords.shield_burn = true;
                    if (cmd.str_val == "UNBLOCKABLE") def.keywords.unblockable = true;
                    if (cmd.str_val == "MUST_BE_CHOSEN") def.keywords.must_be_chosen = true;
                                    if (cmd.str_val == "SUPER_SOUL_X") def.keywords.super_soul_x = true;
                }
            }
        }

        // AI Metadata
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        return def;
    }

    std::map<CardID, CardDefinition> dm::engine::infrastructure::JsonLoader::load_cards(const std::string& filepath) {
        std::map<CardID, CardDefinition> result;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file: " << filepath << std::endl;
            return result;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_str = buffer.str();

        // Prefer parsing directly here to produce the definitions.
        // Attempt to populate the registry as a best-effort but do not rely
        // on it for returning results to avoid registry-side sanitization
        // causing early failures during migration.
        try {
            result = dm::engine::infrastructure::JsonLoader::load_cards_from_string(json_str);
        } catch (const std::exception& e) {
            std::cerr << "JsonLoader::load_cards failed to produce defs: " << e.what() << std::endl;
            // Fallback: attempt to parse for IDs using registry if available
            try {
                dm::engine::infrastructure::CardRegistry::load_from_json(json_str);
                auto j = nlohmann::json::parse(json_str);
                const auto& registry_defs = dm::engine::infrastructure::CardRegistry::get_all_definitions();
                auto process_item = [&](const nlohmann::json& item) {
                    if (item.contains("id")) {
                        int id = item["id"].get<int>();
                        CardID card_id = static_cast<CardID>(id);
                        if (registry_defs.count(card_id)) result[card_id] = registry_defs.at(card_id);
                    }
                };
                if (j.is_array()) for (const auto& item : j) process_item(item);
                else process_item(j);
            } catch (...) {
                // Give up; return empty result
            }
        }

        // Best-effort: try to update the CardRegistry but ignore failures
        try {
            dm::engine::infrastructure::CardRegistry::load_from_json(json_str);
        } catch (...) {}

        return result;
    }

    std::map<CardID, CardDefinition> dm::engine::infrastructure::JsonLoader::load_cards_from_string(const std::string& json_content) {
        std::map<CardID, CardDefinition> result;
        try {
            auto j = nlohmann::json::parse(json_content);

            auto process_item = [&](const nlohmann::json& item) {
                if (!item.contains("id")) return;

                const int schema_version = read_schema_version_or_default(item);
                // 再発防止: 生JSONで先に境界判定することで、型変換時に actions が欠落しても
                // schema_version>=2 で legacy 記法を取り込んでしまう事故を防ぐ。
                if (schema_version >= 2 && has_legacy_actions_in_json(item)) {
                    std::cerr << "JsonLoader::load_cards_from_string rejected raw card because schema_version="
                              << schema_version
                              << " forbids legacy actions" << std::endl;
                    return;
                }

                // Make a mutable copy and sanitize nullable object fields
                nlohmann::json copy = item;

                // Aggressive sanitization: replace explicit JSON nulls with a
                // reasonable default according to the JSON key name when
                // possible. This avoids throwing from `value()` / `get_to()` by
                // providing a type-compatible placeholder for known fields.
                std::function<void(nlohmann::json&)> sanitize_all_nulls = [&](nlohmann::json& node) {
                    static const std::unordered_set<std::string> int_keys = {"id", "cost", "power", "value", "value1", "value2", "amount", "duration", "slot_index", "target_slot_index", "ai_importance_score"};
                    static const std::unordered_set<std::string> string_keys = {"name", "str_val", "type", "scope", "source_zone", "destination_zone", "target_choice", "input_value_key", "output_value_key", "input_value_usage", "timing_mode", "multiplicity"};
                    static const std::unordered_set<std::string> bool_keys = {"optional", "is_key_card", "is_tapped"};
                    static const std::unordered_set<std::string> array_keys = {"civilizations", "races", "triggers", "effects", "commands", "actions", "options", "trigger_list", "trigger_zones", "children", "static_abilities", "metamorph_abilities", "reaction_abilities", "cost_reductions"};

                    if (node.is_object()) {
                        for (auto it = node.begin(); it != node.end(); ++it) {
                            auto &val = it.value();
                            const std::string key = it.key();
                            if (val.is_null()) {
                                if (int_keys.count(key)) val = 0;
                                else if (string_keys.count(key)) val = std::string("");
                                else if (bool_keys.count(key)) val = false;
                                else if (array_keys.count(key)) val = nlohmann::json::array();
                                else if (key == "spell_side" || key == "trigger_descriptor" || key == "condition" || key == "filter" || key == "keywords") {
                                    // Known object-like fields
                                    val = nlohmann::json::object();
                                } else {
                                    // Conservative default: object to preserve nested keys
                                    val = nlohmann::json::object();
                                }
                            } else if (val.is_object() || val.is_array()) {
                                sanitize_all_nulls(val);
                            }
                        }
                    } else if (node.is_array()) {
                        for (auto &el : node) {
                            if (el.is_null()) el = nlohmann::json::object();
                            else if (el.is_object() || el.is_array()) sanitize_all_nulls(el);
                        }
                    }
                };

                sanitize_all_nulls(copy);

                dm::core::CardData data;
                // Manual or automated deserialization from JSON to CardData
                try {
                    dm::core::from_json(copy, data);
                } catch (const std::exception& e) {
                    // Emit the offending JSON for diagnosis before aggressive sanitize
                    try {
                        std::cerr << "JsonLoader::load_cards_from_string -- deserialization failed for item: " << copy.dump(2) << std::endl;
                        // Also write a diagnostic file for offline inspection
                        try {
                            std::ofstream diag("logs/json_loader_offending_item.json", std::ios::app);
                            if (diag.is_open()) {
                                diag << copy.dump(2) << std::endl;
                                diag.close();
                            }
                        } catch (...) {}
                    } catch (...) {}

                    // Fallback: some legacy JSON may contain explicit nulls in
                    // places where older editors emitted `null`. Attempt a
                    // more aggressive sanitization (replace any null with an
                    // empty object) and retry once before giving up.
                    // Fallback aggressive sanitize (same logic as above)
                    std::function<void(nlohmann::json&)> sanitize_all_nulls = [&](nlohmann::json& node) {
                        static const std::unordered_set<std::string> int_keys = {"id", "cost", "power", "value", "value1", "value2", "amount", "duration", "slot_index", "target_slot_index", "ai_importance_score"};
                        static const std::unordered_set<std::string> string_keys = {"name", "str_val", "type", "scope", "source_zone", "destination_zone", "target_choice", "input_value_key", "output_value_key", "input_value_usage", "timing_mode", "multiplicity"};
                        static const std::unordered_set<std::string> bool_keys = {"optional", "is_key_card", "is_tapped"};
                        static const std::unordered_set<std::string> array_keys = {"civilizations", "races", "triggers", "effects", "commands", "actions", "options", "trigger_list", "trigger_zones", "children", "static_abilities", "metamorph_abilities", "reaction_abilities", "cost_reductions"};

                        if (node.is_object()) {
                            for (auto it = node.begin(); it != node.end(); ++it) {
                                auto &val = it.value();
                                const std::string key = it.key();
                                if (val.is_null()) {
                                    if (int_keys.count(key)) val = 0;
                                    else if (string_keys.count(key)) val = std::string("");
                                    else if (bool_keys.count(key)) val = false;
                                    else if (array_keys.count(key)) val = nlohmann::json::array();
                                    else if (key == "keywords") val = nlohmann::json::object();
                                    else val = nlohmann::json::object();
                                } else if (val.is_object() || val.is_array()) {
                                    sanitize_all_nulls(val);
                                }
                            }
                        } else if (node.is_array()) {
                            for (auto &el : node) {
                                if (el.is_null()) el = nlohmann::json::object();
                                else if (el.is_object() || el.is_array()) sanitize_all_nulls(el);
                            }
                        }
                    };

                    sanitize_all_nulls(copy);
                    try {
                        dm::core::from_json(copy, data);
                    } catch (const std::exception& e2) {
                        std::cerr << "JsonLoader::load_cards_from_string Error after aggressive sanitize: " << e2.what() << std::endl;
                        return;
                    }
                }

                // 再発防止: schema_version>=2 は CommandDef 専用。legacy actions を許可すると
                // 新旧JSON境界が曖昧になり、ActionDef 撤去時に回帰しやすくなる。
                if (schema_version >= 2 && has_legacy_actions(data)) {
                    std::cerr << "JsonLoader::load_cards_from_string rejected card id="
                              << data.id
                              << " because schema_version="
                              << schema_version
                              << " forbids legacy actions" << std::endl;
                    return;
                }

                // Convert to Definition
                CardDefinition def = convert_to_def(data);
                result[def.id] = def;
            };

            if (j.is_array()) {
                for (const auto& item : j) {
                    process_item(item);
                }
            } else {
                process_item(j);
            }
        } catch (const std::exception& e) {
            std::cerr << "JsonLoader::load_cards_from_string Error: " << e.what() << std::endl;
        }
        return result;
    }

}
