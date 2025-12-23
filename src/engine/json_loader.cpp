#include "json_loader.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "core/card_def.hpp"
#include "core/keywords.hpp"
#include <algorithm>
#include <cctype>

namespace dm::engine {

    using namespace dm::core;
    using json = nlohmann::json;

    Civilization JsonLoader::parse_civilization(const std::string& civ_str) {
        if (civ_str == "LIGHT") return Civilization::LIGHT;
        if (civ_str == "WATER") return Civilization::WATER;
        if (civ_str == "DARKNESS") return Civilization::DARKNESS;
        if (civ_str == "FIRE") return Civilization::FIRE;
        if (civ_str == "NATURE") return Civilization::NATURE;
        if (civ_str == "ZERO") return Civilization::ZERO;
        return Civilization::NONE;
    }

    CardType JsonLoader::parse_card_type(const std::string& type_str) {
        if (type_str == "CREATURE") return CardType::CREATURE;
        if (type_str == "SPELL") return CardType::SPELL;
        if (type_str == "EVOLUTION_CREATURE") return CardType::EVOLUTION_CREATURE;
        if (type_str == "CROSS_GEAR") return CardType::CROSS_GEAR;
        if (type_str == "CASTLE") return CardType::CASTLE;
        if (type_str == "PSYCHIC_CREATURE") return CardType::PSYCHIC_CREATURE;
        if (type_str == "GR_CREATURE") return CardType::GR_CREATURE;
        if (type_str == "TAMASEED") return CardType::TAMASEED;
        return CardType::CREATURE;
    }

    // Helper to convert Legacy ActionDef to New CommandDef
    static CommandDef convert_legacy_action(const ActionDef& act) {
        CommandDef cmd;
        cmd.optional = act.optional;
        cmd.amount = act.value1;
        cmd.str_param = act.str_val;
        cmd.target_filter = act.filter;
        cmd.target_group = act.scope;

        // Map Legacy Scope to Zone/Target if necessary
        if (act.source_zone.empty() == false) cmd.from_zone = act.source_zone;
        if (act.destination_zone.empty() == false) cmd.to_zone = act.destination_zone;

        switch (act.type) {
            case EffectActionType::DRAW_CARD:
                cmd.type = CommandType::DRAW_CARD;
                break;
            case EffectActionType::ADD_MANA:
                cmd.type = CommandType::MANA_CHARGE;
                break;
            case EffectActionType::DESTROY:
                cmd.type = CommandType::DESTROY;
                break;
            case EffectActionType::RETURN_TO_HAND:
                cmd.type = CommandType::RETURN_TO_HAND;
                break;
            case EffectActionType::TAP:
                cmd.type = CommandType::TAP;
                break;
            case EffectActionType::UNTAP:
                cmd.type = CommandType::UNTAP;
                break;
            case EffectActionType::MODIFY_POWER:
                cmd.type = CommandType::POWER_MOD;
                break;
            case EffectActionType::BREAK_SHIELD:
                cmd.type = CommandType::BREAK_SHIELD;
                break;
            case EffectActionType::DISCARD:
                cmd.type = CommandType::DISCARD;
                break;
            case EffectActionType::SEARCH_DECK:
                cmd.type = CommandType::SEARCH_DECK;
                break;
            case EffectActionType::GRANT_KEYWORD:
                cmd.type = CommandType::ADD_KEYWORD;
                break;
            case EffectActionType::SEND_TO_MANA:
                cmd.type = CommandType::MANA_CHARGE; // Usually Send to Mana
                break;
            case EffectActionType::MOVE_CARD:
                cmd.type = CommandType::TRANSITION;
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

    // Helper to convert CardData (JSON) to CardDefinition (Engine)
    // Forward declare to allow recursive calls
    static CardDefinition convert_to_def(const CardData& data);

    static CardDefinition convert_to_def(const CardData& data) {
        CardDefinition def;
        def.id = static_cast<CardID>(data.id);
        def.name = data.name;

        def.civilizations = data.civilizations;

        def.type = JsonLoader::parse_card_type(data.type);
        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

        // Keywords from explicit 'keywords' block
        if (data.keywords.has_value()) {
            const auto& kws = *data.keywords;
            for (const auto& [key, val] : kws) {
                if (val) {
                    auto kw = string_to_keyword(key);
                    if (kw) {
                        def.keywords.add(*kw);
                    }
                }
            }
        }

        // Auto-infer keywords from effects
        for (const auto& eff : data.effects) {
            if (eff.trigger == TriggerType::ON_PLAY) def.keywords.add(dm::core::Keyword::CIP);
            if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.add(dm::core::Keyword::AT_ATTACK);
            if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.add(dm::core::Keyword::DESTRUCTION);
            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.add(dm::core::Keyword::SHIELD_TRIGGER);

            if (eff.trigger == TriggerType::PASSIVE_CONST) {
                 for (const auto& action : eff.actions) {
                    std::string k_str = action.str_val;
                    std::transform(k_str.begin(), k_str.end(), k_str.begin(), ::tolower);
                    auto kw = string_to_keyword(k_str);
                    if (kw) {
                        def.keywords.add(*kw);
                        if (*kw == dm::core::Keyword::POWER_ATTACKER) {
                            def.power_attacker_bonus = action.value1;
                        }
                    }
                 }
            }
        }

        if (data.revolution_change_condition.has_value()) {
            def.keywords.add(dm::core::Keyword::REVOLUTION_CHANGE);
            def.revolution_change_condition = data.revolution_change_condition;
        }

        def.reaction_abilities = data.reaction_abilities;
        def.cost_reductions = data.cost_reductions;

        // Populate Effect Definitions for Engine with Conversion
        def.effects.clear();
        def.effects.reserve(data.effects.size());
        for (const auto& eff : data.effects) {
            EffectDef engine_eff = eff;

            // Legacy Conversion: If commands are empty but actions exist
            if (engine_eff.commands.empty() && !engine_eff.actions.empty()) {
                // std::cout << "Converting Legacy Actions. Count: " << engine_eff.actions.size() << std::endl;
                engine_eff.commands.reserve(engine_eff.actions.size());
                for (const auto& act : engine_eff.actions) {
                    CommandDef cmd = convert_legacy_action(act);
                    // std::cout << "Converted ActionType " << (int)act.type << " to CommandType " << (int)cmd.type << std::endl;
                    if (cmd.type != CommandType::NONE) {
                        engine_eff.commands.push_back(cmd);
                    }
                }
            }
            def.effects.push_back(engine_eff);
        }

        // Populate Metamorph Abilities with Conversion
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

        // Recursive Spell Side
        if (data.spell_side) {
            def.spell_side = std::make_shared<CardDefinition>(convert_to_def(*data.spell_side));
        }

        // AI Metadata
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        return def;
    }

    void JsonLoader::load_cards(const std::string& filepath, std::map<CardID, CardDefinition>& card_db) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open card data file: " << filepath << std::endl;
            return;
        }

        json j;
        try {
            file >> j;
        } catch (json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            return;
        }

        if (!j.is_array()) {
            std::cerr << "JSON root must be an array." << std::endl;
            return;
        }

        for (const auto& item : j) {
            try {
                CardData data = item.get<CardData>();
                card_db[data.id] = convert_to_def(data);

            } catch (json::exception& e) {
                std::cerr << "Error parsing card definition: " << e.what() << std::endl;
            }
        }
    }

}
