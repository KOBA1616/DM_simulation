#include "json_loader.hpp"
#include "card_registry.hpp"
#include "core/card_json_types.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace dm::engine {

    using namespace dm::core;

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
            case EffectPrimitive::DRAW_CARD:
                cmd.type = CommandType::DRAW_CARD;
                break;
            case EffectPrimitive::ADD_MANA:
                cmd.type = CommandType::MANA_CHARGE;
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
                cmd.type = CommandType::MANA_CHARGE; // Usually Send to Mana
                break;
            case EffectPrimitive::MOVE_CARD:
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

             // Meta Counter Play (e.g. Oriot Judgement)
            if (kws.count("meta_counter_play") && kws.at("meta_counter_play")) {
                 def.keywords.meta_counter_play = true;
                 // Assuming implicit effect generation is handled elsewhere or by specific keywords
            }

            if (kws.count("hyper_energy") && kws.at("hyper_energy")) def.keywords.hyper_energy = true;

            // Friend Burst
            if (kws.count("friend_burst") && kws.at("friend_burst")) {
                def.keywords.friend_burst = true;
                EffectDef fb_effect;
                fb_effect.trigger = TriggerType::ON_PLAY;

                ActionDef fb_action;
                fb_action.type = EffectPrimitive::FRIEND_BURST;
                fb_action.scope = TargetScope::TARGET_SELECT;
                fb_action.optional = true;
                fb_action.filter.owner = "SELF";
                fb_action.filter.zones = {"BATTLE_ZONE"};
                fb_action.filter.types = {"CREATURE"};
                fb_action.filter.is_tapped = false;
                fb_action.filter.count = 1;

                fb_effect.actions.push_back(fb_action);

                // Add Command counterpart for Friend Burst
                CommandDef fb_cmd;
                fb_cmd.type = CommandType::FRIEND_BURST;
                fb_cmd.target_group = TargetScope::TARGET_SELECT;
                fb_cmd.optional = true;
                fb_cmd.target_filter = fb_action.filter;
                fb_effect.commands.push_back(fb_cmd);

                def.effects.push_back(fb_effect);
            }

            // Mega Last Burst
            if (kws.count("mega_last_burst") && kws.at("mega_last_burst")) {
                def.keywords.mega_last_burst = true;
                if (data.spell_side) {
                    EffectDef mlb_effect;
                    mlb_effect.trigger = TriggerType::ON_DESTROY;

                    ActionDef mlb_action;
                    mlb_action.type = EffectPrimitive::CAST_SPELL;
                    mlb_action.scope = TargetScope::SELF; // Use SELF to target the card itself (in graveyard)
                    mlb_action.optional = true;
                    mlb_action.cast_spell_side = true;

                    mlb_effect.actions.push_back(mlb_action);

                    // Add Command counterpart
                    CommandDef mlb_cmd;
                    mlb_cmd.type = CommandType::CAST_SPELL; // Assuming macro exists
                    // Actually, casting spell side might be a specific mutation or flow.
                    // For now, let's assume CAST_SPELL primitive maps.
                    mlb_effect.commands.push_back(mlb_cmd);

                    def.effects.push_back(mlb_effect);
                }
            }
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
                for (const auto& action : eff.actions) {
                    if (action.str_val == "BLOCKER") def.keywords.blocker = true;
                    if (action.str_val == "SPEED_ATTACKER") def.keywords.speed_attacker = true;
                    if (action.str_val == "SLAYER") def.keywords.slayer = true;
                    if (action.str_val == "DOUBLE_BREAKER") def.keywords.double_breaker = true;
                    if (action.str_val == "TRIPLE_BREAKER") def.keywords.triple_breaker = true;
                    if (action.str_val == "POWER_ATTACKER") {
                        def.keywords.power_attacker = true;
                        def.power_attacker_bonus = action.value1;
                    }
                    if (action.str_val == "EVOLUTION") def.keywords.evolution = true;
                    if (action.str_val == "MACH_FIGHTER") def.keywords.mach_fighter = true;
                    if (action.str_val == "G_STRIKE") def.keywords.g_strike = true;
                    if (action.str_val == "JUST_DIVER") def.keywords.just_diver = true;
                    if (action.str_val == "HYPER_ENERGY") def.keywords.hyper_energy = true;
                    if (action.str_val == "META_COUNTER") def.keywords.meta_counter_play = true;
                    if (action.str_val == "SHIELD_BURN") def.keywords.shield_burn = true;
                    if (action.str_val == "UNBLOCKABLE") def.keywords.unblockable = true;
                }
            }
        }

        // AI Metadata
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        return def;
    }

    std::map<CardID, CardDefinition> JsonLoader::load_cards(const std::string& filepath) {
        std::map<CardID, CardDefinition> result;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file: " << filepath << std::endl;
            return result;
        }

        try {
            nlohmann::json j;
            file >> j;

            if (j.is_array()) {
                for (const auto& item : j) {
                    CardData card = item.get<CardData>();
                    // Backward compatibility: allow single "civilization" field
                    if (card.civilizations.empty() && item.contains("civilization")) {
                         // Use enum parsing via JSON library for consistent mapping
                         card.civilizations.push_back(item.at("civilization").get<Civilization>());
                    }
                    // 1. Add to Registry (New System)
                    CardRegistry::load_from_json(item.dump());

                    // 2. Convert to Old Def (Old System compatibility)
                    result[static_cast<CardID>(card.id)] = convert_to_def(card);
                }
            } else {
                CardData card = j.get<CardData>();
                if (card.civilizations.empty() && j.contains("civilization")) {
                     // Use enum parsing via JSON library for consistent mapping
                     card.civilizations.push_back(j.at("civilization").get<Civilization>());
                }
                CardRegistry::load_from_json(j.dump());
                result[static_cast<CardID>(card.id)] = convert_to_def(card);
            }

        } catch (const std::exception& e) {
            std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
        }

        return result;
    }

}
