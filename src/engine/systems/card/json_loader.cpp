#include "json_loader.hpp"
#include "card_registry.hpp"
#include "core/card_json_types.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace dm::engine {

    using namespace dm::core;

    // Helper to convert CardData (JSON) to CardDefinition (Engine)
    // Forward declare to allow recursive calls
    static CardDefinition convert_to_def(const CardData& data);

    static CardDefinition convert_to_def(const CardData& data) {
        CardDefinition def;
        def.id = static_cast<CardID>(data.id);
        def.name = data.name;

        // Civilization mapping
        def.civilizations.clear();
        for (const auto& civ_str : data.civilizations) {
            if (civ_str == "LIGHT") def.civilizations.push_back(Civilization::LIGHT);
            else if (civ_str == "WATER") def.civilizations.push_back(Civilization::WATER);
            else if (civ_str == "DARKNESS") def.civilizations.push_back(Civilization::DARKNESS);
            else if (civ_str == "FIRE") def.civilizations.push_back(Civilization::FIRE);
            else if (civ_str == "NATURE") def.civilizations.push_back(Civilization::NATURE);
            else if (civ_str == "ZERO") def.civilizations.push_back(Civilization::ZERO);
        }
        if (def.civilizations.empty()) def.civilizations.push_back(Civilization::ZERO); // Default to Colorless if empty?

        // Type mapping
        if (data.type == "CREATURE") def.type = CardType::CREATURE;
        else if (data.type == "SPELL") def.type = CardType::SPELL;
        else if (data.type == "EVOLUTION_CREATURE") def.type = CardType::EVOLUTION_CREATURE;
        else if (data.type == "TAMASEED") def.type = CardType::TAMASEED;
        else if (data.type == "CROSS_GEAR") def.type = CardType::CROSS_GEAR;
        else if (data.type == "CASTLE") def.type = CardType::CASTLE;
        else if (data.type == "PSYCHIC_CREATURE") def.type = CardType::PSYCHIC_CREATURE;
        else if (data.type == "GR_CREATURE") def.type = CardType::GR_CREATURE;

        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

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
            // Add other keywords as needed
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
                        card.civilizations.push_back(item.at("civilization").get<std::string>());
                    }
                    // 1. Add to Registry (New System)
                    CardRegistry::load_from_json(item.dump());

                    // 2. Convert to Old Def (Old System compatibility)
                    result[static_cast<CardID>(card.id)] = convert_to_def(card);
                }
            } else {
                CardData card = j.get<CardData>();
                if (card.civilizations.empty() && j.contains("civilization")) {
                    card.civilizations.push_back(j.at("civilization").get<std::string>());
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
