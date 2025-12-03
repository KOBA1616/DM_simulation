#include "json_loader.hpp"
#include "card_registry.hpp"
#include "core/card_json_types.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace dm::engine {

    using namespace dm::core;

    // Helper to convert CardData (JSON) to CardDefinition (Engine)
    static CardDefinition convert_to_def(const CardData& data) {
        CardDefinition def;
        def.id = static_cast<CardID>(data.id);
        def.name = data.name;

        // Civilization mapping
        if (data.civilization == "LIGHT") def.civilization = Civilization::LIGHT;
        else if (data.civilization == "WATER") def.civilization = Civilization::WATER;
        else if (data.civilization == "DARKNESS") def.civilization = Civilization::DARKNESS;
        else if (data.civilization == "FIRE") def.civilization = Civilization::FIRE;
        else if (data.civilization == "NATURE") def.civilization = Civilization::NATURE;
        else def.civilization = Civilization::NONE; // or ZERO

        // Type mapping
        if (data.type == "CREATURE") def.type = CardType::CREATURE;
        else if (data.type == "SPELL") def.type = CardType::SPELL;
        else if (data.type == "EVOLUTION_CREATURE") def.type = CardType::EVOLUTION_CREATURE;
        else if (data.type == "TAMASEED") def.type = CardType::TAMASEED;
        // ... add others

        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

        // Revolution Change
        if (data.revolution_change_condition.has_value()) {
            def.revolution_change_condition = data.revolution_change_condition;
            def.keywords.revolution_change = true;
        }

        // Keywords mapping from effects
        // The engine relies on boolean flags in CardKeywords.
        // We infer these from TriggerType and/or PASSIVE_CONST actions.

        for (const auto& eff : data.effects) {
            // 1. Trigger Flags
            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.shield_trigger = true;
            if (eff.trigger == TriggerType::ON_PLAY) def.keywords.cip = true;
            if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.at_attack = true;
            if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.destruction = true;

            // 2. Passive Keywords (Blocker, Speed Attacker, etc.)
            // Convention: TriggerType::PASSIVE_CONST with ActionType::NONE and str_val="KEYWORD"
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
                    if (action.str_val == "HYPER_ENERGY") def.keywords.hyper_energy = true;
                }
            }
        }

        // IMPORTANT: The requirement is to load JSON.
        // If the JSON schema is insufficient, I might need to update it,
        // but `card_json_types.hpp` was listed as "based on".
        // Let's assume for MVP we just map what we have.

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

            // We also populate the CardRegistry (the new system)
            // CardRegistry expects a string, but we parsed it.
            // Let's dump it back or use a different overload if CardRegistry had one.
            // CardRegistry::load_from_json(j.dump()); --> this works but is inefficient.

            // Better: Iterate and add to both.
            if (j.is_array()) {
                for (const auto& item : j) {
                    CardData card = item.get<CardData>();
                    // 1. Add to Registry (New System)
                    // We need to access CardRegistry::cards directly or use a helper.
                    // CardRegistry::cards is private.
                    // We can use CardRegistry::load_from_json with the dump of this item.
                    CardRegistry::load_from_json(item.dump()); // A bit hacky but works for now.

                    // 2. Convert to Old Def (Old System compatibility)
                    result[static_cast<CardID>(card.id)] = convert_to_def(card);
                }
            } else {
                CardData card = j.get<CardData>();
                CardRegistry::load_from_json(j.dump());
                result[static_cast<CardID>(card.id)] = convert_to_def(card);
            }

        } catch (const std::exception& e) {
            std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
        }

        return result;
    }

}
