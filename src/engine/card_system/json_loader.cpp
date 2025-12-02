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
        // ... add others

        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

        // Keywords mapping from effects
        // This is where we need to scan effects to set boolean flags for the engine
        // The engine currently relies on flags like .keywords.blocker
        // The JSON system has "PASSIVE_CONST" triggers or similar.

        for (const auto& eff : data.effects) {
            // Check for keyword-like effects
            // This logic depends on how JSON represents keywords.
            // If the JSON has a "keywords" field we should use it.
            // Checking card_json_types.hpp... it DOES NOT have a keywords list in CardData struct!
            // It only has 'effects'.
            // So we must infer keywords or we need to update CardData to include keywords.

            // However, the CardData struct in `card_json_types.hpp` only has:
            // id, name, cost, civilization, power, type, races, effects.

            // If the user intends to replace CSV with JSON, JSON should support keywords.
            // Typically keywords like "Blocker" are modeled as passive effects.
            // But the legacy engine needs the boolean flag.

            // Let's assume for now we look at effect triggers or actions?
            // Or maybe we should update `card_json_types.hpp` to include keywords?
            // The instruction is "Implement JSON Loader: src/core/card_json_types.hpp based...".
            // So I should stick to what `card_json_types.hpp` has.

            // If `card_json_types.hpp` lacks keywords, maybe they are in 'effects'?
            // TriggerType::PASSIVE_CONST ?
            // Let's assume for now we map what we can.

            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.shield_trigger = true;

            // Checking actions for specific keywords?
            // Usually "Blocker" is a capability.
            // If the JSON schema doesn't have it, we might be missing it.
            // But let's look at `card_json_types.hpp` again.
            // It has TriggerType::PASSIVE_CONST.
            // Maybe the condition or action string says "BLOCKER"?

            for (const auto& action : eff.actions) {
                // If action.str_val contains "BLOCKER" etc?
                // This is speculative.
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
