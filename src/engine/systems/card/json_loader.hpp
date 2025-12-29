#pragma once
#include "core/card_def.hpp"
#include <string>
#include <map>
#include <vector>

namespace dm::engine {
    class JsonLoader {
    public:
        // Reads a JSON file containing an array of card definitions
        // and returns a map compatible with CsvLoader (CardID -> CardDefinition)
        // Note: This bridges the new JSON "CardData" to the old "CardDefinition" struct
        // used by the engine, or we might need to update the engine to use CardData.
        // For now, let's assume we populate CardRegistry (CardData) AND return CardDefinition map.
        static std::map<dm::core::CardID, dm::core::CardDefinition> load_cards(const std::string& filepath);

        // Helper methods for internal use or testing
        static dm::core::Civilization parse_civilization(const std::string& civ_str);
        static dm::core::CardType parse_card_type(const std::string& type_str);
    };
}
