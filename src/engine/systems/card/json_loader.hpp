#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include "../../../core/card_def.hpp"
#include "../../../core/card_json_types.hpp"
#include <nlohmann/json.hpp>

namespace dm {
namespace engine {

/**
 * @brief Loads card definitions from a JSON file.
 */
class JsonLoader {
public:
    // Loads cards from a JSON file (array of CardData objects)
    // and returns a map (CardID -> CardDefinition)
    static std::map<core::CardID, core::CardDefinition> load_cards(const std::string& filepath);

    // Helper to parse Civilization from string
    static core::Civilization parse_civilization(const std::string& civ_str);

    // Helper to parse CardType from string
    static core::CardType parse_card_type(const std::string& type_str);

    // Parses a single CardData object into a CardDefinition (implementation should match convert_to_def)
    // Note: implementation currently relies on static helper convert_to_def in cpp,
    // if this is needed publicly it should be implemented.
    // For now I'll keep the declaration if it was there, but it seems missing in cpp.
    // However, to fix the build errors for parse_civilization/type, I add those above.
    static core::CardDefinition parse_card_def(const core::CardData& card_data);
};

} // namespace engine
} // namespace dm
