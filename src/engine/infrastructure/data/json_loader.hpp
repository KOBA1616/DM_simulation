#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include "core/card_def.hpp"
#include "core/card_json_types.hpp"
#include <nlohmann/json.hpp>

namespace dm::engine::infrastructure {

/**
 * @brief Loads card definitions from a JSON file.
 */
class JsonLoader {
public:
    // Loads cards from a JSON file (array of CardData objects)
    // and returns a map (CardID -> CardDefinition)
    static std::map<core::CardID, core::CardDefinition> load_cards(const std::string& filepath);

    // Loads cards from a JSON string content
    static std::map<core::CardID, core::CardDefinition> load_cards_from_string(const std::string& json_str);

};

} // namespace dm::engine::infrastructure
