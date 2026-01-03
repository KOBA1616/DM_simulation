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

    // Helper methods for parsing are removed in favor of direct JSON-Enum mapping via NLOHMANN_JSON_SERIALIZE_ENUM

};

} // namespace engine
} // namespace dm
