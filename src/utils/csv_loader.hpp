#pragma once
#include "../core/card_def.hpp"
#include <string>
#include <vector>
#include <map>

namespace dm::utils {

    class CsvLoader {
    public:
        // Loads cards from a CSV file and returns a map of CardID to CardDefinition
        static std::map<dm::core::CardID, dm::core::CardDefinition> load_cards(const std::string& filepath);
    };

}
