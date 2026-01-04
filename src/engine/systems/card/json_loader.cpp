#include "json_loader.hpp"
#include "card_registry.hpp"
#include "core/card_json_types.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace dm::engine {

    using namespace dm::core;

    std::map<CardID, CardDefinition> JsonLoader::load_cards(const std::string& filepath) {
        std::map<CardID, CardDefinition> result;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file: " << filepath << std::endl;
            return result;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_str = buffer.str();

        // 1. Load into Registry (Single Source of Truth)
        CardRegistry::load_from_json(json_str);

        // 2. Parse locally to identify which cards were loaded and retrieve them from Registry
        try {
            auto j = nlohmann::json::parse(json_str);
            const auto& registry_defs = CardRegistry::get_all_definitions();

            auto process_item = [&](const nlohmann::json& item) {
                // We only need the ID to fetch from registry
                if (item.contains("id")) {
                    int id = item["id"].get<int>();
                    CardID card_id = static_cast<CardID>(id);

                    if (registry_defs.count(card_id)) {
                        result[card_id] = registry_defs.at(card_id);
                    }
                }
            };

            if (j.is_array()) {
                for (const auto& item : j) {
                    process_item(item);
                }
            } else {
                process_item(j);
            }

        } catch (const std::exception& e) {
            std::cerr << "JsonLoader Error: " << e.what() << std::endl;
        }

        return result;
    }

}
