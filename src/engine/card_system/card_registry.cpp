#include "card_registry.hpp"
#include <nlohmann/json.hpp>
#include <iostream>

namespace dm::engine {
    
    std::map<int, dm::core::CardData> CardRegistry::cards;

    void CardRegistry::load_from_json(const std::string& json_str) {
        try {
            auto j = nlohmann::json::parse(json_str);
            if (j.is_array()) {
                for (const auto& item : j) {
                    dm::core::CardData card = item.get<dm::core::CardData>();
                    cards[card.id] = card;
                }
            } else {
                // Single object
                dm::core::CardData card = j.get<dm::core::CardData>();
                cards[card.id] = card;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading card JSON: " << e.what() << std::endl;
        }
    }

    const dm::core::CardData* CardRegistry::get_card_data(int id) {
        auto it = cards.find(id);
        if (it != cards.end()) {
            return &it->second;
        }
        std::cerr << "CardRegistry: card not found for ID " << id << ". Total cards: " << cards.size() << std::endl;
        return nullptr;
    }

    const std::map<int, dm::core::CardData>& CardRegistry::get_all_cards() {
        return cards;
    }

    void CardRegistry::clear() {
        cards.clear();
    }
}
