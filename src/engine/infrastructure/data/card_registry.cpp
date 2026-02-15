#include "card_registry.hpp"
#include "json_loader.hpp"
#include <iostream>

namespace dm::engine::infrastructure {

    std::map<int, dm::core::CardData> CardRegistry::cards;
    std::shared_ptr<std::map<dm::core::CardID, dm::core::CardDefinition>> CardRegistry::definitions_ptr;

    void CardRegistry::load_from_json(const std::string& json_str) {
        // Assume json_str is content (called by Python binding register_card_data)
        // If it were a filepath, JsonLoader::load_cards should be used directly.
        auto defs = JsonLoader::load_cards_from_string(json_str);
        definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(defs);

        // Populate legacy 'cards' map if needed (CardData)
        cards.clear();
    }

    const dm::core::CardData* CardRegistry::get_card_data(int id) {
        if (cards.count(id)) return &cards.at(id);
        return nullptr;
    }

    const std::map<int, dm::core::CardData>& CardRegistry::get_all_cards() {
        return cards;
    }

    const std::map<dm::core::CardID, dm::core::CardDefinition>& CardRegistry::get_all_definitions() {
        if (!definitions_ptr) {
            static std::map<dm::core::CardID, dm::core::CardDefinition> empty;
            return empty;
        }
        return *definitions_ptr;
    }

    std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> CardRegistry::get_all_definitions_ptr() {
        return definitions_ptr;
    }

    void CardRegistry::clear() {
        cards.clear();
        definitions_ptr.reset();
    }

    dm::core::CardDefinition CardRegistry::convert_to_def(const dm::core::CardData& data) {
        dm::core::CardDefinition def;
        def.id = data.id;
        def.name = data.name;
        def.cost = data.cost;
        def.power = data.power;
        // def.civilization = data.civilization; // CardData has single, Def has vector
        if (!data.civilizations.empty()) def.civilizations = data.civilizations;
        def.races = data.races;
        def.type = data.type;
        // def.effects = data.effects;
        return def;
    }
}
