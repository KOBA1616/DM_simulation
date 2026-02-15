#include "card_registry.hpp"
#include "json_loader.hpp"
#include "keyword_expander.hpp"
#include <iostream>

namespace dm::engine::infrastructure {

    std::map<int, dm::core::CardData> dm::engine::infrastructure::CardRegistry::cards;
    std::shared_ptr<std::map<dm::core::CardID, dm::core::CardDefinition>> dm::engine::infrastructure::CardRegistry::definitions_ptr;

    void dm::engine::infrastructure::CardRegistry::load_from_json(const std::string& json_str) {
        // Implementation delegated to dm::engine::infrastructure::JsonLoader for now, or direct parsing
        // Since dm::engine::infrastructure::JsonLoader expects a file path, we might need a string parser or stick to file loading.
        // Assuming this method is intended to load from file path actually, based on signature usage in main.
        // But the argument is named json_str...
        // Let's assume it loads from file path for now as that's what dm::engine::infrastructure::JsonLoader does.

        auto defs = dm::engine::infrastructure::JsonLoader::load_cards(json_str);
        definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(defs);
    }

    const dm::core::CardData* dm::engine::infrastructure::CardRegistry::get_card_data(int id) {
        if (cards.count(id)) return &cards.at(id);
        return nullptr;
    }

    const std::map<int, dm::core::CardData>& dm::engine::infrastructure::CardRegistry::get_all_cards() {
        return cards;
    }

    const std::map<dm::core::CardID, dm::core::CardDefinition>& dm::engine::infrastructure::CardRegistry::get_all_definitions() {
        if (!definitions_ptr) {
            static std::map<dm::core::CardID, dm::core::CardDefinition> empty;
            return empty;
        }
        return *definitions_ptr;
    }

    std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> dm::engine::infrastructure::CardRegistry::get_all_definitions_ptr() {
        return definitions_ptr;
    }

    void dm::engine::infrastructure::CardRegistry::clear() {
        cards.clear();
        if (definitions_ptr) definitions_ptr->clear();
    }

    dm::core::CardDefinition dm::engine::infrastructure::CardRegistry::convert_to_def(const dm::core::CardData& data) {
        dm::core::CardDefinition def;
        // Basic conversion logic would go here if not handled by dm::engine::infrastructure::JsonLoader
        return def;
    }

}
