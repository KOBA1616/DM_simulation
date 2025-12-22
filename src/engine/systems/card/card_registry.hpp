#pragma once
#include "core/card_json_types.hpp"
#include "core/card_def.hpp"
#include <map>
#include <string>
#include <memory>

namespace dm::engine {
    class CardRegistry {
    public:
        static void load_from_json(const std::string& json_str);
        static const dm::core::CardData* get_card_data(int id);
        static const std::map<int, dm::core::CardData>& get_all_cards();
        static const std::map<dm::core::CardID, dm::core::CardDefinition>& get_all_definitions();
        static std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> get_all_definitions_ptr();
        static void clear();
    private:
        static std::map<int, dm::core::CardData> cards;
        static std::shared_ptr<std::map<dm::core::CardID, dm::core::CardDefinition>> definitions_ptr;
        static dm::core::CardDefinition convert_to_def(const dm::core::CardData& data);
    };
}
