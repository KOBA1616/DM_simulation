#pragma once
#include "core/card_json_types.hpp"
#include <map>
#include <string>

namespace dm::engine {
    class CardRegistry {
    public:
        static void load_from_json(const std::string& json_str);
        static const dm::core::CardData* get_card_data(int id);
        static const std::map<int, dm::core::CardData>& get_all_cards();
        static void clear();
    private:
        static std::map<int, dm::core::CardData> cards;
    };
}
