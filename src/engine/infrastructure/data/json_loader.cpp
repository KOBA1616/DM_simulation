#include "json_loader.hpp"
#include <iostream>
#include <fstream>

namespace dm::engine::infrastructure {

    // Helper to map string keywords to bitset
    static void map_keywords(const std::map<std::string, bool>& src, core::CardKeywords& dest) {
        if (src.count("BLOCKER") && src.at("BLOCKER")) dest.blocker = true;
        if (src.count("SPEED_ATTACKER") && src.at("SPEED_ATTACKER")) dest.speed_attacker = true;
        if (src.count("SLAYER") && src.at("SLAYER")) dest.slayer = true;
        if (src.count("DOUBLE_BREAKER") && src.at("DOUBLE_BREAKER")) dest.double_breaker = true;
        if (src.count("TRIPLE_BREAKER") && src.at("TRIPLE_BREAKER")) dest.triple_breaker = true;
        if (src.count("WORLD_BREAKER") && src.at("WORLD_BREAKER")) dest.world_breaker = true;
        if (src.count("SHIELD_TRIGGER") && src.at("SHIELD_TRIGGER")) dest.shield_trigger = true;
        if (src.count("POWER_ATTACKER") && src.at("POWER_ATTACKER")) dest.power_attacker = true;
        if (src.count("NEO") && src.at("NEO")) dest.neo = true;
        if (src.count("G_NEO") && src.at("G_NEO")) dest.g_neo = true;
        if (src.count("G_ZERO") && src.at("G_ZERO")) dest.g_zero = true;
        if (src.count("REVOLUTION_CHANGE") && src.at("REVOLUTION_CHANGE")) dest.revolution_change = true;
        if (src.count("MACH_FIGHTER") && src.at("MACH_FIGHTER")) dest.mach_fighter = true;
        if (src.count("G_STRIKE") && src.at("G_STRIKE")) dest.g_strike = true;
        if (src.count("EVOLUTION") && src.at("EVOLUTION")) dest.evolution = true;
        if (src.count("JUST_DIVER") && src.at("JUST_DIVER")) dest.just_diver = true;
        if (src.count("UNBLOCKABLE") && src.at("UNBLOCKABLE")) dest.unblockable = true;
        if (src.count("FRIEND_BURST") && src.at("FRIEND_BURST")) dest.friend_burst = true;
        if (src.count("EX_LIFE") && src.at("EX_LIFE")) dest.ex_life = true;
        if (src.count("MEGA_LAST_BURST") && src.at("MEGA_LAST_BURST")) dest.mega_last_burst = true;
        if (src.count("MUST_BE_CHOSEN") && src.at("MUST_BE_CHOSEN")) dest.must_be_chosen = true;

        // Triggers implicit in keywords
        if (src.count("AT_START_OF_TURN") && src.at("AT_START_OF_TURN")) dest.at_start_of_turn = true;
        if (src.count("AT_END_OF_TURN") && src.at("AT_END_OF_TURN")) dest.at_end_of_turn = true;
        if (src.count("AT_ATTACK") && src.at("AT_ATTACK")) dest.at_attack = true;
        if (src.count("AT_BLOCK") && src.at("AT_BLOCK")) dest.at_block = true;
        if (src.count("ON_PLAY") && src.at("ON_PLAY")) dest.cip = true;
        if (src.count("ON_DESTROY") && src.at("ON_DESTROY")) dest.destruction = true;
    }

    static core::CardDefinition convert_to_def(const core::CardData& data) {
        core::CardDefinition def;
        def.id = data.id;
        def.name = data.name;
        def.cost = data.cost;
        def.power = data.power;
        def.civilizations = data.civilizations;
        def.races = data.races;
        def.type = data.type;

        // Copy effects
        def.effects = data.effects;

        // Map keywords
        if (data.keywords.has_value()) {
            map_keywords(data.keywords.value(), def.keywords);
        }

        // Static abilities
        def.static_abilities = data.static_abilities;
        def.metamorph_abilities = data.metamorph_abilities;
        def.evolution_condition = data.evolution_condition;
        def.revolution_change_condition = data.revolution_change_condition;
        def.reaction_abilities = data.reaction_abilities;
        def.cost_reductions = data.cost_reductions;
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        if (data.spell_side) {
            def.spell_side = std::make_shared<core::CardDefinition>(convert_to_def(*data.spell_side));
        }

        return def;
    }

    static std::map<core::CardID, core::CardDefinition> parse_json_array(const nlohmann::json& j) {
        std::map<core::CardID, core::CardDefinition> db;
        for (const auto& item : j) {
            try {
                 // Deserialize to CardData first
                 core::CardData data = item.get<core::CardData>();
                 // Convert to CardDefinition
                 core::CardDefinition def = convert_to_def(data);
                 db[def.id] = def;
            } catch (const std::exception& e) {
                 std::cerr << "Error parsing card: " << e.what() << std::endl;
            }
        }
        return db;
    }

    std::map<core::CardID, core::CardDefinition> JsonLoader::load_cards(const std::string& filepath) {
        try {
            std::ifstream f(filepath);
            if (!f.is_open()) {
                std::cerr << "Failed to open " << filepath << std::endl;
                return {};
            }
            nlohmann::json j;
            f >> j;
            return parse_json_array(j);
        } catch (const std::exception& e) {
            std::cerr << "JSON Load Error: " << e.what() << std::endl;
            return {};
        }
    }

    std::map<core::CardID, core::CardDefinition> JsonLoader::load_cards_from_string(const std::string& json_str) {
        try {
            nlohmann::json j = nlohmann::json::parse(json_str);
            return parse_json_array(j);
        } catch (const std::exception& e) {
            std::cerr << "JSON Parse Error: " << e.what() << std::endl;
            return {};
        }
    }

} // namespace dm::engine::infrastructure
