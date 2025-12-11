#include "json_loader.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "core/card_def.hpp"

namespace dm::engine {

    using namespace dm::core;
    using json = nlohmann::json;

    Civilization JsonLoader::parse_civilization(const std::string& civ_str) {
        if (civ_str == "LIGHT") return Civilization::LIGHT;
        if (civ_str == "WATER") return Civilization::WATER;
        if (civ_str == "DARKNESS") return Civilization::DARKNESS;
        if (civ_str == "FIRE") return Civilization::FIRE;
        if (civ_str == "NATURE") return Civilization::NATURE;
        if (civ_str == "ZERO") return Civilization::ZERO;
        return Civilization::NONE;
    }

    CardType JsonLoader::parse_card_type(const std::string& type_str) {
        if (type_str == "CREATURE") return CardType::CREATURE;
        if (type_str == "SPELL") return CardType::SPELL;
        if (type_str == "EVOLUTION_CREATURE") return CardType::EVOLUTION_CREATURE;
        if (type_str == "CROSS_GEAR") return CardType::CROSS_GEAR;
        if (type_str == "CASTLE") return CardType::CASTLE;
        if (type_str == "PSYCHIC_CREATURE") return CardType::PSYCHIC_CREATURE;
        if (type_str == "GR_CREATURE") return CardType::GR_CREATURE;
        if (type_str == "TAMASEED") return CardType::TAMASEED;
        return CardType::CREATURE;
    }

    // Helper to convert CardData (JSON) to CardDefinition (Engine)
    // Forward declare to allow recursive calls
    static CardDefinition convert_to_def(const CardData& data);

    static CardDefinition convert_to_def(const CardData& data) {
        CardDefinition def;
        def.id = static_cast<CardID>(data.id);
        def.name = data.name;

        def.civilizations = data.civilizations;

        def.type = JsonLoader::parse_card_type(data.type);
        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

        // Keywords from explicit 'keywords' block
        if (data.keywords.has_value()) {
            const auto& k = data.keywords.value();
            if (k.count("g_zero")) def.keywords.g_zero = k.at("g_zero");
            if (k.count("revolution_change")) def.keywords.revolution_change = k.at("revolution_change");
            if (k.count("mach_fighter")) def.keywords.mach_fighter = k.at("mach_fighter");
            if (k.count("g_strike")) def.keywords.g_strike = k.at("g_strike");
            if (k.count("speed_attacker")) def.keywords.speed_attacker = k.at("speed_attacker");
            if (k.count("blocker")) def.keywords.blocker = k.at("blocker");
            if (k.count("slayer")) def.keywords.slayer = k.at("slayer");
            if (k.count("double_breaker")) def.keywords.double_breaker = k.at("double_breaker");
            if (k.count("triple_breaker")) def.keywords.triple_breaker = k.at("triple_breaker");
            if (k.count("world_breaker")) def.keywords.world_breaker = k.at("world_breaker");
            if (k.count("shield_trigger")) def.keywords.shield_trigger = k.at("shield_trigger");
            if (k.count("evolution")) def.keywords.evolution = k.at("evolution");
            if (k.count("neo")) def.keywords.neo = k.at("neo");
            if (k.count("just_diver")) def.keywords.just_diver = k.at("just_diver");
            if (k.count("hyper_energy")) def.keywords.hyper_energy = k.at("hyper_energy");
            if (k.count("shield_burn")) def.keywords.shield_burn = k.at("shield_burn");
            if (k.count("untap_in")) def.keywords.untap_in = k.at("untap_in");
            if (k.count("meta_counter_play")) def.keywords.meta_counter_play = k.at("meta_counter_play");
            if (k.count("power_attacker")) def.keywords.power_attacker = k.at("power_attacker");
        }

        // Auto-infer keywords from effects
        for (const auto& eff : data.effects) {
            if (eff.trigger == TriggerType::ON_PLAY) def.keywords.cip = true;
            if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.at_attack = true;
            if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.destruction = true;
            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.shield_trigger = true;

            if (eff.trigger == TriggerType::PASSIVE_CONST) {
                 for (const auto& act : eff.actions) {
                      if (act.str_val == "SPEED_ATTACKER") def.keywords.speed_attacker = true;
                      if (act.str_val == "BLOCKER") def.keywords.blocker = true;
                      if (act.str_val == "SLAYER") def.keywords.slayer = true;
                      if (act.str_val == "DOUBLE_BREAKER") def.keywords.double_breaker = true;
                      if (act.str_val == "TRIPLE_BREAKER") def.keywords.triple_breaker = true;
                      if (act.str_val == "WORLD_BREAKER") def.keywords.world_breaker = true;
                      if (act.str_val == "MACH_FIGHTER") def.keywords.mach_fighter = true;
                      if (act.str_val == "POWER_ATTACKER") {
                           def.keywords.power_attacker = true;
                           def.power_attacker_bonus = act.value1;
                      }
                 }
            }
        }

        if (data.revolution_change_condition.has_value()) {
            def.keywords.revolution_change = true;
            def.revolution_change_condition = data.revolution_change_condition;
        }

        def.reaction_abilities = data.reaction_abilities;

        // Populate Effect Definitions for Engine
        def.effects = data.effects;
        def.metamorph_abilities = data.metamorph_abilities;

        // Recursive Spell Side
        if (data.spell_side) {
            def.spell_side = std::make_shared<CardDefinition>(convert_to_def(*data.spell_side));
        }

        // AI Metadata
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        return def;
    }

    void JsonLoader::load_cards(const std::string& filepath, std::map<CardID, CardDefinition>& card_db) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open card data file: " << filepath << std::endl;
            return;
        }

        json j;
        try {
            file >> j;
        } catch (json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            return;
        }

        if (!j.is_array()) {
            std::cerr << "JSON root must be an array." << std::endl;
            return;
        }

        for (const auto& item : j) {
            try {
                CardData data;

                // Compatibility Hack: Allow 'civilization' (string) key if 'civilizations' is missing
                if (item.contains("civilization") && !item.contains("civilizations")) {
                    json item_copy = item;
                    std::string civ_str = item_copy["civilization"].get<std::string>();
                    // Manually map legacy string to Enum because JSON parser expects Enum string in array
                    // Actually, since we defined SERIALIZE_ENUM, passing the string "FIRE" in a vector ["FIRE"] works.
                    item_copy["civilizations"] = std::vector<std::string>{civ_str};
                    item_copy.erase("civilization");
                    data = item_copy.get<CardData>();
                } else if (item.contains("civilizations") && item["civilizations"].is_string()) {
                    json item_copy = item;
                    std::string civ_str = item_copy["civilizations"].get<std::string>();
                    item_copy["civilizations"] = std::vector<std::string>{civ_str};
                    data = item_copy.get<CardData>();
                } else {
                    data = item.get<CardData>();
                }

                card_db[data.id] = convert_to_def(data);

            } catch (json::exception& e) {
                std::cerr << "Error parsing card definition: " << e.what() << std::endl;
            }
        }
    }

}
