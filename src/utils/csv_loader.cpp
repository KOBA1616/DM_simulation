#include "csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace dm::utils {

    using namespace dm::core;

    // Helper to trim whitespace
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(' ');
        if (std::string::npos == first) {
            return "";
        }
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }

    // Helper to parse Civilization
    static Civilization parse_civilization(const std::string& str) {
        std::string s = trim(str);
        if (s == "LIGHT") return Civilization::LIGHT;
        if (s == "WATER") return Civilization::WATER;
        if (s == "DARKNESS") return Civilization::DARKNESS;
        if (s == "FIRE") return Civilization::FIRE;
        if (s == "NATURE") return Civilization::NATURE;
        if (s == "ZERO") return Civilization::ZERO;
        return Civilization::NONE;
    }

    // Helper to parse CardType
    static CardType parse_type(const std::string& str) {
        std::string s = trim(str);
        if (s == "CREATURE") return CardType::CREATURE;
        if (s == "SPELL") return CardType::SPELL;
        if (s == "EVOLUTION_CREATURE") return CardType::EVOLUTION_CREATURE;
        if (s == "CROSS_GEAR") return CardType::CROSS_GEAR;
        if (s == "CASTLE") return CardType::CASTLE;
        if (s == "PSYCHIC_CREATURE") return CardType::PSYCHIC_CREATURE;
        if (s == "GR_CREATURE") return CardType::GR_CREATURE;
        return CardType::CREATURE; // Default
    }

    std::map<CardID, CardDefinition> CsvLoader::load_cards(const std::string& filepath) {
        std::map<CardID, CardDefinition> card_db;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open card database: " + filepath);
        }

        std::string line;
        // Skip header
        std::getline(file, line);

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::stringstream ss(line);
            std::string cell;
            CardDefinition def = {}; // Initialize with defaults
            
            // 1. ID
            if (!std::getline(ss, cell, ',')) continue;
            try {
                def.id = static_cast<CardID>(std::stoi(cell));
            } catch (...) { continue; }

            // 2. Name
            if (!std::getline(ss, def.name, ',')) continue;

            // 3. Civilization
            if (!std::getline(ss, cell, ',')) continue;
            def.civilization = parse_civilization(cell);

            // 4. Type
            if (!std::getline(ss, cell, ',')) continue;
            def.type = parse_type(cell);

            // 5. Cost
            if (!std::getline(ss, cell, ',')) continue;
            try {
                def.cost = std::stoi(cell);
            } catch (...) { def.cost = 0; }

            // 6. Power
            if (!std::getline(ss, cell, ',')) continue;
            try {
                def.power = std::stoi(cell);
            } catch (...) { def.power = 0; }

            // 7. Races (semicolon separated)
            if (!std::getline(ss, cell, ',')) continue;
            std::stringstream race_ss(cell);
            std::string race;
            while (std::getline(race_ss, race, ';')) {
                def.races.push_back(trim(race));
            }

            // 8. Keywords (semicolon separated)
            if (!std::getline(ss, cell, ',')) continue;
            // Parse keywords and set flags in def.keywords
            std::string keywords = trim(cell);
            if (keywords.find("BLOCKER") != std::string::npos) def.keywords.blocker = true;
            if (keywords.find("SPEED_ATTACKER") != std::string::npos) def.keywords.speed_attacker = true;
            if (keywords.find("SHIELD_TRIGGER") != std::string::npos) def.keywords.shield_trigger = true;
            if (keywords.find("G_STRIKE") != std::string::npos) def.keywords.g_strike = true;
            if (keywords.find("MACH_FIGHTER") != std::string::npos) def.keywords.mach_fighter = true;
            if (keywords.find("REVOLUTION_CHANGE") != std::string::npos) def.keywords.revolution_change = true;
            if (keywords.find("G_ZERO") != std::string::npos) def.keywords.g_zero = true;
            if (keywords.find("EVOLUTION") != std::string::npos) def.keywords.evolution = true;
            
            card_db[def.id] = def;
        }

        return card_db;
    }

}
