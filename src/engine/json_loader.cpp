#include "json_loader.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../core/card_def.hpp"

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

    // Helper for TriggerType, TargetScope, EffectActionType are handled by NLOHMANN macros or custom if needed.
    // Since we use direct JSON mapping for CardData structs, we might not need many manual parsers here
    // if the structs are aligned.

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
                // If the JSON structure perfectly matches CardData, we can deserialise directly.
                // However, 'civilization' field might be a string in old JSONs and list in new ones?
                // The user said strictly replace. So we assume the JSONs will be updated or we handle it here.
                // To support transition: check if "civilization" (singular) exists or "civilizations" (plural).
                // Or if "civilizations" can be a string or list.

                // Let's manually parse primarily to handle the conversion from simple JSON types to Engine Enums if Macros aren't enough
                // actually CardData struct uses std::string for Enums, so it's fine.

                CardData data;

                // Compatibility Hack: Allow 'civilization' (string) key if 'civilizations' is missing
                if (item.contains("civilization") && !item.contains("civilizations")) {
                    json item_copy = item;
                    std::string civ = item_copy["civilization"].get<std::string>();
                    item_copy["civilizations"] = std::vector<std::string>{civ};
                    item_copy.erase("civilization");
                    data = item_copy.get<CardData>();
                } else if (item.contains("civilizations") && item["civilizations"].is_string()) {
                    // Handle case where "civilizations" might be a single string by mistake?
                    // Or if we renamed the key but kept string value.
                    json item_copy = item;
                    std::string civ = item_copy["civilizations"].get<std::string>();
                    item_copy["civilizations"] = std::vector<std::string>{civ};
                    data = item_copy.get<CardData>();
                } else {
                    data = item.get<CardData>();
                }

                CardDefinition def;
                def.id = data.id;
                def.name = data.name;
                def.cost = data.cost;
                def.power = data.power;
                def.races = data.races;

                // Parse Enums
                def.type = parse_card_type(data.type);

                def.civilizations.clear();
                for (const auto& civ_str : data.civilizations) {
                    def.civilizations.push_back(parse_civilization(civ_str));
                }

                // Process Effects & Keywords logic
                // (Previously existing logic for inferring keywords from effects)
                def.keywords = {};

                // Explicit keyword overrides from JSON
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
                     if (k.count("shield_trigger")) def.keywords.shield_trigger = k.at("shield_trigger");
                     if (k.count("evolution")) def.keywords.evolution = k.at("evolution");
                     if (k.count("neo")) def.keywords.neo = k.at("neo"); // Added
                     if (k.count("just_diver")) def.keywords.just_diver = k.at("just_diver");
                     if (k.count("hyper_energy")) def.keywords.hyper_energy = k.at("hyper_energy");
                     if (k.count("shield_burn")) def.keywords.shield_burn = k.at("shield_burn");
                     if (k.count("untap_in")) def.keywords.untap_in = k.at("untap_in");
                }

                // Auto-infer keywords from effects (Legacy support, though user prefers explicit)
                for (const auto& eff : data.effects) {
                    if (eff.trigger == TriggerType::ON_PLAY) def.keywords.cip = true;
                    if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.at_attack = true;
                    if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.destruction = true;
                    if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.shield_trigger = true;

                    // PASSIVE_CONST for simple keywords
                    if (eff.trigger == TriggerType::PASSIVE_CONST) {
                         for (const auto& act : eff.actions) {
                              if (act.str_val == "SPEED_ATTACKER") def.keywords.speed_attacker = true;
                              if (act.str_val == "BLOCKER") def.keywords.blocker = true;
                              if (act.str_val == "SLAYER") def.keywords.slayer = true;
                              if (act.str_val == "DOUBLE_BREAKER") def.keywords.double_breaker = true;
                              if (act.str_val == "TRIPLE_BREAKER") def.keywords.triple_breaker = true;
                              if (act.str_val == "MACH_FIGHTER") def.keywords.mach_fighter = true;
                              if (act.str_val == "POWER_ATTACKER") {
                                   def.keywords.power_attacker = true;
                                   def.power_attacker_bonus = act.value1;
                              }
                         }
                    }
                }

                // Revolution Change
                if (data.revolution_change_condition.has_value()) {
                    def.keywords.revolution_change = true;
                    def.revolution_change_condition = data.revolution_change_condition;
                }

                // Reaction Abilities
                def.reaction_abilities = data.reaction_abilities;
                for (const auto& reaction : data.reaction_abilities) {
                    if (reaction.type == ReactionType::NINJA_STRIKE) {
                        // Mark a flag if needed, or HandTrigger
                    }
                }

                // Modes (Not fully parsed in previous version, leaving as empty/placeholder)

                card_db[def.id] = def;

            } catch (json::exception& e) {
                std::cerr << "Error parsing card definition: " << e.what() << std::endl;
            }
        }
    }

}
