#include "card_registry.hpp"
#include "json_loader.hpp"
#include "keyword_expander.hpp"
#include <iostream>
#include <unordered_set>
#include <functional>
#include <nlohmann/json.hpp>

namespace dm::engine::infrastructure {

    std::map<int, dm::core::CardData> dm::engine::infrastructure::CardRegistry::cards;
    std::shared_ptr<std::map<dm::core::CardID, dm::core::CardDefinition>> dm::engine::infrastructure::CardRegistry::definitions_ptr;

    void dm::engine::infrastructure::CardRegistry::load_from_json(const std::string& json_str) {
        // Correctly delegate to string loading
            // Parse and aggressively sanitize explicit JSON nulls produced by
            // legacy editors (replace null -> {}). This avoids downstream
            // nlohmann::json::value() on null exceptions during deserialization.
            try {
                nlohmann::json j = nlohmann::json::parse(json_str);
                std::function<void(nlohmann::json&)> sanitize_all_nulls = [&](nlohmann::json& node) {
                    static const std::unordered_set<std::string> int_keys = {"id", "cost", "power", "value", "value1", "value2", "amount", "duration", "slot_index", "target_slot_index", "ai_importance_score"};
                    static const std::unordered_set<std::string> string_keys = {"name", "str_val", "type", "scope", "source_zone", "destination_zone", "target_choice", "input_value_key", "output_value_key", "input_value_usage", "timing_mode", "multiplicity"};
                    static const std::unordered_set<std::string> bool_keys = {"optional", "is_key_card", "is_tapped"};
                    static const std::unordered_set<std::string> array_keys = {"civilizations", "races", "triggers", "effects", "commands", "actions", "options", "trigger_list", "trigger_zones", "children", "keywords", "static_abilities", "metamorph_abilities", "reaction_abilities", "cost_reductions"};

                    if (node.is_object()) {
                        for (auto it = node.begin(); it != node.end(); ++it) {
                            auto &val = it.value();
                            const std::string key = it.key();
                            if (val.is_null()) {
                                if (int_keys.count(key)) val = 0;
                                else if (string_keys.count(key)) val = std::string("");
                                else if (bool_keys.count(key)) val = false;
                                else if (array_keys.count(key)) val = nlohmann::json::array();
                                else if (key == "spell_side" || key == "trigger_descriptor" || key == "condition" || key == "filter") val = nlohmann::json::object();
                                else val = nlohmann::json::object();
                            } else if (val.is_object() || val.is_array()) sanitize_all_nulls(val);
                        }
                    } else if (node.is_array()) {
                        for (auto &el : node) {
                            if (el.is_null()) el = nlohmann::json::object();
                            else if (el.is_object() || el.is_array()) sanitize_all_nulls(el);
                        }
                    }
                };
                sanitize_all_nulls(j);

                auto defs = dm::engine::infrastructure::JsonLoader::load_cards_from_string(j.dump());
                // Only update if we parsed something
                if (!defs.empty()) {
                    if (!definitions_ptr) {
                        definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(defs);
                    } else {
                        // Merge
                        for (const auto& kv : defs) {
                            (*definitions_ptr)[kv.first] = kv.second;
                        }
                    }
                }
                return;
            } catch (const std::exception& e) {
                std::cerr << "CardRegistry::load_from_json sanitization/parse failed: " << e.what() << std::endl;
                // Fall through to attempt raw parse below
            }
            // Fallback: attempt to delegate without sanitization
            try {
                auto defs = dm::engine::infrastructure::JsonLoader::load_cards_from_string(json_str);
                if (!defs.empty()) {
                    if (!definitions_ptr) definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(defs);
                    else for (const auto& kv : defs) (*definitions_ptr)[kv.first] = kv.second;
                }
            } catch (const std::exception& e) {
                std::cerr << "CardRegistry::load_from_json fallback failed: " << e.what() << std::endl;
            }
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
