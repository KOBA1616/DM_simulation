#include "card_registry.hpp"
#include "core/card_def.hpp"
#include "core/keywords.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace dm::engine {
    
    using namespace dm::core;

    std::map<int, dm::core::CardData> CardRegistry::cards;
    std::shared_ptr<std::map<dm::core::CardID, dm::core::CardDefinition>> CardRegistry::definitions_ptr =
        std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>();

    // Helper for legacy conversion (duplicated from JsonLoader due to architectural split)
    static CommandDef convert_legacy_action_internal(const ActionDef& act) {
        CommandDef cmd;
        cmd.optional = act.optional;
        cmd.amount = act.value1;
        cmd.str_param = act.str_val;
        cmd.target_filter = act.filter;
        cmd.target_group = act.scope;

        if (act.source_zone.empty() == false) cmd.from_zone = act.source_zone;
        if (act.destination_zone.empty() == false) cmd.to_zone = act.destination_zone;

        switch (act.type) {
            case EffectActionType::DRAW_CARD: cmd.type = CommandType::DRAW_CARD; break;
            case EffectActionType::ADD_MANA: cmd.type = CommandType::MANA_CHARGE; break;
            case EffectActionType::DESTROY: cmd.type = CommandType::DESTROY; break;
            case EffectActionType::RETURN_TO_HAND: cmd.type = CommandType::RETURN_TO_HAND; break;
            case EffectActionType::TAP: cmd.type = CommandType::TAP; break;
            case EffectActionType::UNTAP: cmd.type = CommandType::UNTAP; break;
            case EffectActionType::MODIFY_POWER: cmd.type = CommandType::POWER_MOD; break;
            case EffectActionType::BREAK_SHIELD: cmd.type = CommandType::BREAK_SHIELD; break;
            case EffectActionType::DISCARD: cmd.type = CommandType::DISCARD; break;
            case EffectActionType::SEARCH_DECK: cmd.type = CommandType::SEARCH_DECK; break;
            case EffectActionType::GRANT_KEYWORD: cmd.type = CommandType::ADD_KEYWORD; break;
            case EffectActionType::SEND_TO_MANA: cmd.type = CommandType::MANA_CHARGE; break;
            case EffectActionType::MOVE_CARD: cmd.type = CommandType::TRANSITION; break;
            default: cmd.type = CommandType::NONE; break;
        }
        return cmd;
    }

    void CardRegistry::load_from_json(const std::string& json_str) {
        try {
            auto j = nlohmann::json::parse(json_str);
            auto new_defs = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>();

            if (j.is_array()) {
                for (const auto& item : j) {
                    dm::core::CardData card = item.get<dm::core::CardData>();
                    // Backward compatibility: allow single "civilization" field
                    if (card.civilizations.empty() && item.contains("civilization")) {
                        std::string civ_str = item.at("civilization").get<std::string>();
                        if (civ_str == "LIGHT") card.civilizations.push_back(Civilization::LIGHT);
                        else if (civ_str == "WATER") card.civilizations.push_back(Civilization::WATER);
                        else if (civ_str == "DARKNESS") card.civilizations.push_back(Civilization::DARKNESS);
                        else if (civ_str == "FIRE") card.civilizations.push_back(Civilization::FIRE);
                        else if (civ_str == "NATURE") card.civilizations.push_back(Civilization::NATURE);
                        else if (civ_str == "ZERO") card.civilizations.push_back(Civilization::ZERO);
                    }
                    cards[card.id] = card;
                    (*new_defs)[static_cast<CardID>(card.id)] = convert_to_def(card);
                }
            } else {
                // Single object
                dm::core::CardData card = j.get<dm::core::CardData>();
                if (card.civilizations.empty() && j.contains("civilization")) {
                    std::string civ_str = j.at("civilization").get<std::string>();
                    if (civ_str == "LIGHT") card.civilizations.push_back(Civilization::LIGHT);
                    else if (civ_str == "WATER") card.civilizations.push_back(Civilization::WATER);
                    else if (civ_str == "DARKNESS") card.civilizations.push_back(Civilization::DARKNESS);
                    else if (civ_str == "FIRE") card.civilizations.push_back(Civilization::FIRE);
                    else if (civ_str == "NATURE") card.civilizations.push_back(Civilization::NATURE);
                    else if (civ_str == "ZERO") card.civilizations.push_back(Civilization::ZERO);
                }
                cards[card.id] = card;
                (*new_defs)[static_cast<CardID>(card.id)] = convert_to_def(card);
            }
            // Update the shared pointer
            definitions_ptr = new_defs;

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

    const std::map<dm::core::CardID, dm::core::CardDefinition>& CardRegistry::get_all_definitions() {
        if (!definitions_ptr) {
            definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>();
        }
        return *definitions_ptr;
    }

    std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> CardRegistry::get_all_definitions_ptr() {
        if (!definitions_ptr) {
            definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>();
        }
        return definitions_ptr;
    }

    void CardRegistry::clear() {
        cards.clear();
        definitions_ptr = std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>();
    }

    dm::core::CardDefinition CardRegistry::convert_to_def(const dm::core::CardData& data) {
        CardDefinition def;
        def.id = static_cast<CardID>(data.id);
        def.name = data.name;

        // Civilization mapping
        def.civilizations = data.civilizations;

        // Type mapping
        if (data.type == "CREATURE") def.type = CardType::CREATURE;
        else if (data.type == "SPELL") def.type = CardType::SPELL;
        else if (data.type == "EVOLUTION_CREATURE") def.type = CardType::EVOLUTION_CREATURE;
        else if (data.type == "TAMASEED") def.type = CardType::TAMASEED;
        else if (data.type == "CROSS_GEAR") def.type = CardType::CROSS_GEAR;
        else if (data.type == "CASTLE") def.type = CardType::CASTLE;
        else if (data.type == "PSYCHIC_CREATURE") def.type = CardType::PSYCHIC_CREATURE;
        else if (data.type == "GR_CREATURE") def.type = CardType::GR_CREATURE;

        def.cost = data.cost;
        def.power = data.power;
        def.races = data.races;

        // Effects with Hybrid Conversion
        def.effects.clear();
        def.effects.reserve(data.effects.size());
        for (const auto& eff : data.effects) {
            EffectDef engine_eff = eff;
            // Legacy Conversion
            if (engine_eff.commands.empty() && !engine_eff.actions.empty()) {
                engine_eff.commands.reserve(engine_eff.actions.size());
                for (const auto& act : engine_eff.actions) {
                    CommandDef cmd = convert_legacy_action_internal(act);
                    if (cmd.type != CommandType::NONE) {
                        engine_eff.commands.push_back(cmd);
                    }
                }
            }
            def.effects.push_back(engine_eff);
        }

        // Metamorph Abilities
        def.metamorph_abilities.clear();
        def.metamorph_abilities.reserve(data.metamorph_abilities.size());
        for (const auto& eff : data.metamorph_abilities) {
            EffectDef engine_eff = eff;
            if (engine_eff.commands.empty() && !engine_eff.actions.empty()) {
                engine_eff.commands.reserve(engine_eff.actions.size());
                for (const auto& act : engine_eff.actions) {
                    CommandDef cmd = convert_legacy_action_internal(act);
                    if (cmd.type != CommandType::NONE) {
                        engine_eff.commands.push_back(cmd);
                    }
                }
            }
            def.metamorph_abilities.push_back(engine_eff);
        }

        // Revolution Change
        if (data.revolution_change_condition.has_value()) {
            def.revolution_change_condition = data.revolution_change_condition;
            def.keywords.add(dm::core::Keyword::REVOLUTION_CHANGE);
        }

        // Reaction Abilities
        def.reaction_abilities = data.reaction_abilities;
        def.cost_reductions = data.cost_reductions;

        // Twinpact Spell Side (Recursive)
        if (data.spell_side) {
            def.spell_side = std::make_shared<CardDefinition>(convert_to_def(*data.spell_side));
        }

        // Keywords from explicit 'keywords' block
        if (data.keywords.has_value()) {
            const auto& kws = *data.keywords;
            for (const auto& [key, val] : kws) {
                if (val) {
                    auto kw = string_to_keyword(key);
                    if (kw) {
                        def.keywords.add(*kw);
                    }
                }
            }
        }

        // Keywords mapping from effects
        for (const auto& eff : data.effects) {
            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.add(dm::core::Keyword::SHIELD_TRIGGER);
            if (eff.trigger == TriggerType::ON_PLAY) def.keywords.add(dm::core::Keyword::CIP);
            if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.add(dm::core::Keyword::AT_ATTACK);
            if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.add(dm::core::Keyword::DESTRUCTION);

            if (eff.trigger == TriggerType::PASSIVE_CONST) {
                for (const auto& action : eff.actions) {
                    std::string k_str = action.str_val;
                    std::transform(k_str.begin(), k_str.end(), k_str.begin(), ::tolower);
                    auto kw = string_to_keyword(k_str);
                    if (kw) {
                        def.keywords.add(*kw);
                        if (*kw == Keyword::POWER_ATTACKER) {
                            def.power_attacker_bonus = action.value1;
                        }
                    }
                }
            }
        }

        // AI Metadata
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        return def;
    }

}
