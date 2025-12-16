#include "card_registry.hpp"
#include "core/card_def.hpp"
#include <nlohmann/json.hpp>
#include <iostream>

namespace dm::engine {
    
    using namespace dm::core;

    std::map<int, dm::core::CardData> CardRegistry::cards;
    std::map<dm::core::CardID, dm::core::CardDefinition> CardRegistry::definitions;

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
                    definitions[static_cast<CardID>(card.id)] = convert_to_def(card);
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
                definitions[static_cast<CardID>(card.id)] = convert_to_def(card);
            }
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
        return definitions;
    }

    void CardRegistry::clear() {
        cards.clear();
        definitions.clear();
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
            def.keywords.revolution_change = true;
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
            if (kws.count("shield_trigger") && kws.at("shield_trigger")) def.keywords.shield_trigger = true;
            if (kws.count("blocker") && kws.at("blocker")) def.keywords.blocker = true;
            if (kws.count("speed_attacker") && kws.at("speed_attacker")) def.keywords.speed_attacker = true;
            if (kws.count("slayer") && kws.at("slayer")) def.keywords.slayer = true;
            if (kws.count("double_breaker") && kws.at("double_breaker")) def.keywords.double_breaker = true;
            if (kws.count("triple_breaker") && kws.at("triple_breaker")) def.keywords.triple_breaker = true;
            if (kws.count("mach_fighter") && kws.at("mach_fighter")) def.keywords.mach_fighter = true;
            if (kws.count("evolution") && kws.at("evolution")) def.keywords.evolution = true;
            if (kws.count("g_strike") && kws.at("g_strike")) def.keywords.g_strike = true;
            if (kws.count("just_diver") && kws.at("just_diver")) def.keywords.just_diver = true;
            if (kws.count("shield_burn") && kws.at("shield_burn")) def.keywords.shield_burn = true;
            if (kws.count("untap_in") && kws.at("untap_in")) def.keywords.untap_in = true;
            if (kws.count("unblockable") && kws.at("unblockable")) def.keywords.unblockable = true;
            if (kws.count("meta_counter_play") && kws.at("meta_counter_play")) def.keywords.meta_counter_play = true;
            if (kws.count("hyper_energy") && kws.at("hyper_energy")) def.keywords.hyper_energy = true;
            if (kws.count("friend_burst") && kws.at("friend_burst")) def.keywords.friend_burst = true;
        }

        // Keywords mapping from effects
        for (const auto& eff : data.effects) {
            if (eff.trigger == TriggerType::S_TRIGGER) def.keywords.shield_trigger = true;
            if (eff.trigger == TriggerType::ON_PLAY) def.keywords.cip = true;
            if (eff.trigger == TriggerType::ON_ATTACK) def.keywords.at_attack = true;
            if (eff.trigger == TriggerType::ON_DESTROY) def.keywords.destruction = true;

            if (eff.trigger == TriggerType::PASSIVE_CONST) {
                for (const auto& action : eff.actions) {
                    if (action.str_val == "BLOCKER") def.keywords.blocker = true;
                    if (action.str_val == "SPEED_ATTACKER") def.keywords.speed_attacker = true;
                    if (action.str_val == "SLAYER") def.keywords.slayer = true;
                    if (action.str_val == "DOUBLE_BREAKER") def.keywords.double_breaker = true;
                    if (action.str_val == "TRIPLE_BREAKER") def.keywords.triple_breaker = true;
                    if (action.str_val == "POWER_ATTACKER") {
                        def.keywords.power_attacker = true;
                        def.power_attacker_bonus = action.value1;
                    }
                    if (action.str_val == "EVOLUTION") def.keywords.evolution = true;
                    if (action.str_val == "MACH_FIGHTER") def.keywords.mach_fighter = true;
                    if (action.str_val == "G_STRIKE") def.keywords.g_strike = true;
                    if (action.str_val == "JUST_DIVER") def.keywords.just_diver = true;
                    if (action.str_val == "HYPER_ENERGY") def.keywords.hyper_energy = true;
                    if (action.str_val == "META_COUNTER") def.keywords.meta_counter_play = true;
                    if (action.str_val == "SHIELD_BURN") def.keywords.shield_burn = true;
                    if (action.str_val == "UNBLOCKABLE") def.keywords.unblockable = true;
                }
            }
        }

        // AI Metadata
        def.is_key_card = data.is_key_card;
        def.ai_importance_score = data.ai_importance_score;

        return def;
    }
}
