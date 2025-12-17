#include "condition_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "effect_system.hpp"
#include <iostream>

namespace dm::engine {
    using namespace dm::core;

    class TurnEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(GameState& state, const ConditionDef& condition, int source_instance_id, const std::map<CardID, CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            PlayerID controller = EffectSystem::get_controller(state, source_instance_id);
            if (condition.type == "DURING_YOUR_TURN") {
                return state.active_player_id == controller;
            }
            if (condition.type == "DURING_OPPONENT_TURN") {
                return state.active_player_id != controller;
            }
            return true;
        }
    };

    class ManaArmedEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(GameState& state, const ConditionDef& condition, int source_instance_id, const std::map<CardID, CardDefinition>& card_db, const std::map<std::string, int>& /*execution_context*/) override {
            PlayerID controller_id = EffectSystem::get_controller(state, source_instance_id);
            Player& controller = state.players[controller_id];

            int count = 0;
            Civilization target_civ = Civilization::NONE;
            if (condition.str_val == "LIGHT") target_civ = Civilization::LIGHT;
            if (condition.str_val == "WATER") target_civ = Civilization::WATER;
            if (condition.str_val == "DARKNESS") target_civ = Civilization::DARKNESS;
            if (condition.str_val == "FIRE") target_civ = Civilization::FIRE;
            if (condition.str_val == "NATURE") target_civ = Civilization::NATURE;
            if (condition.str_val == "ZERO") target_civ = Civilization::ZERO;

            for (const auto& card : controller.mana_zone) {
                if (card_db.count(card.card_id)) {
                    if (card_db.at(card.card_id).has_civilization(target_civ)) {
                        count++;
                    }
                }
            }
            return count >= condition.value;
        }
    };

    class ShieldCountEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(GameState& state, const ConditionDef& condition, int source_instance_id, const std::map<CardID, CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            PlayerID controller_id = EffectSystem::get_controller(state, source_instance_id);
            Player& controller = state.players[controller_id];
            return (int)controller.shield_zone.size() <= condition.value;
        }
    };

    class OpponentPlayedWithoutManaEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(GameState& state, const ConditionDef& /*condition*/, int /*source_instance_id*/, const std::map<CardID, CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            return state.turn_stats.played_without_mana;
        }
    };

    class CivilizationMatchEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(GameState& state, const ConditionDef& condition, int source_instance_id, const std::map<CardID, CardDefinition>& card_db, const std::map<std::string, int>& /*execution_context*/) override {
             PlayerID controller_id = EffectSystem::get_controller(state, source_instance_id);
             Player& controller = state.players[controller_id];

             Civilization target_civ = Civilization::NONE;
             if (condition.str_val == "LIGHT") target_civ = Civilization::LIGHT;
             if (condition.str_val == "WATER") target_civ = Civilization::WATER;
             if (condition.str_val == "DARKNESS") target_civ = Civilization::DARKNESS;
             if (condition.str_val == "FIRE") target_civ = Civilization::FIRE;
             if (condition.str_val == "NATURE") target_civ = Civilization::NATURE;
             if (condition.str_val == "ZERO") target_civ = Civilization::ZERO;

             for (const auto& card : controller.battle_zone) {
                 if (card_db.count(card.card_id)) {
                     if (card_db.at(card.card_id).has_civilization(target_civ)) return true;
                 }
             }
             return false;
        }
    };

    class CompareStatEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(GameState& state, const ConditionDef& condition, int source_instance_id, const std::map<CardID, CardDefinition>& /*card_db*/, const std::map<std::string, int>& execution_context) override {
            PlayerID self_id = EffectSystem::get_controller(state, source_instance_id);
            PlayerID opp_id = (self_id == 0) ? 1 : 0;
            const Player& self = state.players[self_id];
            const Player& opp = state.players[opp_id];

            int left_value = 0;
            std::string key = condition.stat_key;

            if (execution_context.count(key)) {
                left_value = execution_context.at(key);
            }
            else if (key == "MY_MANA_COUNT") left_value = (int)self.mana_zone.size();
            else if (key == "OPPONENT_MANA_COUNT") left_value = (int)opp.mana_zone.size();
            else if (key == "MY_HAND_COUNT") left_value = (int)self.hand.size();
            else if (key == "OPPONENT_HAND_COUNT") left_value = (int)opp.hand.size();
            else if (key == "MY_SHIELD_COUNT") left_value = (int)self.shield_zone.size();
            else if (key == "OPPONENT_SHIELD_COUNT") left_value = (int)opp.shield_zone.size();
            else if (key == "MY_BATTLE_ZONE_COUNT") left_value = (int)self.battle_zone.size();
            else if (key == "OPPONENT_BATTLE_ZONE_COUNT") left_value = (int)opp.battle_zone.size();

            // Operator
            if (condition.op == ">") return left_value > condition.value;
            if (condition.op == "<") return left_value < condition.value;
            if (condition.op == "=" || condition.op == "==") return left_value == condition.value;
            if (condition.op == ">=") return left_value >= condition.value;
            if (condition.op == "<=") return left_value <= condition.value;
            if (condition.op == "!=") return left_value != condition.value;

            return false;
        }
    };

    void ConditionSystem::initialize_defaults() {
        register_evaluator("DURING_YOUR_TURN", std::make_unique<TurnEvaluator>());
        register_evaluator("DURING_OPPONENT_TURN", std::make_unique<TurnEvaluator>());
        register_evaluator("MANA_ARMED", std::make_unique<ManaArmedEvaluator>());
        register_evaluator("SHIELD_COUNT", std::make_unique<ShieldCountEvaluator>());
        register_evaluator("OPPONENT_PLAYED_WITHOUT_MANA", std::make_unique<OpponentPlayedWithoutManaEvaluator>());
        register_evaluator("CIVILIZATION_MATCH", std::make_unique<CivilizationMatchEvaluator>());
        register_evaluator("COMPARE_STAT", std::make_unique<CompareStatEvaluator>());
    }
}
