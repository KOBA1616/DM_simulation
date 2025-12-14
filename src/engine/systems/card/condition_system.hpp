#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "generic_card_system.hpp"

namespace dm::engine {

    class IConditionEvaluator {
    public:
        virtual ~IConditionEvaluator() = default;
        virtual bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context) = 0;
    };

    class ConditionSystem {
    public:
        static ConditionSystem& instance() {
            static ConditionSystem instance;
            return instance;
        }

        void register_evaluator(const std::string& type, std::unique_ptr<IConditionEvaluator> evaluator) {
            evaluators[type] = std::move(evaluator);
        }

        IConditionEvaluator* get_evaluator(const std::string& type) {
            if (evaluators.count(type)) {
                return evaluators[type].get();
            }
            return nullptr;
        }

        bool evaluate_def(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context) {
            if (condition.type == "NONE" || condition.type.empty()) return true;
            if (IConditionEvaluator* eval = get_evaluator(condition.type)) {
                return eval->evaluate(state, condition, source_instance_id, card_db, execution_context);
            }
            return false;
        }

    private:
        ConditionSystem() = default;
        ConditionSystem(const ConditionSystem&) = delete;
        ConditionSystem& operator=(const ConditionSystem&) = delete;
        std::map<std::string, std::unique_ptr<IConditionEvaluator>> evaluators;
    };

    class TurnEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            using namespace dm::core;
            PlayerID controller = GenericCardSystem::get_controller(state, source_instance_id);

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
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& /*execution_context*/) override {
            using namespace dm::core;
            PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
            Player& controller = state.players[controller_id];

            int count = 0;
            // Evaluator uses simple string civ (LIGHT etc). CardDefinition uses enum.
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
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            using namespace dm::core;
            PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
            Player& controller = state.players[controller_id];
            return (int)controller.shield_zone.size() <= condition.value;
        }
    };

    class OpponentPlayedWithoutManaEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& /*condition*/, int /*source_instance_id*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            return state.turn_stats.played_without_mana;
        }
    };

    class CivilizationMatchEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& /*execution_context*/) override {
             using namespace dm::core;
             PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
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

    class FirstAttackEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& /*condition*/, int /*source_instance_id*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
            return state.turn_stats.attacks_declared_this_turn == 1;
        }
    };

    class CompareStatEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/, const std::map<std::string, int>& execution_context) override {
            using namespace dm::core;
            PlayerID self_id = GenericCardSystem::get_controller(state, source_instance_id);
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

    class OpponentCardsDrawnEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/, const std::map<std::string, int>& /*execution_context*/) override {
             using namespace dm::core;
             PlayerID controller = GenericCardSystem::get_controller(state, source_instance_id);

             // Check if active player is the opponent.
             // TurnStats tracks the active player's draw count.
             if (state.active_player_id != controller) {
                 return state.turn_stats.cards_drawn_this_turn >= condition.value;
             }
             return false;
        }
    };
}
