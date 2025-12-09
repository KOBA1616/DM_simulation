#pragma once
#include "../../../core/game_state.hpp"
#include "../../../core/card_json_types.hpp"
#include "generic_card_system.hpp"

namespace dm::engine {

    class IConditionEvaluator {
    public:
        virtual ~IConditionEvaluator() = default;
        virtual bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) = 0;
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

        bool evaluate_def(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) {
            if (condition.type == "NONE" || condition.type.empty()) return true;
            if (IConditionEvaluator* eval = get_evaluator(condition.type)) {
                return eval->evaluate(state, condition, source_instance_id);
            }
            return false;
        }

    private:
        ConditionSystem() = default;
        std::map<std::string, std::unique_ptr<IConditionEvaluator>> evaluators;
    };

    class TurnEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
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
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
            using namespace dm::core;
            PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
            Player& controller = state.players[controller_id];

            int count = 0;
            std::string civ = condition.str_val;
            for (const auto& card : controller.mana_zone) {
                const CardData* cd = CardRegistry::get_card_data(card.card_id);
                if (cd) {
                    bool match = false;
                    for(const auto& c : cd->civilizations) if(c == civ) match = true;
                    if (match) count++;
                }
            }
            return count >= condition.value;
        }
    };

    class ShieldCountEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
            using namespace dm::core;
            PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
            Player& controller = state.players[controller_id];
            return (int)controller.shield_zone.size() <= condition.value;
        }
    };

    class OpponentPlayedWithoutManaEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
            return state.turn_stats.played_without_mana;
        }
    };

    class CivilizationMatchEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
             using namespace dm::core;
             PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
             Player& controller = state.players[controller_id];

             std::string civ = condition.str_val;
             for (const auto& card : controller.battle_zone) {
                 const CardData* cd = CardRegistry::get_card_data(card.card_id);
                 if (cd) {
                     for(const auto& c : cd->civilizations) if(c == civ) return true;
                 }
             }
             return false;
        }
    };

    class FirstAttackEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
            // First attack of turn means attacks_declared_this_turn is 1?
            // When trigger evaluates:
            // "When this creature attacks, if it's the first attack..."
            // In resolve_attack, we incremented the counter.
            // So if it's the first attack, counter == 1.
            // If condition checks BEFORE attack declaration (which is impossible for ON_ATTACK?), then 0.
            // But triggers run inside resolve_attack AFTER increment.
            // So `state.turn_stats.attacks_declared_this_turn == 1`.
            return state.turn_stats.attacks_declared_this_turn == 1;
        }
    };
}
