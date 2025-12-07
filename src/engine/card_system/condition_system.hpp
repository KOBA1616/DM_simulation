#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
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
            // Fallback: If unknown condition, default to false or true?
            // Usually safest to false if logic missing.
            return false;
        }

    private:
        ConditionSystem() = default;
        std::map<std::string, std::unique_ptr<IConditionEvaluator>> evaluators;
    };

    // ... (Evaluator implementations will be in .cpp or defined here if header-only was intended, but avoiding redefinition errors if included multiple times)
    // Actually, generic_card_system.cpp had them inline? No, they were in generic_card_system.cpp in previous context but here I see them in header?
    // The file `condition_system.hpp` I read earlier had the implementations INLINE.
    // If I keep them inline in header, I must mark them `inline` or put them in `.cpp`.
    // The previous file content I read was `src/engine/card_system/condition_system.hpp`.
    // It seems they were defined in class body, which is implicitly inline.
    // I will keep them there but ensure I didn't break anything.
    // Actually, `GenericCardSystem.cpp` had `ensure_evaluators_registered` which created `std::make_unique<TurnEvaluator>()`.
    // This implies `TurnEvaluator` must be visible to `GenericCardSystem.cpp`.
    // So keeping them in header is correct.

    // I will include the implementations I read back into this file to preserve them.
    // And add `evaluate_def` helper.

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
                // civ logic
                // CardData now has `civilizations` vector.
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
}
