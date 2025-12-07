#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "generic_card_system.hpp" // For get_controller

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

    private:
        ConditionSystem() = default;
        std::map<std::string, std::unique_ptr<IConditionEvaluator>> evaluators;
    };

    // Concrete Evaluators

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
            std::string civ = condition.str_val; // e.g., "FIRE"
            for (const auto& card : controller.mana_zone) {
                const CardData* cd = CardRegistry::get_card_data(card.card_id);
                if (cd && cd->civilization == civ) {
                    count++;
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

            if (condition.str_val == "MAX") {
                 return (int)controller.shield_zone.size() <= condition.value;
            }
            // Default "MIN" or exact? Usually "has X or fewer" (recheck logic elsewhere, but let's assume value is threshold)
            // Existing logic was mostly implicit or checked elsewhere.
            // Let's implement standard ">= value" or "<= value" depending on usage.
            // Usually Shield Trigger conditions are "If you have 0 shields".
            // But ConditionDef is generic.
            // If condition.value is 0, maybe it means == 0?
            // "SHIELD_COUNT" usually implies checking own shields.

            // NOTE: The current codebase didn't implement logic for SHIELD_COUNT in GenericCardSystem::check_condition!
            // It only had "DURING_YOUR/OPPONENT_TURN".
            // So this is new/extension logic.
            // I'll implement "Self Shield Count <= value" for now as that's common (Revolution).

            return (int)controller.shield_zone.size() <= condition.value;
        }
    };

    class OpponentPlayedWithoutManaEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
            // Checks if opponent played a card without paying mana this turn.
            // This flag is in state.turn_stats.played_without_mana?
            // Or tracked per player?
            // state.turn_stats has `played_without_mana` boolean.
            // But does it distinguish WHO played it?
            // "played_without_mana" seems to be a global flag for the *turn player*?
            // Or maybe it tracks if *any* card was played without mana?
            // Memory says: "The played_without_mana flag in TurnStats is set by ManaSystem::auto_tap_mana when a card is paid for with 0 mana."
            // If it's in TurnStats, it's global for the turn.
            // If condition is OPPONENT_PLAYED_WITHOUT_MANA, it implies we check if the opponent (who is likely the active player during their turn) did it.

            // If it's my turn, opponent can't play cards (except via triggers/ninjas).
            // Usually this condition (Meta Counter) triggers on Opponent's turn.
            // So if it's Opponent's turn (Active Player = Opponent), checking TurnStats is correct.

            return state.turn_stats.played_without_mana;
        }
    };

    class CivilizationMatchEvaluator : public IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id) override {
             using namespace dm::core;
             PlayerID controller_id = GenericCardSystem::get_controller(state, source_instance_id);
             Player& controller = state.players[controller_id];

             // Check if I have a card of civilization X in Battle Zone or Mana?
             // Usually "If you have a Fire creature".
             std::string civ = condition.str_val;
             // Check Battle Zone
             for (const auto& card : controller.battle_zone) {
                 const CardData* cd = CardRegistry::get_card_data(card.card_id);
                 if (cd && cd->civilization == civ) return true;
             }
             // Check Mana Zone? Usually not, that's Mana Armed.
             return false;
        }
    };

}
