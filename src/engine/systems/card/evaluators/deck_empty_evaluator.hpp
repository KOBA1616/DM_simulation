#pragma once
#include "engine/systems/rules/condition_system.hpp"
#include "engine/systems/card/effect_system.hpp"

namespace dm::engine {

    class DeckEmptyEvaluator : public dm::engine::rules::IConditionEvaluator {
    public:
        bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>&, const std::map<std::string, int>&) override {
            dm::core::PlayerID controller = EffectSystem::get_controller(state, source_instance_id);
            bool is_empty = state.players[controller].deck.empty();
            return is_empty == (condition.value != 0);
        }
    };
}
