#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"

namespace dm::engine {
    class GenericCardSystem {
    public:
        // Called when a specific event happens (e.g. CIP) for a card instance
        static void resolve_trigger(dm::core::GameState& game_state, dm::core::TriggerType trigger, int source_instance_id);
        
        // Execute a specific effect definition
        static void resolve_effect(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id);
        
    private:
        static void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id);
        static std::vector<int> select_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id);
        static bool check_condition(dm::core::GameState& game_state, const dm::core::ConditionDef& condition, int source_instance_id);
    };
}
