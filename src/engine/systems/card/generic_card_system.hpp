#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"

namespace dm::engine {
    class GenericCardSystem {
    public:
        // Called when a specific event happens (e.g. CIP) for a card instance
        static void resolve_trigger(dm::core::GameState& game_state, dm::core::TriggerType trigger, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        
        // Execute a specific effect definition
        static void resolve_effect(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db = {});
        static void resolve_effect_with_context(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db = {});
        // Resolve an effect using explicit targets (used for TARGET_SELECT pending effects)
        static void resolve_effect_with_targets(dm::core::GameState& game_state, const dm::core::EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context);
        // Overload for backward compatibility
        static void resolve_effect_with_targets(dm::core::GameState& game_state, const dm::core::EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db = {});

        // Exposed for bindings/testing
        static void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db = {});
        // Overload for backward compatibility/bindings
        static void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id);

        static std::vector<int> select_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, const dm::core::EffectDef& continuation, std::map<std::string, int>& execution_context);

        // Updated check_condition to accept execution_context
        static bool check_condition(dm::core::GameState& game_state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db = {}, const std::map<std::string, int>& execution_context = {});

        static dm::core::PlayerID get_controller(const dm::core::GameState& game_state, int instance_id);
    };
}
