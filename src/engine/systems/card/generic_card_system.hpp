#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "effect_system.hpp"

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
        static void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, bool* interrupted, const std::vector<dm::core::ActionDef>* remaining_actions);
        // Overload for backward compatibility/bindings
        static void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id);
        // Helper overload for bindings that don't pass optional args
        static void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        static std::vector<int> select_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, const dm::core::EffectDef& continuation, std::map<std::string, int>& execution_context);

        // Helper to delegate selection handling from handlers
        static void delegate_selection(const ResolutionContext& ctx);

        // Updated check_condition to accept execution_context
        static bool check_condition(dm::core::GameState& game_state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db = {}, const std::map<std::string, int>& execution_context = {});

        static dm::core::PlayerID get_controller(const dm::core::GameState& game_state, int instance_id);

        // Check and queue Mega Last Burst if applicable
        static void check_mega_last_burst(dm::core::GameState& game_state, const dm::core::CardInstance& card, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
    };
}
