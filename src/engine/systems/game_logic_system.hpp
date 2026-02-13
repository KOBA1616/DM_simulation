#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "core/action.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <map>
#include <vector>

namespace dm::engine::systems {

    class GameLogicSystem {
    public:
        // Main Entry Points

        // Dispatches action to appropriate handler using the provided pipeline
        static void dispatch_action(PipelineExecutor& pipeline, core::GameState& state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Dispatches command to appropriate handler using the provided pipeline
        static void dispatch_command(PipelineExecutor& pipeline, core::GameState& state, const core::CommandDef& cmd, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Creates a temporary pipeline to resolve a single action (Legacy/Test support)
        static void resolve_action_oneshot(core::GameState& state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);
        // Alias for compatibility with EffectResolver::resolve_action
        static void resolve_action(core::GameState& state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db) {
            resolve_action_oneshot(state, action, card_db);
        }

        // Static helper to replace EffectResolver::resolve_play_from_stack
        static void resolve_play_from_stack(core::GameState& game_state, int stack_instance_id, int cost_reduction, core::SpawnSource spawn_source, core::PlayerID controller, const std::map<core::CardID, core::CardDefinition>& card_db, int evo_source_id = -1, core::ZoneDestination dest_override = core::ZoneDestination::NONE);

        // Main handlers for High-Level Game Actions
        static void handle_play_card(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_pay_cost(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_play(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_effect(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_attack(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_block(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_battle(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_break_shield(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_check_s_trigger(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Misc Actions
        static void handle_mana_charge(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);
        static void handle_apply_buffer_move(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_reaction(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_use_ability(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_select_target(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);
        static void handle_game_result(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);

        // Command Support
        static void handle_execute_command(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);

        // Trigger Support
        static void handle_check_creature_enter_triggers(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_check_spell_cast_triggers(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Exposed Utils
        static int get_creature_power(const core::CardInstance& creature, const core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db);
        static int get_breaker_count(const core::GameState& state, const core::CardInstance& creature, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Helper to push instructions
        static void push_trigger_check(PipelineExecutor& pipeline, core::TriggerType type, int source_id);

    private:
        static std::pair<core::Zone, core::PlayerID> get_card_location(const core::GameState& state, int instance_id);
    };

}
