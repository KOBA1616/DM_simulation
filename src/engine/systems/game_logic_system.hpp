#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <map>
#include <vector>

namespace dm::engine::systems {

    class GameLogicSystem {
    public:
        // Main handlers for High-Level Game Actions
        static void handle_play_card(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_play(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_attack(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_block(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_battle(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_break_shield(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Misc Actions
        static void handle_mana_charge(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);
        static void handle_resolve_reaction(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_use_ability(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_select_target(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);

        // Helper to push instructions
        static void push_trigger_check(PipelineExecutor& pipeline, core::TriggerType type, int source_id);
    };

}
