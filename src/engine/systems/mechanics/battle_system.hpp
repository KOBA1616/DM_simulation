#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"

namespace dm::engine::systems {

    class BattleSystem {
    public:
        static BattleSystem& instance() {
            static BattleSystem instance;
            return instance;
        }

        // Main handlers called by LogicSystem/Pipeline
        void handle_attack(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                           const std::map<core::CardID, core::CardDefinition>& card_db);

        void handle_block(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                          const std::map<core::CardID, core::CardDefinition>& card_db);

        void handle_resolve_battle(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                                   const std::map<core::CardID, core::CardDefinition>& card_db);

        // Utilities
        int get_creature_power(const core::CardInstance& creature, const core::GameState& state,
                               const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        BattleSystem() = default;
    };

}
