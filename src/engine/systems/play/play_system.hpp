#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine::systems {

    class PlaySystem {
    public:
        static PlaySystem& instance() {
            static PlaySystem instance;
            return instance;
        }

        void handle_play_card(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                              const std::map<core::CardID, core::CardDefinition>& card_db);

        void handle_mana_charge(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst);

        void handle_resolve_play(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                                 const std::map<core::CardID, core::CardDefinition>& card_db);

        void handle_use_ability(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                                const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        PlaySystem() = default;
    };

}
