#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine::systems {

    class ShieldSystem {
    public:
        static ShieldSystem& instance() {
            static ShieldSystem instance;
            return instance;
        }

        void handle_break_shield(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                                 const std::map<core::CardID, core::CardDefinition>& card_db);

        void check_s_trigger(PipelineExecutor& exec, core::GameState& state, const core::Instruction& inst,
                             const std::map<core::CardID, core::CardDefinition>& card_db);

        int get_breaker_count(const core::GameState& state, const core::CardInstance& creature,
                              const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        ShieldSystem() = default;
    };

}
