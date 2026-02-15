#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "core/instruction.hpp"
#include <vector>
#include <map>

namespace dm::engine::systems {

    class CommandSystem {
    public:
        // Main entry point: Generate Instructions from a Command Definition
        static std::vector<core::Instruction> generate_instructions(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Legacy/Compatibility execution (Creates a temporary pipeline or executes directly)
        static void execute_command(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

    private:
        // Helper to expand Macros
        static void generate_macro_instructions(std::vector<core::Instruction>& out, core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Helper to generate Primitives
        static void generate_primitive_instructions(std::vector<core::Instruction>& out, core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Utilities
        static std::vector<int> resolve_targets(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Variable linking
        static int resolve_amount(const core::CommandDef& cmd, const std::map<std::string, int>& execution_context);
    };

}
