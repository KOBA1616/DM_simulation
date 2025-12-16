#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine::systems {

    class CommandSystem {
    public:
        // Main entry point: Execute a Command Definition (Macro or Primitive)
        static void execute_command(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id);

    private:
        // Helper to expand Macros into Primitives
        static void expand_and_execute_macro(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id);

        // Helper to execute Primitives directly
        static void execute_primitive(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id);

        // Utilities
        static std::vector<int> resolve_targets(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id);
    };

}
