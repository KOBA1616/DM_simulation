#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine::systems {

    class CommandSystem {
    public:
        // Main entry point: Execute a Command Definition (Macro or Primitive)
        // Returns true if command completed synchronously, false if suspended (e.g. pending input)
        static bool execute_command(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

    private:
        // Helper to expand Macros into Primitives
        static bool expand_and_execute_macro(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Helper to execute Primitives directly
        static bool execute_primitive(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Utilities
        static std::vector<int> resolve_targets(core::GameState& state, const core::CommandDef& cmd, int source_instance_id, core::PlayerID player_id, std::map<std::string, int>& execution_context);

        // Variable linking
        static int resolve_amount(const core::CommandDef& cmd, const std::map<std::string, int>& execution_context);
    };

}
