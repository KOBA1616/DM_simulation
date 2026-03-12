#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include <vector>
#include <map>
#include <string>

namespace dm::engine::mechanics {
    class SelectionSystem {
    public:
        static SelectionSystem& instance() {
            static SelectionSystem instance;
            return instance;
        }

        std::vector<int> select_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, const dm::core::EffectDef& continuation, std::map<std::string, int>& execution_context);
        // New API: accept CommandDef directly (preferred). The old ActionDef overload
        // remains for backward compatibility and converts to CommandDef internally.
        std::vector<int> select_targets(dm::core::GameState& game_state, const dm::core::CommandDef& command, int source_instance_id, const dm::core::EffectDef& continuation, std::map<std::string, int>& execution_context);

    private:
        SelectionSystem() = default;
        SelectionSystem(const SelectionSystem&) = delete;
        SelectionSystem& operator=(const SelectionSystem&) = delete;
    };
}
