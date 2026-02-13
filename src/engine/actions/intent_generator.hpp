#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "core/card_def.hpp"
#include "engine/game_command/game_command.hpp"
#include "action_strategy.hpp"
#include "strategies/pending_strategy.hpp"
#include "strategies/stack_strategy.hpp"
#include "strategies/phase_strategies.hpp"
#include <vector>
#include <map>
#include <memory>

namespace dm::engine {

    class IntentGenerator {
    public:
        // New Command-based generation
        static std::vector<dm::core::CommandDef> generate_legal_commands(
            const dm::core::GameState& game_state,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );
    };

}
