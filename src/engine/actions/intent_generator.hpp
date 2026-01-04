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
        // Legacy support
        static std::vector<dm::core::Action> generate_legal_actions(
            const dm::core::GameState& game_state, 
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );

        // New GameCommand-based generation
        static std::vector<std::shared_ptr<game_command::GameCommand>> generate_legal_commands(
            const dm::core::GameState& game_state,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );
    };

}
