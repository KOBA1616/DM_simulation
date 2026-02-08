#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include <map>

namespace dm::engine {

    class CommandGenerator {
    public:
        // Produce a vector of core::CommandDef from the given game state and card DB.
        // This v1 implementation bridges by calling the legacy IntentGenerator (actions)
        // and mapping Action -> CommandDef. Later versions should implement direct
        // command-first generation.
        static std::vector<dm::core::CommandDef> generate_legal_commands(
            const dm::core::GameState& game_state,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );
    };

}
