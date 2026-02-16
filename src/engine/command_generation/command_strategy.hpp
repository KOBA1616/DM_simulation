#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/card_json_types.hpp"
#include <vector>
#include <map>

namespace dm::engine {

    struct CommandGenContext {
        const dm::core::GameState& game_state;
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db;
        dm::core::PlayerID active_player_id;
    };

    class ICommandStrategy {
    public:
        virtual ~ICommandStrategy() = default;
        // Generate actions.
        virtual std::vector<dm::core::CommandDef> generate(const CommandGenContext& ctx) = 0;
    };

}
