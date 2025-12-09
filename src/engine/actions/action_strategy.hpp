#pragma once
#include "../../../core/game_state.hpp"
#include "../../../core/action.hpp"
#include "../../../core/card_def.hpp"
#include <vector>
#include <map>

namespace dm::engine {

    struct ActionGenContext {
        const dm::core::GameState& game_state;
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db;
        dm::core::PlayerID active_player_id;
    };

    class IActionStrategy {
    public:
        virtual ~IActionStrategy() = default;
        // Generate actions.
        virtual std::vector<dm::core::Action> generate(const ActionGenContext& ctx) = 0;
    };

}
