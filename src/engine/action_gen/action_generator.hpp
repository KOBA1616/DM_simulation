#pragma once
#include "../../core/game_state.hpp"
#include "../../core/action.hpp"
#include "../../core/card_def.hpp"
#include <vector>
#include <map>

namespace dm::engine {

    class ActionGenerator {
    public:
        static std::vector<dm::core::Action> generate_legal_actions(
            const dm::core::GameState& game_state, 
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );
    };

}
