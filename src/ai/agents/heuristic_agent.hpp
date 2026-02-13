#pragma once

#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "core/card_def.hpp"
#include <vector>
#include <map>
#include <random>

namespace dm::ai {

    class HeuristicAgent {
    public:
        HeuristicAgent(int player_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        dm::core::CommandDef get_action(const dm::core::GameState& state,
                                    const std::vector<dm::core::CommandDef>& legal_actions);

    private:
        int player_id_;
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db_;
        std::mt19937 rng_;

        // Helper to get card definition
        const dm::core::CardDefinition* get_def(dm::core::CardID cid) const;
        // Helper to get card ID from instance ID in command
        dm::core::CardID get_card_id(const dm::core::GameState& state, int instance_id) const;
    };

}
