#pragma once

#include "../../core/game_state.hpp"
#include "../../core/scenario_config.hpp"
#include "../../core/card_def.hpp"
#include "../self_play/self_play.hpp" // For GameResultInfo
#include <map>
#include <vector>

namespace dm::ai {

    class ScenarioExecutor {
    public:
        // Owns the card database to ensure lifetime safety for GameInstances
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db;

        ScenarioExecutor(const std::map<dm::core::CardID, dm::core::CardDefinition>& db);

        GameResultInfo run_scenario(const dm::core::ScenarioConfig& config, int max_steps = 1000);
    };

}
