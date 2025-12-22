#pragma once

#include "core/game_state.hpp"
#include "core/scenario_config.hpp"
#include "core/card_def.hpp"
#include "ai/self_play/self_play.hpp" // For GameResultInfo
#include <map>
#include <vector>
#include <memory>

namespace dm::ai {

    class ScenarioExecutor {
    public:
        // Use shared_ptr to share ownership
        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db;

        ScenarioExecutor(const std::map<dm::core::CardID, dm::core::CardDefinition>& db);
        ScenarioExecutor(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> db);
        ScenarioExecutor(); // Default using Registry

        GameResultInfo run_scenario(const dm::core::ScenarioConfig& config, int max_steps = 1000);
    };

}
