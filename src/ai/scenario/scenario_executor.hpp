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

        // Preferred: Shared Pointer constructor
        explicit ScenarioExecutor(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> db);

        // Deprecated: Reference constructor (forces copy)
        ScenarioExecutor(const std::map<dm::core::CardID, dm::core::CardDefinition>& db);

        // Default using Registry
        ScenarioExecutor();

        GameResultInfo run_scenario(const dm::core::ScenarioConfig& config, int max_steps = 1000);
    };

}
