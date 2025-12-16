#pragma once
#include "core/game_state.hpp"
#include "core/scenario_config.hpp"
#include "core/card_def.hpp"
#include "systems/trigger_system/trigger_manager.hpp"
#include <map>
#include <memory>

namespace dm::engine {
    class GameInstance {
    public:
        core::GameState state;
        const std::map<core::CardID, core::CardDefinition>& card_db;
        std::shared_ptr<systems::TriggerManager> trigger_manager;

        // Constructor moved to CPP to avoid circular dependency / allow lambda
        GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db);

        void reset_with_scenario(const core::ScenarioConfig& config);

        // Phase 6 Step 3: GameCommand Integration
        void undo();

        // Helper to allow python to access state easily
        core::GameState& get_state() { return state; }
    };
}
