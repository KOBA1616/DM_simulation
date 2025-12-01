#pragma once
#include "../core/game_state.hpp"
#include "../core/scenario_config.hpp"
#include "../core/card_def.hpp"
#include <map>

namespace dm::engine {
    class GameInstance {
    public:
        core::GameState state;
        const std::map<core::CardID, core::CardDefinition>& card_db;

        GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db)
            : state(seed), card_db(db) {}

        void reset_with_scenario(const core::ScenarioConfig& config);

        // Helper to allow python to access state easily
        core::GameState& get_state() { return state; }
    };
}
