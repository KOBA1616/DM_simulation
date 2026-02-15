#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/scenario_config.hpp"

namespace dm::engine::flow {

    class PhaseSystem {
    public:
        static PhaseSystem& instance() {
            static PhaseSystem instance;
            return instance;
        }

        void start_game(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Setup a specific scenario
        void setup_scenario(core::GameState& game_state, const core::ScenarioConfig& config, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Fast forward the game until a decision is needed or game over
        void fast_forward(core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Returns true if the game has ended
        bool check_game_over(core::GameState& game_state, core::GameResult& result);

        void next_phase(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_pass(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Phase specific logic
        void on_start_turn(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void on_draw_phase(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void on_mana_phase(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void on_end_turn(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        PhaseSystem() = default;
    };

}
