#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/scenario_config.hpp"
#include <map>

namespace dm::engine {

    // Transitional PhaseSystem: contains fast_forward wrapper to new flow
    class PhaseSystem {
    public:
        static void start_game(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void setup_scenario(dm::core::GameState& game_state, const dm::core::ScenarioConfig& config, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void start_turn(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void draw_card(dm::core::GameState& game_state, dm::core::Player& player);
        static void next_phase(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void fast_forward(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static bool check_game_over(dm::core::GameState& game_state, dm::core::GameResult& result);
    };

}
