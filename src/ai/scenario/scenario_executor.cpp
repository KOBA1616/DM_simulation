#include "scenario_executor.hpp"
#include "../../engine/game_instance.hpp"
#include "../../engine/flow/phase_manager.hpp"
#include "../../engine/action_gen/action_generator.hpp"
#include "../../engine/effects/effect_resolver.hpp"
#include "../agents/heuristic_agent.hpp"
#include <random>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    ScenarioExecutor::ScenarioExecutor(const std::map<CardID, CardDefinition>& db)
        : card_db(db) {}

    GameResultInfo ScenarioExecutor::run_scenario(const ScenarioConfig& config, int max_steps) {
        // Use a random seed for the game instance
        std::random_device rd;
        uint32_t seed = rd();

        // Create GameInstance with reference to OUR card_db
        GameInstance instance(seed, card_db);
        instance.reset_with_scenario(config);

        HeuristicAgent agent0(0, card_db);
        HeuristicAgent agent1(1, card_db);

        int steps = 0;
        GameResult result = GameResult::NONE;

        while (steps < max_steps) {
            bool game_over = PhaseManager::check_game_over(instance.state, result);
            if (game_over) {
                break;
            }

            std::vector<Action> legal_actions = ActionGenerator::generate_legal_actions(instance.state, card_db);
            if (legal_actions.empty()) {
                PhaseManager::next_phase(instance.state, card_db);
                continue;
            }

            // Select action
            Action action;
            if (instance.state.active_player_id == 0) {
                action = agent0.get_action(instance.state, legal_actions);
            } else {
                action = agent1.get_action(instance.state, legal_actions);
            }

            EffectResolver::resolve_action(instance.state, action, card_db);
            steps++;
        }

        GameResultInfo info;
        info.result = result;
        info.turn_count = instance.state.turn_number;
        return info;
    }

}
