#include "scenario_executor.hpp"
#include "engine/game_instance.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/actions/intent_generator.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "ai/agents/heuristic_agent.hpp"
#include <random>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;
    using namespace dm::engine::systems;

    ScenarioExecutor::ScenarioExecutor(std::shared_ptr<const std::map<CardID, CardDefinition>> db)
        : card_db(db) {}

    ScenarioExecutor::ScenarioExecutor(const std::map<CardID, CardDefinition>& db)
        : card_db(std::make_shared<std::map<CardID, CardDefinition>>(db)) {}

    ScenarioExecutor::ScenarioExecutor()
        : card_db(CardRegistry::get_all_definitions_ptr()) {}

    GameResultInfo ScenarioExecutor::run_scenario(const ScenarioConfig& config, int max_steps) {
        // Use a random seed for the game instance
        std::random_device rd;
        uint32_t seed = rd();

        // Create GameInstance with shared pointer to card_db
        GameInstance instance(seed, card_db);
        instance.reset_with_scenario(config);

        HeuristicAgent agent0(0, *card_db);
        HeuristicAgent agent1(1, *card_db);

        int steps = 0;
        GameResult result = GameResult::NONE;

        while (steps < max_steps) {
            bool game_over = PhaseManager::check_game_over(instance.state, result);
            if (game_over) {
                break;
            }

            std::vector<CommandDef> legal_actions = IntentGenerator::generate_legal_commands(instance.state, *card_db);
            if (legal_actions.empty()) {
                PhaseManager::next_phase(instance.state, *card_db);
                continue;
            }

            // Select action
            CommandDef action;
            if (instance.state.active_player_id == 0) {
                action = agent0.get_action(instance.state, legal_actions);
            } else {
                action = agent1.get_action(instance.state, legal_actions);
            }

            GameLogicSystem::resolve_command_oneshot(instance.state, action, *card_db);
            steps++;
        }

        GameResultInfo info;
        info.result = result;
        info.turn_count = instance.state.turn_number;
        return info;
    }

}
