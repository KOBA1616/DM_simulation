#ifndef DM_ENGINE_GAME_INSTANCE_HPP
#define DM_ENGINE_GAME_INSTANCE_HPP

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/action.hpp"
#include "core/scenario_config.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include <map>
#include <vector>
#include <memory>
#include <string>

namespace dm::engine {

    /**
     * @brief Manages the high-level game loop, state, and pipeline execution.
     *        It owns the GameState and the ActionGenerator (implied or separate).
     */
    class GameInstance {
    public:
        // Core Components
        core::GameState state;
        const std::map<core::CardID, core::CardDefinition>& card_db;

        // Pipeline Executor
        std::shared_ptr<systems::PipelineExecutor> pipeline;
        std::shared_ptr<systems::TriggerManager> trigger_manager;

        // Ownership of DB if needed (via shared_ptr constructor)
        std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> card_db_ptr;

        // Constructor
        GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db);
        GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db);

        // Core API
        void start_game();
        void resolve_action(const core::Action& action);
        void undo();
        void initialize_card_stats(int deck_size);
        void reset_with_scenario(const core::ScenarioConfig& config);

        // A. Interactive Processing (Resume Mechanism)
        // Accepts user input (selection of targets or option) to resume a paused pipeline
        void resume_processing(const std::vector<int>& inputs);

        // Accessors
        bool is_game_over() const { return state.winner != core::GameResult::NONE; }
        int get_winner() const { return (int)state.winner; }
        bool is_waiting_for_input() const { return pipeline && pipeline->execution_paused; }

    private:
        // Phase Handling
        void advance_phase();
    };

} // namespace dm::engine

#endif // DM_ENGINE_GAME_INSTANCE_HPP
