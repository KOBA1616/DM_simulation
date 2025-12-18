#ifndef DM_ENGINE_GAME_INSTANCE_HPP
#define DM_ENGINE_GAME_INSTANCE_HPP

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/pipeline_executor.hpp"
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
        systems::PipelineExecutor pipeline;

        // Constructor
        GameInstance(const std::map<core::CardID, core::CardDefinition>& db, int initial_seed = 42);

        // Core API
        void start_game();
        void process_action(const core::Action& action);

        // A. Interactive Processing (Resume Mechanism)
        // Accepts user input (selection of targets or option) to resume a paused pipeline
        void resume_processing(const std::vector<int>& inputs);

        // Accessors
        bool is_game_over() const { return state.game_over; }
        int get_winner() const { return state.winner; }
        bool is_waiting_for_input() const { return pipeline.execution_paused; }

    private:
        // Phase Handling
        void advance_phase();
    };

} // namespace dm::engine

#endif // DM_ENGINE_GAME_INSTANCE_HPP
