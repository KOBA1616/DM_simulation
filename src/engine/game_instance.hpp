#ifndef DM_ENGINE_GAME_INSTANCE_HPP
#define DM_ENGINE_GAME_INSTANCE_HPP

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp" // Added
#include "core/action.hpp" // Ensure Action is defined
#include "core/scenario_config.hpp" // Ensure ScenarioConfig is defined
#include <map>
#include <vector>
#include <memory>
#include <string>

namespace dm::engine {

    /**
     * @brief Manages the high-level game loop, state, and pipeline execution.
     *        It owns the GameState and the IntentGenerator (implied or separate).
     */
    class GameInstance {
    public:
        // Core Components
        core::GameState state;

        // Use shared_ptr to own or share the card database
        // UNIFIED: card_db is now a shared_ptr. The reference member is removed.
        std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> card_db;

        // Pipeline Executor (Pointer)
        std::shared_ptr<systems::PipelineExecutor> pipeline;

        // Trigger Manager
        std::shared_ptr<systems::TriggerManager> trigger_manager;

        // Constructor
        // 1. With shared pointer (Recommended)
        GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db);
        // 2. Default (Uses CardRegistry Singleton)
        GameInstance(uint32_t seed);

        // Core API
        void start_game();
        void resolve_action(const core::Action& action);

        void undo();
        void initialize_card_stats(int deck_size);
        void reset_with_scenario(const core::ScenarioConfig& config);

        // A. Interactive Processing (Resume Mechanism)
        void resume_processing(const std::vector<int>& inputs);

        // Accessors
        bool is_game_over() const { return state.game_over; }
        int get_winner() const { return static_cast<int>(state.winner); }
        bool is_waiting_for_input() const { return pipeline && pipeline->execution_paused; }

    private:
        void advance_phase();
    };

} // namespace dm::engine

#endif // DM_ENGINE_GAME_INSTANCE_HPP
