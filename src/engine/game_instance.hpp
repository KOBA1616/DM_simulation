#pragma once
#include "core/game_state.hpp"
#include "core/scenario_config.hpp"
#include "core/card_def.hpp"
#include "core/action.hpp"
#include "systems/trigger_system/trigger_manager.hpp"
#include "systems/pipeline_executor.hpp"
#include <map>
#include <memory>
#include <optional>

namespace dm::engine {
    class GameInstance {
    public:
        core::GameState state;

        // Storage for Python interop or ownership
        std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> card_db_ptr;

        // Reference accessor (backward compatibility and convenience)
        const std::map<core::CardID, core::CardDefinition>& card_db;

        std::shared_ptr<systems::TriggerManager> trigger_manager;
        std::shared_ptr<systems::PipelineExecutor> pipeline;

        // Constructor 1: Reference (Caller owns map)
        GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db);

        // Constructor 2: Shared Ptr (Shared ownership)
        GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db);

        void reset_with_scenario(const core::ScenarioConfig& config);

        // Phase 7: Direct Action Resolution
        void resolve_action(const core::Action& action);

        // Phase 6 Step 3: GameCommand Integration
        void undo();

        // Phase 7: Stats Init
        void initialize_card_stats(int deck_size = 40);

        // Helper to allow python to access state easily
        core::GameState& get_state() { return state; }
    };
}
