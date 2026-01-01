#pragma once
#include "core/game_event.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <functional>
#include <vector>
#include <map>
#include <memory>

namespace dm::engine::systems {

    // Callback signature: void(event, state)
    using EventCallback = std::function<void(const core::GameEvent&, core::GameState&)>;

    class TriggerManager {
    public:
        TriggerManager() = default;

        // Subscribe to a specific event type
        void subscribe(core::EventType type, EventCallback callback);

        // Dispatch an event to all subscribers
        void dispatch(const core::GameEvent& event, core::GameState& state);

        // Central trigger check logic
        void check_triggers(const core::GameEvent& event, core::GameState& state,
                            const std::map<core::CardID, core::CardDefinition>& card_db);

        // Reaction Check Logic
        bool check_reactions(const core::GameEvent& event, core::GameState& state,
                             const std::map<core::CardID, core::CardDefinition>& card_db);

        // Clear all listeners
        void clear();

        // Static setup to wire GameState to TriggerManager
        static void setup_event_handling(core::GameState& state,
                                         std::shared_ptr<TriggerManager> trigger_manager,
                                         std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> card_db);

    private:
        std::map<core::EventType, std::vector<EventCallback>> listeners;
    };

}
