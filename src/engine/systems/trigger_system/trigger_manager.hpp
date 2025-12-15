#pragma once
#include "core/game_event.hpp"
#include "core/game_state.hpp"
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
        // Returns a subscription ID (optional, for unsubscription later - skipped for now)
        void subscribe(core::EventType type, EventCallback callback);

        // Dispatch an event to all subscribers
        void dispatch(const core::GameEvent& event, core::GameState& state);

        // Clear all listeners (useful for reset)
        void clear();

    private:
        std::map<core::EventType, std::vector<EventCallback>> listeners;
    };

}
