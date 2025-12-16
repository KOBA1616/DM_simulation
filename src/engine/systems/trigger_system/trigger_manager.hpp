#pragma once
#include "core/game_event.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <functional>
#include <vector>
#include <map>
#include <memory>
#include <optional>

namespace dm::engine::systems {

    // Callback signature: void(event, state)
    using EventCallback = std::function<void(const core::GameEvent&, core::GameState&)>;

    class TriggerManager {
    public:
        TriggerManager() = default;

        // Subscribe to a specific event type
        void subscribe(core::EventType type, EventCallback callback);

        // Dispatch an event to all subscribers and check for card triggers
        void dispatch(const core::GameEvent& event, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Check for card-based triggers (e.g., ON_PLAY, ON_ATTACK) and add them to pending_effects
        void check_triggers(const core::GameEvent& event, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Clear all listeners
        void clear();

    private:
        std::map<core::EventType, std::vector<EventCallback>> listeners;
    };

}
