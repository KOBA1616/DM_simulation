#include "trigger_manager.hpp"

namespace dm::engine::systems {

    void TriggerManager::subscribe(core::EventType type, EventCallback callback) {
        listeners[type].push_back(callback);
    }

    void TriggerManager::dispatch(const core::GameEvent& event, core::GameState& state) {
        auto it = listeners.find(event.type);
        if (it != listeners.end()) {
            // Create a copy of the listeners to handle re-entrancy (callbacks adding/removing listeners)
            // Note: This adds overhead. Optimization: Use index-based loop or delayed modification queue.
            // For now, safety first.
            std::vector<EventCallback> callbacks = it->second;
            for (const auto& callback : callbacks) {
                callback(event, state);
            }
        }
    }

    void TriggerManager::clear() {
        listeners.clear();
    }

}
