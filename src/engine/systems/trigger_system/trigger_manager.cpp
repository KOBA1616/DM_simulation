#include "trigger_manager.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "core/card_def.hpp"

namespace dm::engine::systems {

    using namespace core;

    void TriggerManager::subscribe(EventType type, EventCallback callback) {
        listeners[type].push_back(callback);
    }

    void TriggerManager::dispatch(const GameEvent& event, GameState& state) {
        auto it = listeners.find(event.type);
        if (it != listeners.end()) {
            std::vector<EventCallback> callbacks = it->second;
            for (const auto& callback : callbacks) {
                callback(event, state);
            }
        }
    }

    void TriggerManager::check_triggers(const GameEvent& event, GameState& state,
                                        const std::map<CardID, CardDefinition>& card_db) {
        // 1. Passive Effects Check
        // In the new architecture, passive effects are constantly recalculated or checked here.
        // For events, we might check "Interceptors" that modify the event itself,
        // but Requirements 5.1 says "1. Passive Effects check".
        // This generally means checking if any continuous effect changes the context or prevents the event.
        // For now, we assume PassiveEffectSystem handles the static state, and this step is for
        // "Event Modification" passives (like "Instead of destruction, put to shield").
        // These are effectively Interceptors.

        // 2. Triggered Abilities Search
        // Iterate over all cards in relevant zones (Battle, Hand for Ninja, etc.)
        // Optimization: GameState should have a 'trigger_map' cache.
        // For Phase 6 Step 1, we implement a naive iteration or rely on what's available.
        // We will scan Battle Zone and Hand (for simple triggers).

        // TODO: This naive scan is slow (O(N)). Future optimization: specialized observer lists in GameState.

        // Define zones to check based on event type
        std::vector<Zone> zones_to_check;
        zones_to_check.push_back(Zone::BATTLE);
        // Some triggers work from hand (Ninja Strike, S-Back) or Graveyard (Piggyback)
        // For MVP, just Battle Zone.

        for (PlayerID pid : {state.active_player_id, static_cast<PlayerID>(1 - state.active_player_id)}) {
            // Updated to use the new helper
            const auto& battle_zone = state.get_zone(pid, Zone::BATTLE);
            for (int instance_id : battle_zone) {
                if (instance_id < 0) continue;
                // Correct pointer access
                const auto* card_ptr = state.get_card_instance(instance_id);
                if (!card_ptr) continue;
                const auto& card = *card_ptr;

                if (!card_db.count(card.card_id)) continue;
                const auto& def = card_db.at(card.card_id);

                // Check card definition for triggers matching 'event'
                // Currently, CardDefinition stores `effects` list.
                // We need to match TriggerType (e.g. ON_PLAY) with EventType (ZONE_ENTER).

                // Mapping logic (Hardcoded for Step 1/2)
                // ZONE_ENTER + BATTLE -> ON_PLAY
                // ATTACK_INIT -> ON_ATTACK
                // CARD_DESTROYED -> ON_DESTROY

                // This logic mirrors `GenericCardSystem::resolve_trigger` but event-driven.

                // TODO: Implement the mapping and PendingEffect generation.
                // Since `PendingEffect` requires `EffectDef`, we look it up from `def.effects`.
            }
        }

        // 3. Interceptor Application
        // Check for replacement effects.
        // If found, modify the event or cancel it.
        // Implementation pending "Interceptor" structure in CardDef.
    }

    void TriggerManager::clear() {
        listeners.clear();
    }

}
