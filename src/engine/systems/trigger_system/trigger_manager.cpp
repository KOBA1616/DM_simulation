#include "trigger_manager.hpp"
// #include <iostream> // Logging disabled for core engine code

namespace dm::engine::systems {

    void TriggerManager::subscribe(core::EventType type, EventCallback callback) {
        listeners[type].push_back(callback);
    }

    void TriggerManager::dispatch(const core::GameEvent& event, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        // 1. Notify listeners (Passive checks, Logging, UI updates)
        auto it = listeners.find(event.type);
        if (it != listeners.end()) {
            std::vector<EventCallback> callbacks = it->second;
            for (const auto& callback : callbacks) {
                callback(event, state);
            }
        }

        // 2. Check for triggers (Trigger Search) -> Generates PendingEffects
        check_triggers(event, state, card_db);

        // 3. Interceptor checks (TODO: Phase 6.1 - Interceptors/Replacement Effects)
    }

    void TriggerManager::check_triggers(const core::GameEvent& event, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        // Helper to check triggers for a specific card instance
        auto check_card_trigger = [&](const core::CardInstance& card, int instance_id) {
            if (card.card_id == 0) return; // Ignore dummy cards
            if (card_db.find(card.card_id) == card_db.end()) return; // Safety check

            const auto& def = card_db.at(card.card_id);
            bool triggers = false;

            // Map EventType to CardKeywords or TriggerType

            if (event.type == core::EventType::ZONE_ENTER) {
                // Check CIP (On Play)
                if (event.target_id == instance_id) { // Self entered
                    if (def.keywords.cip) triggers = true;
                }
            }
            else if (event.type == core::EventType::ATTACK_INITIATE) {
                // Check AT_ATTACK
                if (event.source_id == instance_id) { // Self attacking
                    if (def.keywords.at_attack) triggers = true;
                }
            }
            else if (event.type == core::EventType::ZONE_LEAVE) {
                 // Check ON_DESTROY (if dest is Grave)
                 // NOTE: ZONE_LEAVE must be dispatched BEFORE the card is removed from the Battle Zone,
                 // otherwise this loop will not find the card to trigger the effect.
                 // This assumes the event is pre-movement or concurrent.
                 if (event.target_id == instance_id) {
                     if (def.keywords.destruction) triggers = true;
                 }
            }

            if (triggers) {
                // Create PendingEffect using the constructor: type, source, controller
                core::PendingEffect effect(core::EffectType::TRIGGER_ABILITY, instance_id, card.owner);

                // Note: Optional effect_def loading will happen later or here if needed

                state.pending_effects.push_back(effect);
            }
        };

        // Scan zones based on event type
        // For Phase 6 Step 1, we replicate the scan.

        for (int p_id = 0; p_id < 2; ++p_id) {
             const auto& battle_zone = state.players[p_id].battle_zone;
             for (size_t i = 0; i < battle_zone.size(); ++i) {
                 check_card_trigger(battle_zone[i], battle_zone[i].instance_id);
             }
        }
    }

    void TriggerManager::clear() {
        listeners.clear();
    }

}
