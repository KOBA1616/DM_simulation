#include <iostream>
#include <cassert>
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/types.hpp"

// Minimal test to verify TriggerManager logic
int main() {
    std::cout << "Running TriggerManager Test..." << std::endl;

    // 1. Setup
    dm::engine::systems::TriggerManager trigger_manager;
    dm::core::GameState state(100); // 100 cards

    // Initialize owner map for safety (though simple push_back handles logic in test)
    // GameState constructor might not init this if it expects deck loading
    // We manually resize it
    state.card_owner_map.resize(100, 255);

    std::map<dm::core::CardID, dm::core::CardDefinition> card_db;

    // Create a card definition with CIP
    dm::core::CardDefinition def;
    def.id = 1;
    def.name = "TestCreature";
    def.keywords.cip = true; // Enable OnPlay trigger
    card_db[1] = def;

    // Place an instance of this card in Battle Zone
    int instance_id = 0;
    dm::core::CardInstance card;
    card.instance_id = instance_id; // Fixed: was card.id
    card.card_id = 1; // Links to def
    card.owner = 0;

    state.players[0].battle_zone.push_back(card);
    state.card_owner_map[instance_id] = 0;

    // 2. Dispatch Event (ZONE_ENTER)
    dm::core::GameEvent event(dm::core::EventType::ZONE_ENTER);
    event.target_id = instance_id; // The card itself entered
    event.player_id = 0;

    std::cout << "Dispatching ZONE_ENTER event..." << std::endl;
    trigger_manager.dispatch(event, state, card_db);

    // 3. Verify
    // Expect 1 PendingEffect in state
    if (state.pending_effects.size() == 1) {
        const auto& eff = state.pending_effects[0];
        if (eff.type == dm::core::EffectType::TRIGGER_ABILITY && eff.source_instance_id == instance_id) {
            std::cout << "[PASS] Trigger correctly added." << std::endl;
        } else {
            std::cout << "[FAIL] Incorrect effect details." << std::endl;
            return 1;
        }
    } else {
        std::cout << "[FAIL] Expected 1 pending effect, found " << state.pending_effects.size() << std::endl;
        return 1;
    }

    std::cout << "All TriggerManager tests passed." << std::endl;
    return 0;
}
