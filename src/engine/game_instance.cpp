#include "game_instance.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    void GameInstance::reset_with_scenario(const ScenarioConfig& config) {
        // 1. Reset Game State
        // Should we re-seed? The constructor took a seed, but state is already constructed.
        // We can just clear vectors.

        // Reset basic counters
        state.turn_number = 1; // Or some meaningful turn? Spec says maybe turn 5 for combo.
        // Let's default to turn 5 as per spec "state.current_turn = 5".
        state.turn_number = 5;
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN; // Start in Main Phase usually for scenarios?
        state.winner = GameResult::NONE;
        state.pending_effects.clear();
        state.current_attack = AttackRequest(); // Reset attack context

        // Clear all zones for both players
        for (auto& p : state.players) {
            p.hand.clear();
            p.battle_zone.clear();
            p.mana_zone.clear();
            p.graveyard.clear();
            p.shield_zone.clear();
            p.deck.clear(); // We might not need deck for scenario?
            // Usually scenarios assume deck is irrelevant or we can fill it with dummy cards?
        }

        // Instance ID counter
        int instance_id_counter = 0;

        // 2. Setup My Resources (Player 0)
        Player& me = state.players[0];

        // Hand
        for (int cid : config.my_hand_cards) {
            me.hand.emplace_back((CardID)cid, instance_id_counter++);
        }

        // Battle Zone
        for (int cid : config.my_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++);
            c.summoning_sickness = false; // Assume creatures on board are ready
            me.battle_zone.push_back(c);
        }

        // Mana Zone
        // Config provides IDs for mana cards
        for (int cid : config.my_mana_zone) {
            CardInstance c((CardID)cid, instance_id_counter++);
            c.is_tapped = false; // Untapped by default?
            me.mana_zone.push_back(c);
        }

        // If my_mana count is specified but not enough cards in mana_zone,
        // we might need to add dummy cards or rely on the user to provide enough IDs?
        // The spec has `int my_mana` AND `vector<int> my_mana_zone`.
        // If `my_mana_zone` is empty but `my_mana` > 0, we can add dummy cards or generic mana sources?
        // Or `my_mana` implies available mana?
        // Let's prioritize explicit `my_mana_zone` if provided.
        // If `my_mana_zone` is empty and `my_mana` > 0, we'll try to add dummy cards (ID 1?)
        if (config.my_mana_zone.empty() && config.my_mana > 0) {
            for (int i = 0; i < config.my_mana; ++i) {
                // Use a default card ID if possible, or 0? 0 might be invalid.
                // We don't have a reliable "Basic Land" equivalent.
                // Let's assume ID 1 exists for now or use 0.
                me.mana_zone.emplace_back(1, instance_id_counter++);
            }
        }

        // Graveyard
        for (int cid : config.my_grave_yard) {
            me.graveyard.emplace_back((CardID)cid, instance_id_counter++);
        }

        // My Shields (Player 0)
        for (int cid : config.my_shields) {
             me.shield_zone.emplace_back((CardID)cid, instance_id_counter++);
        }

        // 3. Setup Enemy Resources (Player 1)
        Player& enemy = state.players[1];

        // Enemy Battle Zone
        for (int cid : config.enemy_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++);
            c.summoning_sickness = false;
            enemy.battle_zone.push_back(c);
        }

        // Enemy Shields
        for (int i = 0; i < config.enemy_shield_count; ++i) {
             // Use ID 0 or some dummy ID for shields if not specified
             // Shields are face down, so ID matters for triggers.
             // If we want to test triggers, we need to specify them.
             // The config `enemy_can_use_trigger` is a boolean flag, maybe controlling AI behavior?
             // But for shields, we need cards.
             // Let's add dummy cards (ID 1) for now.
             enemy.shield_zone.emplace_back(1, instance_id_counter++);
        }

        // 4. Update Stats / Cache (if any)
        // GameState::initialize_card_stats should have been called before or we can call it here if we had access to full DB?
        // But initialize_card_stats is for stats tracking system.
        // We can just assume stats are handled externally.

        std::cout << "Scenario loaded. Turn: " << state.turn_number << std::endl;
    }

}
