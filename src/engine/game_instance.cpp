#include "game_instance.hpp"
#include "flow/phase_manager.hpp"

namespace dm::engine {

    using namespace dm::core;

    void GameInstance::reset_with_scenario(const ScenarioConfig& config) {
        // 1. Reset Game State
        state.turn_number = 5;
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN;
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
            p.deck.clear();
        }

        // Instance ID counter
        int instance_id_counter = 0;

        // Fill decks to prevent immediate deckout
        for (auto& p : state.players) {
             for(int i=0; i<30; ++i) {
                  p.deck.emplace_back((CardID)1, instance_id_counter++);
             }
        }

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
        for (int cid : config.my_mana_zone) {
            CardInstance c((CardID)cid, instance_id_counter++);
            c.is_tapped = false;
            me.mana_zone.push_back(c);
        }

        if (config.my_mana_zone.empty() && config.my_mana > 0) {
            for (int i = 0; i < config.my_mana; ++i) {
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
             enemy.shield_zone.emplace_back(1, instance_id_counter++);
        }
    }

}
