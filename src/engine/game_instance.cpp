#include "game_instance.hpp"
#include "systems/card/generic_card_system.hpp"
#include "systems/battle_system.hpp"
#include "systems/card/play_system.hpp"
#include "systems/mana/mana_system.hpp"
#include "game_command/commands.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;
    using namespace dm::engine::systems;

    GameInstance::GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db)
        : state(0), card_db(db) // Seed unused in constructor for state, but maybe used later? State takes card_count usually
    {
        // Initialize trigger manager if needed, but it's a singleton usually.
        // But here we might want to attach it to state.
        trigger_manager = std::make_shared<TriggerManager>();

        // Connect TriggerManager to GameState
        state.event_dispatcher = [this](const GameEvent& evt) {
            trigger_manager->dispatch(evt, state);
        };
    }

    void GameInstance::reset_with_scenario(const core::ScenarioConfig& config) {
        // Reset state
        state = GameState(0);
        state.event_dispatcher = [this](const GameEvent& evt) {
             trigger_manager->dispatch(evt, state);
        };

        // Players setup
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN;
        state.turn_number = 1;

        // Apply config
        auto setup_zone = [&](PlayerID pid, Zone zone, const std::vector<int>& cards) {
            Player& p = state.players[pid];
            std::vector<CardInstance>* target_zone = nullptr;
            if (zone == Zone::HAND) target_zone = &p.hand;
            else if (zone == Zone::MANA) target_zone = &p.mana_zone;
            else if (zone == Zone::BATTLE) target_zone = &p.battle_zone;
            else if (zone == Zone::SHIELD) target_zone = &p.shield_zone;
            else if (zone == Zone::GRAVEYARD) target_zone = &p.graveyard;
            else if (zone == Zone::DECK) target_zone = &p.deck;

            if (target_zone) {
                target_zone->clear();
                int id_base = (pid + 1) * 10000;
                for (size_t i = 0; i < cards.size(); ++i) {
                    CardInstance c;
                    c.card_id = cards[i];
                    c.instance_id = id_base + i;
                    c.owner = pid;
                    if (zone == Zone::MANA) c.is_tapped = false; // Mana enters untapped usually in scenario
                    if (zone == Zone::BATTLE) {
                         c.is_tapped = false;
                         c.summoning_sickness = true;
                         c.turn_played = state.turn_number;
                    }
                    target_zone->push_back(c);

                    // Update owner map
                    if (state.card_owner_map.size() <= (size_t)c.instance_id) {
                        state.card_owner_map.resize(c.instance_id + 1000, 255);
                    }
                    state.card_owner_map[c.instance_id] = pid;
                }
            }
        };

        // My Zones
        setup_zone(0, Zone::HAND, config.my_hand_cards);
        setup_zone(0, Zone::MANA, config.my_mana_zone);
        setup_zone(0, Zone::BATTLE, config.my_battle_zone);
        setup_zone(0, Zone::GRAVEYARD, config.my_grave_yard);
        setup_zone(0, Zone::SHIELD, config.my_shields);
        setup_zone(0, Zone::DECK, config.my_deck);

        // Enemy Zones
        setup_zone(1, Zone::BATTLE, config.enemy_battle_zone);
        setup_zone(1, Zone::DECK, config.enemy_deck);

        // Enemy Shields (Just count)
        Player& p2 = state.players[1];
        p2.shield_zone.clear();
        for (int i=0; i<config.enemy_shield_count; ++i) {
            CardInstance c;
            c.card_id = 0; // Dummy
            c.instance_id = 20000 + 500 + i;
            c.owner = 1;
            p2.shield_zone.push_back(c);
             if (state.card_owner_map.size() <= (size_t)c.instance_id) {
                 state.card_owner_map.resize(c.instance_id + 1000, 255);
             }
             state.card_owner_map[c.instance_id] = 1;
        }

        // Loop Proof Mode
        state.loop_proven = false;
    }

    void GameInstance::undo() {
        if (state.command_history.empty()) return;
        // Basic undo implementation
        auto cmd = state.command_history.back();
        state.command_history.pop_back();
        // cmd->revert(state); // If implemented
    }
}
