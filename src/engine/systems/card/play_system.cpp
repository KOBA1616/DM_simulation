#include "play_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/cost_payment_system.hpp"
#include "engine/systems/command_system.hpp"

#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void PlaySystem::handle_play_card(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        // Step 1: Transition Hand -> Stack
        auto move_cmd = std::make_shared<TransitionCommand>(
            action.source_instance_id,
            Zone::HAND,
            Zone::STACK,
            game_state.active_player_id
        );
        game_state.execute_command(move_cmd);

        // Step 2: Initialize Stack State
        if (!game_state.stack_zone.empty() && game_state.stack_zone.back().instance_id == action.source_instance_id) {
            auto& stack_card = game_state.stack_zone.back();
            stack_card.is_tapped = false;
            stack_card.summoning_sickness = true;

            // Handle metadata / evolution source passing
            // Legacy convention: target_instance_id used for evolution source or power mod
            if (action.target_instance_id != -1) {
                stack_card.power_mod = action.target_instance_id;
            } else {
                stack_card.power_mod = -1;
            }
        }
    }

    void PlaySystem::handle_pay_cost(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& player = game_state.players[game_state.active_player_id];
        CardInstance* card = nullptr;
        if (!game_state.stack_zone.empty() && game_state.stack_zone.back().instance_id == action.source_instance_id) {
            card = &game_state.stack_zone.back();
        }

        if (card && card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);
            bool paid = ManaSystem::auto_tap_mana(game_state, player, def, card_db);

            if (paid) {
                // Mark as paid (using is_tapped as flag on stack)
                card->is_tapped = true;
            } else {
                // Payment Failed: Rollback
                if (!game_state.stack_zone.empty() && game_state.stack_zone.back().instance_id == action.source_instance_id) {
                    // Pop from stack, return to hand
                    CardInstance c = game_state.stack_zone.back();
                    game_state.stack_zone.pop_back();
                    c.is_tapped = false;
                    player.hand.push_back(c);
                }
            }
        }
    }

    void PlaySystem::resolve_play_from_stack(GameState& game_state, int stack_instance_id, int cost_reduction, SpawnSource spawn_source, PlayerID controller, const std::map<CardID, CardDefinition>& card_db, int evo_source_id, int dest_override) {
        // Locate card in Stack or Buffers
        auto& stack = game_state.stack_zone;
        auto it = std::find_if(stack.begin(), stack.end(), [&](const CardInstance& c){ return c.instance_id == stack_instance_id; });

        CardInstance card;
        bool found = false;

        if (it != stack.end()) {
            card = *it;
            if (evo_source_id == -1 && card.power_mod != -1) {
                 evo_source_id = card.power_mod; // Retrieve passed evo source
            }
            card.power_mod = 0; // Reset temp storage
            stack.erase(it);
            found = true;
        } else {
             // Check buffers logic (simplified here, assuming standard play for now)
             // ... buffer logic if needed ...
        }

        if (!found) return;

        Player& player = game_state.players[controller];
        const CardDefinition* def = nullptr;
        if (card_db.count(card.card_id)) def = &card_db.at(card.card_id);

        Zone dest_zone = Zone::BATTLE; // Default for creatures
        if (def && def->type == CardType::SPELL) {
            dest_zone = Zone::GRAVEYARD;
            if (dest_override == 1) dest_zone = Zone::DECK; // Logic for bottom of deck needs handling in CommandSystem
        }

        if (def && def->type == CardType::SPELL) {
             // Spells go to GY
             if (dest_override == 1) {
                 // Deck Bottom
                 player.deck.insert(player.deck.begin(), card);
             } else {
                 player.graveyard.push_back(card);
             }

             // Trigger Events
             GameEvent play_event(EventType::PLAY, card.instance_id, controller);
             TriggerManager::instance().dispatch(play_event, game_state);

             GameEvent spell_event(EventType::CAST_SPELL, card.instance_id, controller);
             TriggerManager::instance().dispatch(spell_event, game_state);

             game_state.turn_stats.spells_cast_this_turn++;

        } else {
            // Creatures
            card.summoning_sickness = true;
            if (def && (def->keywords.speed_attacker || def->keywords.evolution)) {
                card.summoning_sickness = false;
            }
            card.is_tapped = false;
            card.turn_played = game_state.turn_number;

            // Handle Evolution
            if (evo_source_id != -1) {
                auto s_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == evo_source_id; });
                if (s_it != player.battle_zone.end()) {
                    CardInstance source = *s_it;
                    player.battle_zone.erase(s_it);
                    card.underlying_cards.push_back(source);
                }
            }

            player.battle_zone.push_back(card);

            GameEvent play_event(EventType::PLAY, card.instance_id, controller);
            TriggerManager::instance().dispatch(play_event, game_state);

            game_state.turn_stats.creatures_played_this_turn++;
        }

        // Final Callback logic (stats)
        game_state.on_card_play(card.card_id, game_state.turn_number, spawn_source != SpawnSource::HAND_SUMMON, cost_reduction, controller);
    }

    void PlaySystem::handle_mana_charge(GameState& game_state, const Action& action) {
        // Command: Hand -> Mana
        auto move_cmd = std::make_shared<TransitionCommand>(
            action.source_instance_id,
            Zone::HAND,
            Zone::MANA,
            game_state.active_player_id
        );
        game_state.execute_command(move_cmd);

        // Command: Untap (Mana enters untapped)
        auto untap_cmd = std::make_shared<MutateCommand>(
            action.source_instance_id,
            MutateCommand::MutationType::UNTAP
        );
        game_state.execute_command(untap_cmd);
    }

}
