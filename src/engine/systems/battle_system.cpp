#include "battle_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
#include "engine/systems/flow/reaction_system.hpp"
#include "engine/systems/command_system.hpp"

#include <algorithm>
#include <iostream>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void BattleSystem::handle_attack(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        // 1. Identify Attacker
        Player& attacker_player = game_state.players[game_state.active_player_id];
        auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(),
             [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });

        if (it == attacker_player.battle_zone.end()) {
            std::cerr << "BattleSystem: Attacker not found in battle zone." << std::endl;
            return;
        }

        // 2. Set GameState Attack Info (Context)
        game_state.current_attack.source_instance_id = action.source_instance_id;
        game_state.current_attack.target_instance_id = (action.type == ActionType::ATTACK_CREATURE) ? action.target_instance_id : -1;
        game_state.current_attack.target_player = action.target_player;
        game_state.current_attack.is_blocked = false;
        game_state.current_attack.blocker_instance_id = -1;

        // 3. Tap Attacker (Mutation Command)
        auto tap_cmd = std::make_shared<MutateCommand>(
            action.source_instance_id,
            MutateCommand::MutationType::TAP
        );
        game_state.execute_command(tap_cmd);

        // 4. Trigger Events (Event-Driven)
        // Dispatch ATTACK event
        GameEvent attack_event(EventType::ATTACK, action.source_instance_id, game_state.active_player_id);
        attack_event.context.target_instance_id = game_state.current_attack.target_instance_id;
        attack_event.context.target_player = game_state.current_attack.target_player;
        TriggerManager::instance().dispatch(attack_event, game_state);

        // Also trigger specific legacy triggers if needed (via TriggerManager in Phase 6 Step 1, or direct calls if transition incomplete)
        // Ideally TriggerManager handles everything. But for now, we rely on TriggerManager to queue effects.

        // 5. Reaction Window (Ninja Strike / Revolution Change)
        // Note: Revolution Change (ON_ATTACK_FROM_HAND) is checked during Attack Handling usually
        // ReactionSystem logic remains for now as it handles specific window opening.
        // Ideally this should be triggered by the Event.
        ReactionSystem::check_and_open_window(game_state, card_db, "ON_ATTACK", game_state.active_player_id);
    }

    void BattleSystem::handle_block(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        // 1. Update Context
        game_state.current_attack.is_blocked = true;
        game_state.current_attack.blocker_instance_id = action.source_instance_id;

        // 2. Tap Blocker (Mutation Command)
        auto tap_cmd = std::make_shared<MutateCommand>(
            action.source_instance_id,
            MutateCommand::MutationType::TAP
        );
        game_state.execute_command(tap_cmd);

        // 3. Trigger Event
        GameEvent block_event(EventType::BLOCK, action.source_instance_id, game_state.get_non_active_player().id);
        block_event.context.target_instance_id = game_state.current_attack.source_instance_id; // Blocked creature
        TriggerManager::instance().dispatch(block_event, game_state);

        // 4. Queue Battle Resolution
        // We explicitly queue RESOLVE_BATTLE.
        // Note: EffectResolver had logic to queue this if PASS was selected.
        // Here we queue it immediately after block.
        game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, action.source_instance_id, game_state.active_player_id);
    }

    void BattleSystem::resolve_battle(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        int attacker_id = game_state.current_attack.source_instance_id;
        int defender_id = -1;

        if (game_state.current_attack.is_blocked) {
            defender_id = game_state.current_attack.blocker_instance_id;
        } else if (game_state.current_attack.target_instance_id != -1) {
            defender_id = game_state.current_attack.target_instance_id;
        } else {
            // Direct Attack - Transition to Break Shield
            Player& attacker_player = game_state.get_active_player();
            int breaker_count = 1;

            // Find attacker to check breaker count
            auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(),
                [&](const CardInstance& c){ return c.instance_id == attacker_id; });

            if (it != attacker_player.battle_zone.end()) {
                breaker_count = get_breaker_count(*it, card_db);
            }

            Player& defender = game_state.get_non_active_player();
            if (defender.shield_zone.empty()) {
                 game_state.pending_effects.emplace_back(EffectType::BREAK_SHIELD, attacker_id, game_state.active_player_id);
            } else {
                int shields_to_break = std::min((int)defender.shield_zone.size(), breaker_count);
                for (int i=0; i<shields_to_break; ++i) {
                     game_state.pending_effects.emplace_back(EffectType::BREAK_SHIELD, attacker_id, game_state.active_player_id);
                }
            }
            return;
        }

        // Creature vs Creature Battle
        Player& p1 = game_state.get_active_player();
        Player& p2 = game_state.get_non_active_player();
        CardInstance* attacker = nullptr;
        CardInstance* defender = nullptr;

        // Locate instances (could be anywhere if moved, but usually Battle Zone)
        for (auto& c : p1.battle_zone) if (c.instance_id == attacker_id) attacker = &c;
        if (!attacker) {
             // If blocked, we might check blocker owner's battle zone for attacker? No, attacker is always Active Player.
             // If attacker is removed before battle, battle ends.
             return;
        }

        // Defender could be in P2 (Blocker or Target) or P1 (Targeting own creature? rare but possible)
        // Assuming P2 for standard battle
        for (auto& c : p2.battle_zone) if (c.instance_id == defender_id) defender = &c;

        if (!defender) {
            if (game_state.current_attack.is_blocked) return; // Blocker vanished
             // If target vanished, attack fizzles? Or goes to player? Rules say fizzles.
            return;
        }

        int p_att = get_creature_power(*attacker, game_state, card_db);
        int p_def = get_creature_power(*defender, game_state, card_db);

        bool att_wins = p_att > p_def;
        bool def_wins = p_def > p_att;
        bool draw = p_att == p_def;

        if (att_wins || draw) {
            // Destroy Defender
            auto destroy_cmd = std::make_shared<TransitionCommand>(
                defender->instance_id, Zone::BATTLE, Zone::GRAVEYARD, defender->owner_id // Use owner_id if available or infer
            );
            // Infer owner: Usually P2.
            destroy_cmd->player_id = game_state.get_non_active_player().id;
            game_state.execute_command(destroy_cmd);

            GameEvent event(EventType::DESTROY, defender->instance_id, destroy_cmd->player_id);
            TriggerManager::instance().dispatch(event, game_state);
        }

        if (def_wins || draw) {
            // Destroy Attacker
            auto destroy_cmd = std::make_shared<TransitionCommand>(
                attacker->instance_id, Zone::BATTLE, Zone::GRAVEYARD, game_state.active_player_id
            );
            game_state.execute_command(destroy_cmd);

            GameEvent event(EventType::DESTROY, attacker->instance_id, game_state.active_player_id);
            TriggerManager::instance().dispatch(event, game_state);
        }
    }

    void BattleSystem::resolve_break_shield(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         PlayerID target_pid = action.target_player;
         if (target_pid == 255) {
             target_pid = game_state.get_non_active_player().id;
         }
         Player& defender = game_state.players[target_pid];

         if (defender.shield_zone.empty()) {
             // Direct Attack Success
             if (target_pid != game_state.active_player_id) {
                 game_state.winner = (game_state.active_player_id == 0) ? GameResult::P1_WIN : GameResult::P2_WIN;
             }
             return;
         }

         // Trigger AT_BREAK_SHIELD event
         GameEvent break_event(EventType::BREAK_SHIELD, action.source_instance_id, game_state.active_player_id);
         TriggerManager::instance().dispatch(break_event, game_state);

         // Identify Shield
         int shield_index = -1;
         if (action.target_instance_id != -1) {
             for (size_t i = 0; i < defender.shield_zone.size(); ++i) {
                 if (defender.shield_zone[i].instance_id == action.target_instance_id) {
                     shield_index = i;
                     break;
                 }
             }
         }
         if (shield_index == -1) {
             shield_index = defender.shield_zone.size() - 1;
         }
         CardInstance shield = defender.shield_zone[shield_index];

         // Check Shield Burn Logic
         bool shield_burn = false;
         Player& attacker_player = game_state.get_active_player();
         auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(),
             [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });
         if (it != attacker_player.battle_zone.end() && card_db.count(it->card_id)) {
             if (card_db.at(it->card_id).keywords.shield_burn) {
                 shield_burn = true;
             }
         }

         if (shield_burn) {
             auto move_cmd = std::make_shared<TransitionCommand>(
                 shield.instance_id, Zone::SHIELD, Zone::GRAVEYARD, defender.id
             );
             game_state.execute_command(move_cmd);

             // Dispatch Destroy Event
             GameEvent destroy_event(EventType::DESTROY, shield.instance_id, defender.id);
             TriggerManager::instance().dispatch(destroy_event, game_state);
         } else {
             auto move_cmd = std::make_shared<TransitionCommand>(
                 shield.instance_id, Zone::SHIELD, Zone::HAND, defender.id
             );
             game_state.execute_command(move_cmd);

             // Check ST (Shield Trigger)
             bool is_trigger = false;
             if (card_db.count(shield.card_id)) {
                 const auto& def = card_db.at(shield.card_id);
                 if (TargetUtils::has_keyword_simple(game_state, shield, def, "SHIELD_TRIGGER")) {
                     is_trigger = true;
                 }
             }

             if (is_trigger) {
                 // Trigger S_TRIGGER event
                 // Currently using PendingEffect directly for ST execution queue
                 game_state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield.instance_id, defender.id);
             }

             // Reaction Window (Strike Back)
             ReactionSystem::check_and_open_window(game_state, card_db, "ON_SHIELD_ADD", defender.id);
         }
    }

    int BattleSystem::get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;
        power += creature.power_mod;
        power += PassiveEffectSystem::instance().get_power_buff(game_state, creature, card_db);
        return power;
    }

    int BattleSystem::get_breaker_count(const CardInstance& creature, const std::map<CardID, CardDefinition>& card_db) {
         if (!card_db.count(creature.card_id)) return 1;
         const auto& k = card_db.at(creature.card_id).keywords;
         if (k.triple_breaker) return 3;
         if (k.double_breaker) return 2;
         return 1;
    }
}
