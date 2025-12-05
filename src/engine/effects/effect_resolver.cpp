#include "effect_resolver.hpp"
#include "../action_gen/action_generator.hpp"
#include "../mana/mana_system.hpp"
#include "../card_system/card_registry.hpp"
#include "../card_system/target_utils.hpp"
#include "../card_system/generic_card_system.hpp"
#include "../flow/reaction_system.hpp"

#include <iostream>
#include <algorithm>

namespace dm::engine {

    using namespace dm::core;

    // Helper to find and remove from hand
    static CardInstance remove_from_hand(Player& player, int instance_id) {
        auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == instance_id; });
        if (it != player.hand.end()) {
            CardInstance c = *it;
            player.hand.erase(it);
            return c;
        }
        return CardInstance(); // Should not happen if validated
    }

    void EffectResolver::resolve_action(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        switch (action.type) {
             case ActionType::PASS:
                 break;
             case ActionType::MANA_CHARGE:
             case ActionType::MOVE_CARD:
                 resolve_mana_charge(game_state, action);
                 break;
             case ActionType::PLAY_CARD:
             case ActionType::DECLARE_PLAY:
                 resolve_play_card(game_state, action, card_db);
                 break;
             case ActionType::PAY_COST:
                 {
                     Player& player = game_state.players[game_state.active_player_id];
                     CardInstance* card = nullptr;
                     if (!game_state.stack_zone.empty() && game_state.stack_zone.back().instance_id == action.source_instance_id) {
                         card = &game_state.stack_zone.back();
                     }
                     if (card && card_db.count(card->card_id)) {
                         const auto& def = card_db.at(card->card_id);
                         ManaSystem::auto_tap_mana(game_state, player, def, card_db);
                         card->is_tapped = true;
                     }
                 }
                 break;
             case ActionType::RESOLVE_PLAY:
                 {
                     int stack_id = action.source_instance_id;
                     resolve_play_from_stack(game_state, stack_id, 0, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db);
                 }
                 break;
             case ActionType::ATTACK_PLAYER:
             case ActionType::ATTACK_CREATURE:
                 resolve_attack(game_state, action, card_db);
                 break;
             case ActionType::BLOCK:
                 resolve_block(game_state, action, card_db);
                 break;
             case ActionType::USE_SHIELD_TRIGGER:
                 resolve_use_shield_trigger(game_state, action, card_db);
                 break;
             case ActionType::SELECT_TARGET:
                 resolve_select_target(game_state, action);
                 break;
             case ActionType::RESOLVE_EFFECT:
                 {
                     if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                         auto& pe = game_state.pending_effects[action.slot_index];
                         if (pe.resolve_type == ResolveType::TARGET_SELECT && pe.effect_def) {
                             GenericCardSystem::resolve_effect_with_targets(game_state, *pe.effect_def, pe.target_instance_ids, pe.source_instance_id, card_db);
                         }
                         if (action.slot_index < (int)game_state.pending_effects.size()) {
                             game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                         }
                     }
                 }
                 break;
             case ActionType::USE_ABILITY:
                 resolve_use_ability(game_state, action, card_db);
                 break;
             case ActionType::PLAY_CARD_INTERNAL:
             {
                 int stack_id = action.source_instance_id;
                 resolve_play_from_stack(game_state, stack_id, 999, action.spawn_source, game_state.active_player_id, card_db);
                 break;
             }
             case ActionType::RESOLVE_BATTLE:
                 execute_battle(game_state, card_db);
                 break;
             case ActionType::BREAK_SHIELD:
                 resolve_break_shield(game_state, action, card_db);
                 break;
             case ActionType::DECLARE_REACTION:
                 resolve_reaction(game_state, action, card_db);
                 break;
             default:
                 break;
        }
    }

    void EffectResolver::resolve_mana_charge(GameState& game_state, const Action& action) {
        Player& player = game_state.players[game_state.active_player_id];
        auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        if (it != player.hand.end()) {
            CardInstance card = *it;
            player.hand.erase(it);
            card.is_tapped = false;
            player.mana_zone.push_back(card);
        }
    }

    void EffectResolver::resolve_play_card(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& player = game_state.players[game_state.active_player_id];

        // Handle Hyper Energy (Targeted)
        if (action.target_player == 254) { // Hyper Energy metadata
             auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
             if (it != player.hand.end()) {
                 CardInstance card = *it;
                 player.hand.erase(it);
                 game_state.stack_zone.push_back(card);

                 int taps_needed = action.target_slot_index;
                 int taps_done = 0;
                 for (auto& c : player.battle_zone) {
                     if (!c.is_tapped && !c.summoning_sickness) {
                         c.is_tapped = true;
                         taps_done++;
                         if (taps_done >= taps_needed) break;
                     }
                 }
                 // Immediate resolve for Hyper Energy
                 resolve_play_from_stack(game_state, card.instance_id, taps_done * 2, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db);
             }
             return;
        }

        auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        if (it != player.hand.end()) {
            CardInstance card = *it;
            player.hand.erase(it);
            card.is_tapped = false;
            card.summoning_sickness = true;
            game_state.stack_zone.push_back(card);
        }
    }

    void EffectResolver::resolve_attack(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& attacker = game_state.get_active_player();
        Player& defender = game_state.get_non_active_player();

        auto it = std::find_if(attacker.battle_zone.begin(), attacker.battle_zone.end(),
            [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });
        
        if (it == attacker.battle_zone.end()) return;
        CardInstance& card = *it;

        card.is_tapped = true;
        game_state.current_attack.source_instance_id = action.source_instance_id;
        game_state.current_attack.target_instance_id = (action.type == ActionType::ATTACK_CREATURE) ? action.target_instance_id : -1;
        game_state.current_attack.target_player = (action.type == ActionType::ATTACK_PLAYER) ? action.target_player : -1;
        game_state.current_attack.is_blocked = false;
        game_state.current_attack.blocker_instance_id = -1;

        GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_ATTACK, card.instance_id);

        ReactionSystem::check_and_open_window(game_state, card_db, "ON_ATTACK", defender.id);
    }

    void EffectResolver::resolve_block(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         game_state.current_attack.is_blocked = true;
         game_state.current_attack.blocker_instance_id = action.source_instance_id;

         Player& defender = game_state.get_non_active_player();
         auto it = std::find_if(defender.battle_zone.begin(), defender.battle_zone.end(),
             [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });
         if (it != defender.battle_zone.end()) {
             it->is_tapped = true;
             GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_BLOCK, it->instance_id);
         }

         game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, action.source_instance_id, game_state.active_player_id);
    }

    void EffectResolver::resolve_use_shield_trigger(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        GenericCardSystem::resolve_trigger(game_state, TriggerType::S_TRIGGER, action.source_instance_id);
    }

    void EffectResolver::resolve_select_target(GameState& game_state, const Action& action) {
        if (action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
            auto& pe = game_state.pending_effects[action.slot_index];
            if (pe.resolve_type == ResolveType::TARGET_SELECT) {
                pe.target_instance_ids.push_back(action.target_instance_id);
            }
        }
    }

    void EffectResolver::resolve_use_ability(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         Player& player = game_state.players[game_state.active_player_id];
         auto hand_it = std::find_if(player.hand.begin(), player.hand.end(),
             [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });

         if (hand_it != player.hand.end()) {
             CardInstance hand_card = *hand_it;
             int attacker_id = game_state.current_attack.source_instance_id;
             auto battle_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(),
                 [&](const CardInstance& c){ return c.instance_id == attacker_id; });

             if (battle_it != player.battle_zone.end()) {
                 CardInstance battle_card = *battle_it;
                 player.hand.erase(hand_it);
                 player.battle_zone.erase(battle_it);

                 hand_card.is_tapped = true;
                 hand_card.summoning_sickness = true;
                 player.battle_zone.push_back(hand_card);
                 game_state.current_attack.source_instance_id = hand_card.instance_id;

                 battle_card.is_tapped = false;
                 battle_card.summoning_sickness = true;
                 player.hand.push_back(battle_card);

                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, hand_card.instance_id);
             }
         }
    }

    int EffectResolver::get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;
        power += creature.power_mod;
        return power;
    }

    int EffectResolver::get_breaker_count(const CardInstance& creature, const std::map<CardID, CardDefinition>& card_db) {
         if (!card_db.count(creature.card_id)) return 1;
         const auto& k = card_db.at(creature.card_id).keywords;
         if (k.triple_breaker) return 3;
         if (k.double_breaker) return 2;
         return 1;
    }

    void EffectResolver::execute_battle(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        int attacker_id = game_state.current_attack.source_instance_id;
        int defender_id = -1;

        if (game_state.current_attack.is_blocked) {
            defender_id = game_state.current_attack.blocker_instance_id;
        } else if (game_state.current_attack.target_instance_id != -1) {
            defender_id = game_state.current_attack.target_instance_id;
        } else {
            game_state.pending_effects.emplace_back(EffectType::BREAK_SHIELD, attacker_id, game_state.active_player_id);
            return;
        }
        
        Player& p1 = game_state.get_active_player();
        Player& p2 = game_state.get_non_active_player();

        CardInstance* attacker = nullptr;
        CardInstance* defender = nullptr;

        for (auto& c : p1.battle_zone) if (c.instance_id == attacker_id) attacker = &c;
        if (!attacker) return;

        for (auto& c : p2.battle_zone) if (c.instance_id == defender_id) defender = &c;
        if (!defender) {
            if (game_state.current_attack.is_blocked) return;
            return;
        }

        int p_att = get_creature_power(*attacker, game_state, card_db);
        int p_def = get_creature_power(*defender, game_state, card_db);

        bool att_wins = p_att > p_def;
        bool def_wins = p_def > p_att;
        bool draw = p_att == p_def;

        if (att_wins || draw) {
            auto it = std::find_if(p2.battle_zone.begin(), p2.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == defender_id; });
            if (it != p2.battle_zone.end()) {
                p2.graveyard.push_back(*it);
                p2.battle_zone.erase(it);
                GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_DESTROY, defender_id);
            }
        }

        if (def_wins || draw) {
            auto it = std::find_if(p1.battle_zone.begin(), p1.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == attacker_id; });
            if (it != p1.battle_zone.end()) {
                p1.graveyard.push_back(*it);
                p1.battle_zone.erase(it);
                GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_DESTROY, attacker_id);
            }
        }
    }

    void EffectResolver::resolve_break_shield(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         Player& defender = game_state.get_non_active_player();
         if (defender.shield_zone.empty()) {
             game_state.winner = (game_state.active_player_id == 0) ? GameResult::P1_WIN : GameResult::P2_WIN;
             return;
         }

         CardInstance shield = defender.shield_zone.back();
         defender.shield_zone.pop_back();

         bool is_trigger = false;
         if (card_db.count(shield.card_id)) {
             const auto& def = card_db.at(shield.card_id);
             if (def.keywords.shield_trigger) {
                 is_trigger = true;
             }
         }

         if (is_trigger) {
             defender.hand.push_back(shield);
             game_state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield.instance_id, defender.id);
         } else {
             defender.hand.push_back(shield);
         }
    }

    void EffectResolver::resolve_play_from_stack(GameState& game_state, int stack_instance_id, int cost_reduction, SpawnSource spawn_source, PlayerID controller, const std::map<CardID, CardDefinition>& card_db) {
        auto& stack = game_state.stack_zone;
        auto it = std::find_if(stack.begin(), stack.end(), [&](const CardInstance& c){ return c.instance_id == stack_instance_id; });

        CardInstance card;
        bool found = false;

        if (it != stack.end()) {
            card = *it;
            stack.erase(it);
            found = true;
        } else {
            auto& buf = game_state.effect_buffer;
            auto bit = std::find_if(buf.begin(), buf.end(), [&](const CardInstance& c){ return c.instance_id == stack_instance_id; });
            if (bit != buf.end()) {
                card = *bit;
                buf.erase(bit);
                found = true;
            }
        }

        if (!found) return;

        Player& player = game_state.players[controller];
        const CardDefinition* def = nullptr;
        if (card_db.count(card.card_id)) def = &card_db.at(card.card_id);

        if (def && def->type == CardType::SPELL) {
            player.graveyard.push_back(card);
            GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, card.instance_id);
            game_state.turn_stats.spells_cast_this_turn++;
        } else {
            card.summoning_sickness = true;
            if (def && def->keywords.speed_attacker) card.summoning_sickness = false;
            if (def && def->keywords.evolution) card.summoning_sickness = false;

            player.battle_zone.push_back(card);
            GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, card.instance_id);
            game_state.turn_stats.creatures_played_this_turn++;
        }

        game_state.on_card_play(card.card_id, game_state.turn_number, spawn_source != SpawnSource::HAND_SUMMON, cost_reduction, controller);
    }

    void EffectResolver::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context) {
        GenericCardSystem::resolve_effect(game_state, effect, source_instance_id);
    }

    void EffectResolver::resolve_reaction(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& controller = game_state.players[action.target_player];
         CardInstance card = remove_from_hand(controller, action.source_instance_id);
         game_state.stack_zone.push_back(card);
         resolve_play_from_stack(game_state, card.instance_id, 999, SpawnSource::EFFECT_SUMMON, controller.id, card_db);
    }

}
