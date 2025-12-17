#include "action_dispatcher.hpp"
#include "battle_system.hpp"
#include "card/play_system.hpp"
#include "card/generic_card_system.hpp"
#include "mana/mana_system.hpp"
#include "flow/reaction_system.hpp"
#include "engine/systems/card/handlers/attack_handler.hpp" // For shared logic if any
#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace dm::core;

    void ActionDispatcher::dispatch(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        switch (action.type) {
             case ActionType::PASS:
                 // In BLOCK phase, a pass means no blockers were declared, so queue battle resolution.
                 if (game_state.current_phase == Phase::BLOCK) {
                     const bool has_battle_pending = std::any_of(
                         game_state.pending_effects.begin(),
                         game_state.pending_effects.end(),
                         [](const PendingEffect& eff) { return eff.type == EffectType::RESOLVE_BATTLE; }
                     );

                     if (!has_battle_pending && game_state.current_attack.source_instance_id != -1) {
                         game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                     }
                 }
                 break;
             case ActionType::MANA_CHARGE:
                 PlaySystem::handle_mana_charge(game_state, action);
                 break;
             case ActionType::MOVE_CARD:
                 if (game_state.current_phase == Phase::MANA) {
                     PlaySystem::handle_mana_charge(game_state, action);
                 }
                 break;
             case ActionType::PLAY_CARD:
             case ActionType::DECLARE_PLAY:
                 PlaySystem::handle_play_card(game_state, action, card_db);
                 break;
             case ActionType::PAY_COST:
                 PlaySystem::handle_pay_cost(game_state, action, card_db);
                 break;
             case ActionType::RESOLVE_PLAY:
                 PlaySystem::resolve_play_from_stack(game_state, action.source_instance_id, 0, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db, action.target_instance_id);
                 break;
             case ActionType::ATTACK_PLAYER:
             case ActionType::ATTACK_CREATURE:
                 BattleSystem::handle_attack(game_state, action, card_db);
                 break;
             case ActionType::BLOCK:
                 BattleSystem::handle_block(game_state, action, card_db);
                 break;
             case ActionType::USE_SHIELD_TRIGGER:
                 GenericCardSystem::resolve_trigger(game_state, TriggerType::S_TRIGGER, action.source_instance_id, card_db);
                 break;
             case ActionType::SELECT_TARGET:
                 handle_select_target(game_state, action);
                 break;
             case ActionType::RESOLVE_EFFECT:
                 handle_resolve_effect(game_state, action, card_db);
                 break;
             case ActionType::USE_ABILITY:
                 handle_use_ability(game_state, action, card_db);
                 break;
             case ActionType::PLAY_CARD_INTERNAL:
             {
                 int stack_id = action.source_instance_id;
                 PlayerID controller = action.target_player;
                 if (action.spawn_source == SpawnSource::HAND_SUMMON) {
                     Player& player = game_state.players[controller];
                     auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == stack_id; });
                     if (it != player.hand.end()) {
                         CardInstance c = *it;
                         player.hand.erase(it);
                         game_state.stack_zone.push_back(c);
                     }
                 }
                 int dest_override = action.destination_override;
                 PlaySystem::resolve_play_from_stack(game_state, stack_id, 999, action.spawn_source, controller, card_db, -1, dest_override);

                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                 }
                 break;
             }
             case ActionType::RESOLVE_BATTLE:
                 BattleSystem::resolve_battle(game_state, card_db);
                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                 }
                 break;
             case ActionType::BREAK_SHIELD:
                 BattleSystem::resolve_break_shield(game_state, action, card_db);
                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                 }
                 break;
             case ActionType::DECLARE_REACTION:
                 handle_declare_reaction(game_state, action, card_db);
                 break;
             case ActionType::SELECT_OPTION:
                 handle_select_option(game_state, action, card_db);
                 break;
             case ActionType::SELECT_NUMBER:
                 handle_select_number(game_state, action, card_db);
                 break;
             default:
                 break;
        }
    }

    void ActionDispatcher::handle_select_target(GameState& game_state, const Action& action) {
        if (action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
            auto& pe = game_state.pending_effects[action.slot_index];
            if (pe.resolve_type == ResolveType::TARGET_SELECT) {
                pe.target_instance_ids.push_back(action.target_instance_id);
            }
        }
    }

    void ActionDispatcher::handle_resolve_effect(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
             auto& pe = game_state.pending_effects[action.slot_index];

             // Loop Prevention Check
             if (pe.chain_depth > 50) {
                 std::cerr << "Loop Prevention: Fizzling effect chain depth " << pe.chain_depth << std::endl;
             } else {
                 int prev_depth = game_state.turn_stats.current_chain_depth;
                 game_state.turn_stats.current_chain_depth = pe.chain_depth;

                 if (pe.resolve_type == ResolveType::TARGET_SELECT && pe.effect_def) {
                     GenericCardSystem::resolve_effect_with_targets(game_state, *pe.effect_def, pe.target_instance_ids, pe.source_instance_id, card_db, pe.execution_context);
                 } else if (pe.type == EffectType::TRIGGER_ABILITY && pe.effect_def) {
                     GenericCardSystem::resolve_effect(game_state, *pe.effect_def, pe.source_instance_id, card_db);
                 }

                 game_state.turn_stats.current_chain_depth = prev_depth;
             }

             if (action.slot_index < (int)game_state.pending_effects.size()) {
                 game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
             }
         }
    }

    void ActionDispatcher::handle_use_ability(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         Player& player = game_state.players[game_state.active_player_id];
         auto hand_it = std::find_if(player.hand.begin(), player.hand.end(),
             [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });
         if (hand_it != player.hand.end()) {
             CardInstance hand_card = *hand_it;

             int attacker_id = action.target_instance_id;
             if (attacker_id == -1) {
                 attacker_id = game_state.current_attack.source_instance_id;
             }

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

                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, hand_card.instance_id, card_db);
             }
         }
    }

    void ActionDispatcher::handle_declare_reaction(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         Player& controller = game_state.players[action.target_player];
         auto it = std::find_if(controller.hand.begin(), controller.hand.end(), [&](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
         if (it != controller.hand.end()) {
             CardInstance card = *it;
             controller.hand.erase(it);
             game_state.stack_zone.push_back(card);
             PlaySystem::resolve_play_from_stack(game_state, card.instance_id, 999, SpawnSource::EFFECT_SUMMON, controller.id, card_db);
         }
    }

    void ActionDispatcher::handle_select_option(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
             auto& pe = game_state.pending_effects[action.slot_index];
             if (pe.type == EffectType::SELECT_OPTION) {
                 int option_index = action.target_slot_index;
                 if (option_index >= 0 && option_index < (int)pe.options.size()) {
                     const auto& selected_actions = pe.options[option_index];
                     EffectDef temp_effect;
                     temp_effect.actions = selected_actions;
                     temp_effect.trigger = TriggerType::NONE;
                     GenericCardSystem::resolve_effect_with_context(game_state, temp_effect, pe.source_instance_id, pe.execution_context, card_db);
                 }
             }
             if (action.slot_index < (int)game_state.pending_effects.size()) {
                 game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
             }
         }
    }

    void ActionDispatcher::handle_select_number(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
            auto& pe = game_state.pending_effects[action.slot_index];
            if (pe.type == EffectType::SELECT_NUMBER) {
                int chosen_val = action.target_instance_id;
                std::string output_key;
                if (pe.effect_def && !pe.effect_def->condition.str_val.empty()) {
                    output_key = pe.effect_def->condition.str_val;
                }
                if (!output_key.empty()) {
                    pe.execution_context[output_key] = chosen_val;
                }
                if (pe.effect_def && !pe.effect_def->actions.empty()) {
                     GenericCardSystem::resolve_effect_with_context(game_state, *pe.effect_def, pe.source_instance_id, pe.execution_context, card_db);
                }
            }
             if (action.slot_index < (int)game_state.pending_effects.size()) {
                 game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
             }
        }
    }

}
