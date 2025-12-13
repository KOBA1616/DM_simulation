#include "effect_resolver.hpp"
#include "engine/actions/action_generator.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/flow/reaction_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
#include "engine/cost_payment_system.hpp"

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
        return CardInstance();
    }

    void EffectResolver::resolve_action(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
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
                        // Mark cost paid so StackStrategy can emit RESOLVE_PLAY
                        card->is_tapped = true;
                     }
                 }
                 break;
             case ActionType::RESOLVE_PLAY:
                 {
                     int stack_id = action.source_instance_id;
                     int evo_source_id = action.target_instance_id;
                     // Check destination override if passed via Action struct?
                     // RESOLVE_PLAY comes from StackStrategy, usually standard play.
                     // But if we want to override, we should pass it.
                     // Currently resolve_play_from_stack does not take override.
                     // But PLAY_CARD_INTERNAL calls resolve_play_from_stack.
                     resolve_play_from_stack(game_state, stack_id, 0, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db, evo_source_id);
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
                             GenericCardSystem::resolve_effect_with_targets(game_state, *pe.effect_def, pe.target_instance_ids, pe.source_instance_id, card_db, pe.execution_context);
                         } else if (pe.type == EffectType::TRIGGER_ABILITY && pe.effect_def) {
                             GenericCardSystem::resolve_effect(game_state, *pe.effect_def, pe.source_instance_id);
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

                 // Step 3-3: Destination Override logic
                 int dest_override = action.destination_override; // 0=Default, 1=Deck Bottom

                 resolve_play_from_stack(game_state, stack_id, 999, action.spawn_source, controller, card_db, -1, dest_override);

                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                 }
                 break;
             }
             case ActionType::RESOLVE_BATTLE:
                 execute_battle(game_state, card_db);
                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                 }
                 break;
             case ActionType::BREAK_SHIELD:
                 resolve_break_shield(game_state, action, card_db);
                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                 }
                 break;
             case ActionType::DECLARE_REACTION:
                 resolve_reaction(game_state, action, card_db);
                 break;
             case ActionType::SELECT_OPTION:
                 // Resolve the option selection by queuing the actions of the selected option
                 if (!game_state.pending_effects.empty() && action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
                     auto& pe = game_state.pending_effects[action.slot_index];
                     if (pe.type == EffectType::SELECT_OPTION) {
                         int option_index = action.target_slot_index;
                         if (option_index >= 0 && option_index < (int)pe.options.size()) {
                             const auto& selected_actions = pe.options[option_index];

                             // Create a temporary effect def to hold these actions
                             EffectDef temp_effect;
                             temp_effect.actions = selected_actions;
                             temp_effect.trigger = TriggerType::NONE;

                             // Queue resolution of these actions
                             GenericCardSystem::resolve_effect_with_context(game_state, temp_effect, pe.source_instance_id, pe.execution_context, card_db);
                         }
                     }
                     // Remove the pending effect after selection
                     if (action.slot_index < (int)game_state.pending_effects.size()) {
                         game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
                     }
                 }
                 break;
             default:
                 break;
        }
    }

    // ... resolve functions (keep) ...
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
        if (action.target_player == 254) {
             auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
             if (it != player.hand.end()) {
                 CardInstance card = *it;
                 player.hand.erase(it);
                 game_state.stack_zone.push_back(card);
                 int units = action.target_slot_index;
                 int reduction_amount = 0;

                 if (card_db.count(card.card_id)) {
                     const auto& def = card_db.at(card.card_id);
                     for (const auto& reduction : def.cost_reductions) {
                         if (reduction.type == ReductionType::ACTIVE_PAYMENT) {
                             reduction_amount = CostPaymentSystem::execute_payment(game_state, player.id, reduction, units, card_db);
                             break; // Assume single active reduction type per card for now
                         }
                     }
                     // If no active reduction found but target_player=254, something is wrong or it's legacy hyper_energy?
                     // Legacy fallback:
                     if (reduction_amount == 0 && def.keywords.hyper_energy) {
                         // Fallback to hardcoded tapping if new system yields 0 (e.g. data mismatch)
                         int taps_done = 0;
                         for (auto& c : player.battle_zone) {
                             if (!c.is_tapped && !c.summoning_sickness) {
                                 c.is_tapped = true;
                                 taps_done++;
                                 if (taps_done >= units) break;
                             }
                         }
                         reduction_amount = taps_done * 2;
                     }

                     // Now pay remaining mana
                     // Apply passive reductions first (Standard Cost Logic)
                     int base_adjusted = ManaSystem::get_adjusted_cost(game_state, player, def);
                     // Apply active reduction
                     int final_cost = std::max(0, base_adjusted - reduction_amount);
                     // Enforce min cost from active reductions if any (Phase 4 requirement)
                     // CostPaymentSystem logic respects min_mana_cost during calculation, so final_cost should be safe to pay.

                     ManaSystem::auto_tap_mana(game_state, player, def, final_cost, card_db);
                 }

                 resolve_play_from_stack(game_state, card.instance_id, reduction_amount, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db);
             }
             return;
        }
        auto it = std::find_if(player.hand.begin(), player.hand.end(), [&](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        if (it != player.hand.end()) {
            CardInstance card = *it;
            player.hand.erase(it);
            card.is_tapped = false;
            card.summoning_sickness = true;
            if (action.target_instance_id != -1) {
                card.power_mod = action.target_instance_id;
            } else {
                card.power_mod = -1;
            }
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
        game_state.turn_stats.attacks_declared_this_turn++;
        GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_ATTACK, card.instance_id, card_db);

        if (game_state.current_phase == Phase::ATTACK) {
            game_state.current_phase = Phase::BLOCK;
        }
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
             GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_BLOCK, it->instance_id, card_db);
         }
         game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, action.source_instance_id, game_state.active_player_id);
    }

    void EffectResolver::resolve_use_shield_trigger(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        GenericCardSystem::resolve_trigger(game_state, TriggerType::S_TRIGGER, action.source_instance_id, card_db);
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

             // Check if target is explicitly set (e.g., from ActionGenerator)
             int attacker_id = action.target_instance_id;
             if (attacker_id == -1) {
                 // Fallback to current attack source if not set
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

    int EffectResolver::get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;
        power += creature.power_mod;
        power += PassiveEffectSystem::instance().get_power_buff(game_state, creature, card_db);
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
            Player& defender = game_state.get_non_active_player();
            int breaker_count = 1;
            Player& attacker_player = game_state.get_active_player();
            auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == attacker_id; });
            if (it != attacker_player.battle_zone.end()) {
                breaker_count = get_breaker_count(*it, card_db);
            }
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
                GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_DESTROY, defender_id, card_db);
            }
        }
        if (def_wins || draw) {
            auto it = std::find_if(p1.battle_zone.begin(), p1.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == attacker_id; });
            if (it != p1.battle_zone.end()) {
                p1.graveyard.push_back(*it);
                p1.battle_zone.erase(it);
                GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_DESTROY, attacker_id, card_db);
            }
        }
    }

    void EffectResolver::resolve_break_shield(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         PlayerID target_pid = action.target_player;
         if (target_pid == 255 || target_pid < 0) {
             target_pid = game_state.get_non_active_player().id;
         }
         Player& defender = game_state.players[target_pid];

         if (defender.shield_zone.empty()) {
             // If this break was intended for the opponent and they have no shields, it's a win.
             // But if it was self-break or friendly fire, it shouldn't trigger win?
             // Standard BREAK_SHIELD logic usually implies winning if empty.
             // For safety, only trigger win if target is opponent of active player.
             if (target_pid != game_state.active_player_id) {
                 game_state.winner = (game_state.active_player_id == 0) ? GameResult::P1_WIN : GameResult::P2_WIN;
             }
             return;
         }

         GenericCardSystem::resolve_trigger(game_state, TriggerType::AT_BREAK_SHIELD, action.source_instance_id, card_db);

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
         defender.shield_zone.erase(defender.shield_zone.begin() + shield_index);

         bool shield_burn = false;
         Player& attacker_player = game_state.get_active_player();
         auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(),
             [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });
         if (it != attacker_player.battle_zone.end()) {
             if (card_db.count(it->card_id)) {
                 if (card_db.at(it->card_id).keywords.shield_burn) {
                     shield_burn = true;
                 }
             }
         }
         if (shield_burn) {
             defender.graveyard.push_back(shield);
             GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_DESTROY, shield.instance_id, card_db);
         } else {
             bool is_trigger = false;
             if (card_db.count(shield.card_id)) {
                 const auto& def = card_db.at(shield.card_id);
                 // Updated for Conditional S-Trigger support
                 if (TargetUtils::has_keyword_simple(game_state, shield, def, "SHIELD_TRIGGER")) {
                     is_trigger = true;
                 }
             }
             if (is_trigger) {
                 defender.hand.push_back(shield);
                 game_state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield.instance_id, defender.id);
             } else {
                 defender.hand.push_back(shield);
             }
             // Reaction Window: ON_SHIELD_ADD (Strike Back)
             ReactionSystem::check_and_open_window(game_state, card_db, "ON_SHIELD_ADD", defender.id);
         }
    }

    // Updated with destination override
    void EffectResolver::resolve_play_from_stack(GameState& game_state, int stack_instance_id, int cost_reduction, SpawnSource spawn_source, PlayerID controller, const std::map<CardID, CardDefinition>& card_db, int evo_source_id, int dest_override) {
        auto& stack = game_state.stack_zone;
        auto it = std::find_if(stack.begin(), stack.end(), [&](const CardInstance& c){ return c.instance_id == stack_instance_id; });
        CardInstance card;
        bool found = false;
        if (it != stack.end()) {
            card = *it;
            if (evo_source_id == -1 && card.power_mod != 0) {
                 evo_source_id = card.power_mod;
            }
            card.power_mod = 0;
            stack.erase(it);
            found = true;
        } else {
            // Check buffers (P0)
            auto& buf0 = game_state.players[0].effect_buffer;
            auto bit0 = std::find_if(buf0.begin(), buf0.end(), [&](const CardInstance& c){ return c.instance_id == stack_instance_id; });
            if (bit0 != buf0.end()) {
                card = *bit0;
                buf0.erase(bit0);
                found = true;
            } else {
                 // Check buffers (P1)
                auto& buf1 = game_state.players[1].effect_buffer;
                auto bit1 = std::find_if(buf1.begin(), buf1.end(), [&](const CardInstance& c){ return c.instance_id == stack_instance_id; });
                if (bit1 != buf1.end()) {
                    card = *bit1;
                    buf1.erase(bit1);
                    found = true;
                }
            }
        }
        if (!found) return;
        Player& player = game_state.players[controller];
        const CardDefinition* def = nullptr;
        if (card_db.count(card.card_id)) def = &card_db.at(card.card_id);
        if (def && def->type == CardType::SPELL) {

            // Check destination override
            if (dest_override == 1) { // Deck Bottom
                player.deck.insert(player.deck.begin(), card); // Bottom is begin? Or end?
                // Standard: Back is TOP (Draw pops back). Front is BOTTOM.
                // Verify ZoneUtils if available, but vector convention in this project:
                // pop_back = draw. push_back = put on top.
                // begin = bottom.
                // So insert at begin.
            } else {
                player.graveyard.push_back(card);
            }

            GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, card.instance_id, card_db);
            game_state.turn_stats.spells_cast_this_turn++;
        } else {
            // Creatures always go to Battle Zone (unless other override logic exists?)
            // Assuming Creature override (e.g. God Link?) is not requested here.

            card.summoning_sickness = true;
            if (def && def->keywords.speed_attacker) card.summoning_sickness = false;
            if (def && def->keywords.evolution) card.summoning_sickness = false;
            // Ensure freshly played creatures enter untapped even though cost payment marks stack card as tapped.
            card.is_tapped = false;
            card.turn_played = game_state.turn_number;
            if (evo_source_id != -1) {
                auto s_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == evo_source_id; });
                if (s_it != player.battle_zone.end()) {
                    CardInstance source = *s_it;
                    player.battle_zone.erase(s_it);
                    card.underlying_cards.push_back(source);
                }
            }
            player.battle_zone.push_back(card);
            GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, card.instance_id, card_db);
            game_state.turn_stats.creatures_played_this_turn++;
        }
        game_state.on_card_play(card.card_id, game_state.turn_number, spawn_source != SpawnSource::HAND_SUMMON, cost_reduction, controller);
    }

    void EffectResolver::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context) {
        GenericCardSystem::resolve_effect_with_context(game_state, effect, source_instance_id, execution_context);
    }

    void EffectResolver::resolve_reaction(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& controller = game_state.players[action.target_player];
         CardInstance card = remove_from_hand(controller, action.source_instance_id);
         game_state.stack_zone.push_back(card);
         resolve_play_from_stack(game_state, card.instance_id, 999, SpawnSource::EFFECT_SUMMON, controller.id, card_db);
    }

}
