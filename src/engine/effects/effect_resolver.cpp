#include "effect_resolver.hpp"
#include "generated_effects.hpp"
#include "../card_system/generic_card_system.hpp"
#include "../card_system/card_registry.hpp"
#include "../card_system/target_utils.hpp"
#include "../mana/mana_system.hpp"
#include "../flow/reaction_system.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    static CardInstance remove_from_hand(Player& player, int instance_id) {
        auto it = std::find_if(player.hand.begin(), player.hand.end(), 
            [instance_id](const CardInstance& c) { return c.instance_id == instance_id; });
        
        if (it != player.hand.end()) {
            CardInstance c = *it;
            player.hand.erase(it);
            return c;
        }
        throw std::runtime_error("Card not found in hand");
    }

    void EffectResolver::resolve_action(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        switch (action.type) {
            case ActionType::PASS:
                if (game_state.current_phase == Phase::BLOCK) {
                    bool is_blocked = game_state.current_attack.is_blocked;
                    int target_id = game_state.current_attack.target_instance_id;

                    if (is_blocked || target_id != -1) {
                         game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                    } else {
                         int attacker_id = game_state.current_attack.source_instance_id;
                         Player& active = game_state.get_active_player();
                         auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
                            [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });

                         int break_count = 1;
                         if (attacker_it != active.battle_zone.end()) {
                             break_count = get_breaker_count(*attacker_it, card_db);
                         }

                         for (int i=0; i<break_count; ++i) {
                             game_state.pending_effects.emplace_back(EffectType::BREAK_SHIELD, attacker_id, game_state.active_player_id);
                         }
                    }
                }
                break;
            case ActionType::MANA_CHARGE:
                resolve_mana_charge(game_state, action);
                break;
            case ActionType::MOVE_CARD:
                if (game_state.current_phase == Phase::MANA) {
                    resolve_mana_charge(game_state, action);
                }
                break;
            case ActionType::DECLARE_PLAY:
            case ActionType::PAY_COST:
            case ActionType::RESOLVE_PLAY:
            case ActionType::PLAY_CARD:
            case ActionType::PLAY_CARD_INTERNAL:
                resolve_play_card(game_state, action, card_db);
                break;
            case ActionType::ATTACK_PLAYER:
            case ActionType::ATTACK_CREATURE:
                resolve_attack(game_state, action, card_db);
                break;
            case ActionType::RESOLVE_EFFECT:
                resolve_pending_effect(game_state, action, card_db);
                break;
            case ActionType::USE_SHIELD_TRIGGER:
                resolve_use_shield_trigger(game_state, action, card_db);
                break;
            case ActionType::BLOCK:
                resolve_block(game_state, action, card_db);
                break;
            case ActionType::SELECT_TARGET:
                resolve_select_target(game_state, action);
                break;
            case ActionType::USE_ABILITY:
                resolve_use_ability(game_state, action, card_db);
                break;
            case ActionType::RESOLVE_BATTLE:
                resolve_battle_outcome(game_state, card_db);
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
        game_state.update_loop_check();
    }

    void EffectResolver::resolve_select_target(GameState& game_state, const Action& action) {
        if (game_state.pending_effects.empty()) return;
        
        int index = action.slot_index;
        if (index >= 0 && index < static_cast<int>(game_state.pending_effects.size())) {
            game_state.pending_effects[index].target_instance_ids.push_back(action.target_instance_id);
        }
    }

    void EffectResolver::resolve_use_ability(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        if (game_state.pending_effects.empty()) return;

        int target_pe_index = -1;
        for (int i = 0; i < (int)game_state.pending_effects.size(); ++i) {
             if (game_state.pending_effects[i].type == EffectType::ON_ATTACK_FROM_HAND) {
                 target_pe_index = i;
                 break;
             }
        }

        if (target_pe_index == -1) return;

        Player& player = game_state.get_active_player();

        int attacker_id = game_state.current_attack.source_instance_id;
        auto attacker_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(),
            [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });

        if (attacker_it != player.battle_zone.end()) {
            CardInstance returned_creature = *attacker_it;
            returned_creature.is_tapped = false;
            player.battle_zone.erase(attacker_it);
            player.hand.push_back(returned_creature);
        } else {
            game_state.pending_effects.erase(game_state.pending_effects.begin() + target_pe_index);
            return;
        }

        try {
            CardInstance new_creature = remove_from_hand(player, action.source_instance_id);
            new_creature.is_tapped = true;
            new_creature.summoning_sickness = false;

            player.battle_zone.push_back(new_creature);
            game_state.current_attack.source_instance_id = new_creature.instance_id;

            if (card_db.count(new_creature.card_id)) {
                const auto& def = card_db.at(new_creature.card_id);
                if (def.keywords.cip) {
                    dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, new_creature.instance_id);
                }
            }
        } catch (...) {}

        if (target_pe_index < (int)game_state.pending_effects.size()) {
             game_state.pending_effects.erase(game_state.pending_effects.begin() + target_pe_index);
        }
        game_state.current_phase = Phase::BLOCK;
    }

    void EffectResolver::resolve_block(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        game_state.current_attack.is_blocked = true;
        game_state.current_attack.blocker_instance_id = action.source_instance_id;
        
        Player& defender = game_state.get_non_active_player();
        auto it = std::find_if(defender.battle_zone.begin(), defender.battle_zone.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        if (it != defender.battle_zone.end()) {
            it->is_tapped = true;
        }

        bool reaction_opened = ReactionSystem::check_and_open_window(
             game_state, card_db, "ON_BLOCK_OR_ATTACK", defender.id
        );

        bool reaction_opened_attacker = ReactionSystem::check_and_open_window(
             game_state, card_db, "ON_BLOCK_OR_ATTACK", game_state.active_player_id
        );

        game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
    }

    int EffectResolver::get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        if (card_db.find(creature.card_id) == card_db.end()) return 0;
        const auto& def = card_db.at(creature.card_id);
        int power = def.power;

        if (game_state.current_phase == Phase::ATTACK || game_state.current_phase == Phase::BLOCK) {
            if (game_state.current_attack.source_instance_id == creature.instance_id) {
                if (def.keywords.power_attacker) {
                    power += def.power_attacker_bonus;
                }
            }
        }
        return power;
    }

    int EffectResolver::get_breaker_count(const CardInstance& creature, const std::map<CardID, CardDefinition>& card_db) {
        if (card_db.find(creature.card_id) == card_db.end()) return 1;
        const auto& def = card_db.at(creature.card_id);
        
        if (def.keywords.triple_breaker) return 3;
        if (def.keywords.double_breaker) return 2;
        return 1;
    }

    void EffectResolver::resolve_battle_outcome(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
            if (game_state.pending_effects[i].type == EffectType::RESOLVE_BATTLE) {
                game_state.pending_effects.erase(game_state.pending_effects.begin() + i);
                break;
            }
        }

        Player& active = game_state.get_active_player();
        Player& opponent = game_state.get_non_active_player();
        
        int attacker_id = game_state.current_attack.source_instance_id;
        int target_id = game_state.current_attack.target_instance_id;
        bool is_blocked = game_state.current_attack.is_blocked;
        int blocker_id = game_state.current_attack.blocker_instance_id;

        auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
            [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });
        
        if (attacker_it == active.battle_zone.end()) return;
        CardInstance attacker = *attacker_it;

        if (is_blocked) {
            auto blocker_it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                [blocker_id](const CardInstance& c) { return c.instance_id == blocker_id; });
            
            if (blocker_it == opponent.battle_zone.end()) return;
            CardInstance blocker = *blocker_it;

            int attacker_power = get_creature_power(attacker, game_state, card_db);
            int blocker_power = get_creature_power(blocker, game_state, card_db);
            
            bool destroy_attacker = (attacker_power <= blocker_power);
            bool destroy_blocker = (blocker_power <= attacker_power);

            if (card_db.at(attacker.card_id).keywords.slayer) destroy_blocker = true;
            if (card_db.at(blocker.card_id).keywords.slayer) destroy_attacker = true;

            if (destroy_attacker) {
                auto it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
                    [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });
                if (it != active.battle_zone.end()) {
                    active.graveyard.push_back(*it);
                    active.battle_zone.erase(it);
                }
            }
            if (destroy_blocker) {
                auto it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                    [blocker_id](const CardInstance& c) { return c.instance_id == blocker_id; });
                if (it != opponent.battle_zone.end()) {
                    opponent.graveyard.push_back(*it);
                    opponent.battle_zone.erase(it);
                }
            }
        } else if (target_id != -1) {
             auto defender_it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                [target_id](const CardInstance& c) { return c.instance_id == target_id; });

            if (defender_it != opponent.battle_zone.end()) {
                CardInstance defender = *defender_it;
                int attacker_power = get_creature_power(attacker, game_state, card_db);
                int defender_power = get_creature_power(defender, game_state, card_db);

                bool destroy_attacker = (attacker_power <= defender_power);
                bool destroy_defender = (defender_power <= attacker_power);

                if (card_db.at(attacker.card_id).keywords.slayer) destroy_defender = true;
                if (card_db.at(defender.card_id).keywords.slayer) destroy_attacker = true;

                if (destroy_attacker) {
                    auto it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
                        [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });
                    if (it != active.battle_zone.end()) {
                        active.graveyard.push_back(*it);
                        active.battle_zone.erase(it);
                    }
                }
                if (destroy_defender) {
                    auto it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                        [target_id](const CardInstance& c) { return c.instance_id == target_id; });
                    if (it != opponent.battle_zone.end()) {
                        opponent.graveyard.push_back(*it);
                        opponent.battle_zone.erase(it);
                    }
                }
            }
        }

        game_state.current_phase = Phase::ATTACK;
    }

    void EffectResolver::resolve_break_shield(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         if (action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
             if (game_state.pending_effects[action.slot_index].type == EffectType::BREAK_SHIELD) {
                 game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
             } else {
                 for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
                    if (game_state.pending_effects[i].type == EffectType::BREAK_SHIELD) {
                        game_state.pending_effects.erase(game_state.pending_effects.begin() + i);
                        break;
                    }
                }
             }
         } else {
             for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
                if (game_state.pending_effects[i].type == EffectType::BREAK_SHIELD) {
                    game_state.pending_effects.erase(game_state.pending_effects.begin() + i);
                    break;
                }
            }
         }

         Player& active = game_state.get_active_player();
         Player& opponent = game_state.get_non_active_player();

         if (!opponent.shield_zone.empty()) {
            CardInstance shield = opponent.shield_zone.back();
            opponent.shield_zone.pop_back();

            opponent.hand.push_back(shield);

            if (card_db.count(shield.card_id)) {
                const auto& def = card_db.at(shield.card_id);
                if (def.keywords.shield_trigger) {
                    game_state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield.instance_id, opponent.id);
                }
            }
        } else {
             if (active.id == 0) game_state.winner = GameResult::P1_WIN;
             else game_state.winner = GameResult::P2_WIN;
        }

        bool more_breaks = false;
        for(const auto& pe : game_state.pending_effects) {
            if (pe.type == EffectType::BREAK_SHIELD) {
                more_breaks = true;
                break;
            }
        }
        if (!more_breaks && game_state.winner == GameResult::NONE) {
             game_state.current_phase = Phase::ATTACK;
        }
    }

    void EffectResolver::execute_battle(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
         bool is_blocked = game_state.current_attack.is_blocked;
         int target_id = game_state.current_attack.target_instance_id;

         if (is_blocked || target_id != -1) {
             resolve_battle_outcome(game_state, card_db);
         } else {
             int attacker_id = game_state.current_attack.source_instance_id;
             Player& active = game_state.get_active_player();
             auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
                [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });

             int break_count = 1;
             if (attacker_it != active.battle_zone.end()) {
                 break_count = get_breaker_count(*attacker_it, card_db);
             }

             for(int i=0; i<break_count; ++i) {
                 Action a;
                 resolve_break_shield(game_state, a, card_db);
                 if (game_state.winner != GameResult::NONE) break;
             }
         }
    }

    void EffectResolver::resolve_pending_effect(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        if (game_state.pending_effects.empty()) return;
        
        int index = action.slot_index;
        if (index < 0 || index >= static_cast<int>(game_state.pending_effects.size())) {
            index = game_state.pending_effects.size() - 1;
        }

        PendingEffect effect = game_state.pending_effects[index];

        if (effect.type == EffectType::RESOLVE_BATTLE) {
             game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
             resolve_battle_outcome(game_state, card_db);
             return;
        }
        if (effect.type == EffectType::BREAK_SHIELD) {
             game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
             Action dummy;
             resolve_break_shield(game_state, dummy, card_db);
             return;
        }

        if (effect.effect_def.has_value()) {
            if (effect.num_targets_needed > static_cast<int>(effect.target_instance_ids.size())) {
                return;
            }
            game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
        } else {
            game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
        }

        Player& controller = game_state.players[effect.controller];
        
        CardID card_id = 0;
        auto find_in_vec = [&](const std::vector<CardInstance>& vec)->CardID{
            for (const auto& c : vec) if (c.instance_id == effect.source_instance_id) return c.card_id;
            return (CardID)0;
        };

        card_id = find_in_vec(controller.battle_zone);
        if (card_id == 0) card_id = find_in_vec(controller.hand);
        if (card_id == 0) card_id = find_in_vec(controller.mana_zone);
        if (card_id == 0) card_id = find_in_vec(controller.graveyard);
        if (card_id == 0) card_id = find_in_vec(controller.shield_zone);
        if (card_id == 0) card_id = find_in_vec(game_state.stack_zone);

        if (card_id == 0) return;

        if (effect.effect_def.has_value()) {
            dm::engine::GenericCardSystem::resolve_effect_with_targets(game_state, effect.effect_def.value(), effect.target_instance_ids, effect.source_instance_id, card_db);
            return;
        }

        const dm::core::CardData* data = dm::engine::CardRegistry::get_card_data(card_id);
        if (data) {
            dm::core::TriggerType trig = dm::core::TriggerType::NONE;
            switch (effect.type) {
                case EffectType::CIP: trig = dm::core::TriggerType::ON_PLAY; break;
                case EffectType::AT_ATTACK: trig = dm::core::TriggerType::ON_ATTACK; break;
                case EffectType::DESTRUCTION: trig = dm::core::TriggerType::ON_DESTROY; break;
                case EffectType::SHIELD_TRIGGER: trig = dm::core::TriggerType::S_TRIGGER; break;
                case EffectType::AT_START_OF_TURN: trig = dm::core::TriggerType::TURN_START; break;
                case EffectType::AT_END_OF_TURN: trig = dm::core::TriggerType::PASSIVE_CONST; break;
                default: trig = dm::core::TriggerType::NONE; break;
            }

            if (trig != dm::core::TriggerType::NONE) {
                    bool created_selection_pe = false;
                    for (const auto &ef : data->effects) {
                        for (const auto &act : ef.actions) {
                            if (act.scope == dm::core::TargetScope::TARGET_SELECT) {
                                PendingEffect sel(effect.type, effect.source_instance_id, effect.controller);
                                sel.num_targets_needed = act.filter.count.has_value() ? act.filter.count.value() : 1;
                                EffectDef ed;
                                ed.trigger = TriggerType::NONE;
                                ed.condition = ConditionDef{"NONE", 0, ""};
                                ed.actions = { act };
                                sel.effect_def = ed;
                                game_state.pending_effects.push_back(sel);
                                created_selection_pe = true;
                                break;
                            }
                        }
                        if (created_selection_pe) break;
                    }
                    if (created_selection_pe) return;

                    dm::engine::GenericCardSystem::resolve_trigger(game_state, trig, effect.source_instance_id);
                    return;
            }
        }

        GeneratedEffects::resolve(game_state, effect, card_id);
    }

    void EffectResolver::resolve_use_shield_trigger(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        if (!game_state.pending_effects.empty()) {
             int index = action.slot_index;
             if (index >= 0 && index < static_cast<int>(game_state.pending_effects.size())) {
                 if (game_state.pending_effects[index].type == EffectType::SHIELD_TRIGGER) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
                 }
             }
        }
        game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, action.source_instance_id, action.target_player);
    }

    void EffectResolver::resolve_mana_charge(GameState& game_state, const Action& action) {
        Player& player = game_state.get_active_player();
        try {
            CardInstance card = remove_from_hand(player, action.source_instance_id);
            card.is_tapped = false; 
            player.mana_zone.push_back(card);
        } catch (...) {}
    }

    void EffectResolver::resolve_play_card(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& player = game_state.get_active_player();

        if (action.type == ActionType::PLAY_CARD_INTERNAL) {
            int index = action.slot_index;
            if (index < 0 || index >= static_cast<int>(game_state.pending_effects.size())) {
                return;
            }

            const auto& pe = game_state.pending_effects[index];
            if (pe.type != EffectType::INTERNAL_PLAY && pe.type != EffectType::META_COUNTER) {
                return;
            }

            int card_instance_id = action.source_instance_id;

            CardInstance* card_ptr = nullptr;
            std::vector<CardInstance>* source_zone = nullptr;

            auto find_and_set = [&](std::vector<CardInstance>& zone) {
                auto it = std::find_if(zone.begin(), zone.end(),
                    [card_instance_id](const CardInstance& c) { return c.instance_id == card_instance_id; });
                if (it != zone.end()) {
                    card_ptr = &(*it);
                    source_zone = &zone;
                    return true;
                }
                return false;
            };

            if (!find_and_set(game_state.effect_buffer)) {
                Player& controller = game_state.players[pe.controller];
                if (!find_and_set(controller.hand)) {
                     if (!find_and_set(controller.graveyard)) {
                     }
                }
            }

            if (card_ptr && source_zone) {
                CardInstance card = *card_ptr;
                source_zone->erase(std::remove_if(source_zone->begin(), source_zone->end(),
                    [card_instance_id](const CardInstance& c) { return c.instance_id == card_instance_id; }),
                    source_zone->end());

                game_state.stack_zone.push_back(card);
                game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
                resolve_play_from_stack(game_state, card.instance_id, 999, action.spawn_source, pe.controller, card_db);
            }
            return;
        }

        if (action.type == ActionType::DECLARE_PLAY) {
             auto it = std::find_if(player.hand.begin(), player.hand.end(),
                [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });

             if (it == player.hand.end()) return;

             CardInstance card = *it;
             player.hand.erase(it);

             if (action.target_player == 254) {
                 int required_taps = action.target_slot_index;

                 PendingEffect pending(EffectType::NONE, card.instance_id, player.id);
                 pending.resolve_type = ResolveType::TARGET_SELECT;
                 FilterDef filter;
                 filter.zones = {"BATTLE_ZONE"};
                 filter.owner = "SELF";
                 filter.is_tapped = false;
                 pending.filter = filter;
                 pending.num_targets_needed = required_taps;

                 EffectDef continuation;
                 ActionDef mark_paid;
                 mark_paid.type = EffectActionType::COST_REFERENCE;
                 mark_paid.str_val = "MARK_STACK_PAID";
                 continuation.actions.push_back(mark_paid);
                 pending.effect_def = continuation;

                 game_state.pending_effects.push_back(pending);
             }

             game_state.stack_zone.push_back(card);
             return;
        }

        if (action.type == ActionType::PAY_COST) {
            if (game_state.stack_zone.empty()) return;
            CardInstance& card = game_state.stack_zone.back();
            if (card.instance_id != action.source_instance_id) return;

            const CardDefinition& def = card_db.at(card.card_id);
            if (ManaSystem::auto_tap_mana(game_state, player, def, card_db)) {
                card.is_tapped = true;
                game_state.on_card_play(card.card_id, game_state.turn_number, false, 0, player.id);
            } else {
                player.hand.push_back(card);
                game_state.stack_zone.pop_back();
            }
            return;
        }

        if (action.type == ActionType::RESOLVE_PLAY) {
            if (game_state.stack_zone.empty()) return;
            CardInstance card = game_state.stack_zone.back();
            game_state.stack_zone.pop_back();

            const CardDefinition& def = card_db.at(card.card_id);

            if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
                card.summoning_sickness = true;
                if (def.keywords.speed_attacker) card.summoning_sickness = false;
                if (def.keywords.evolution) card.summoning_sickness = false;

                card.is_tapped = false;
                card.turn_played = game_state.turn_number;

                player.battle_zone.push_back(card);
                if (def.keywords.cip) {
                    dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
                }
            } else if (def.type == CardType::SPELL) {
                card.is_tapped = false;
                player.graveyard.push_back(card);
                dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
            }
            return;
        }

        if (action.type == ActionType::PLAY_CARD) {
            auto it = std::find_if(player.hand.begin(), player.hand.end(),
                [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
            if (it == player.hand.end()) return;
            
            if (action.target_player == 254) {
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
                 resolve_play_from_stack(game_state, card.instance_id, taps_done * 2, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db);
                 return;
            }

            CardInstance card = *it;
            player.hand.erase(it);
            game_state.stack_zone.push_back(card);

            const CardDefinition& def = card_db.at(card.card_id);
            if (ManaSystem::auto_tap_mana(game_state, player, def, card_db)) {
                 game_state.stack_zone.back().is_tapped = true;
                 game_state.on_card_play(card.card_id, game_state.turn_number, false, 0, player.id);

                 Action res_act = action;
                 res_act.type = ActionType::RESOLVE_PLAY;
                 resolve_play_card(game_state, res_act, card_db);
            } else {
                 game_state.stack_zone.pop_back();
                 player.hand.push_back(card);
            }
        }
    }

    void EffectResolver::resolve_play_from_stack(GameState& game_state, int stack_instance_id, int cost_reduction, SpawnSource spawn_source, PlayerID controller, const std::map<CardID, CardDefinition>& card_db) {
        auto s_it = std::find_if(game_state.stack_zone.begin(), game_state.stack_zone.end(),
             [stack_instance_id](const CardInstance& c) { return c.instance_id == stack_instance_id; });

        if (s_it == game_state.stack_zone.end()) {
            return;
        }
        CardInstance card = *s_it;
        Player& player = game_state.players[controller];
        const CardDefinition& def = card_db.at(card.card_id);

        bool payment_success = false;

        if (cost_reduction >= 999) {
            payment_success = true;
        } else {
            CostModifier mod;
            mod.reduction_amount = cost_reduction;
            mod.condition_filter.types = {};
            mod.turns_remaining = 1;
            mod.controller = player.id;
            mod.source_instance_id = -1;

            game_state.active_modifiers.push_back(mod);
            payment_success = ManaSystem::auto_tap_mana(game_state, player, def, card_db);
            game_state.active_modifiers.pop_back();
        }

        if (!payment_success) {
            game_state.stack_zone.erase(s_it);
            player.hand.push_back(card);
            return;
        }

        game_state.on_card_play(card.card_id, game_state.turn_number, false, cost_reduction, player.id);
        game_state.stack_zone.erase(s_it);

        if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
            card.summoning_sickness = true;
            if (def.keywords.speed_attacker) card.summoning_sickness = false;
            if (def.keywords.evolution) card.summoning_sickness = false;

            card.is_tapped = false;
            card.turn_played = game_state.turn_number;

            player.battle_zone.push_back(card);

            game_state.turn_stats.creatures_played_this_turn++;

            if (def.keywords.cip) {
                dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
            }
        } else if (def.type == CardType::SPELL) {
            card.is_tapped = false;
            player.graveyard.push_back(card);

            game_state.turn_stats.spells_cast_this_turn++;

            dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
        }
    }

    void EffectResolver::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context) {
        if (!GenericCardSystem::check_condition(game_state, effect.condition, source_instance_id)) return;

        for (size_t i = 0; i < effect.actions.size(); ++i) {
            ActionDef action = effect.actions[i];

            if (!action.input_value_key.empty()) {
                if (execution_context.count(action.input_value_key)) {
                    action.value1 = execution_context[action.input_value_key];
                    action.value = std::to_string(action.value1);
                }
            }

            if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                EffectDef continuation;
                continuation.trigger = TriggerType::NONE;
                continuation.condition = ConditionDef{"NONE", 0, ""};
                for (size_t j = i; j < effect.actions.size(); ++j) {
                    continuation.actions.push_back(effect.actions[j]);
                }
                GenericCardSystem::select_targets(game_state, action, source_instance_id, continuation);
                return;
            }

            if (action.type == EffectActionType::COUNT_CARDS) {
                int count = 0;
                Player& active = game_state.get_active_player();

                auto count_in_vec = [&](const std::vector<CardInstance>& vec, PlayerID owner_id) {
                    for (const auto& c : vec) {
                         const CardData* cd = CardRegistry::get_card_data(c.card_id);
                         if (cd) {
                             if (TargetUtils::is_valid_target(c, *cd, action.filter, game_state, active.id, owner_id)) {
                                 count++;
                             }
                         }
                    }
                };

                std::vector<Player*> players_to_check;
                std::string own_req = action.filter.owner.value_or("SELF");

                if (own_req == "SELF") {
                    players_to_check.push_back(&active);
                } else if (own_req == "OPPONENT") {
                    players_to_check.push_back(&game_state.get_non_active_player());
                } else if (own_req == "BOTH") {
                    players_to_check.push_back(&active);
                    players_to_check.push_back(&game_state.get_non_active_player());
                } else {
                    players_to_check.push_back(&active);
                }

                if (!action.filter.zones.empty()) {
                    for (Player* p : players_to_check) {
                        for (const auto& zone : action.filter.zones) {
                            if (zone == "MANA_ZONE") count_in_vec(p->mana_zone, p->id);
                            else if (zone == "BATTLE_ZONE") count_in_vec(p->battle_zone, p->id);
                            else if (zone == "HAND") count_in_vec(p->hand, p->id);
                            else if (zone == "GRAVEYARD") count_in_vec(p->graveyard, p->id);
                            else if (zone == "SHIELD_ZONE") count_in_vec(p->shield_zone, p->id);
                        }
                    }
                }

                if (!action.output_value_key.empty()) {
                    execution_context[action.output_value_key] = count;
                }

            } else if (action.type == EffectActionType::GET_GAME_STAT) {
                int val = 0;
                if (action.str_val == "cards_drawn_this_turn") val = game_state.turn_stats.cards_drawn_this_turn;
                else if (action.str_val == "cards_discarded_this_turn") val = game_state.turn_stats.cards_discarded_this_turn;
                else if (action.str_val == "creatures_played_this_turn") val = game_state.turn_stats.creatures_played_this_turn;
                else if (action.str_val == "spells_cast_this_turn") val = game_state.turn_stats.spells_cast_this_turn;

                if (!action.output_value_key.empty()) {
                    execution_context[action.output_value_key] = val;
                }

            } else if (action.type == EffectActionType::APPLY_MODIFIER) {
                CostModifier mod;
                mod.reduction_amount = action.value1;
                mod.condition_filter = action.filter;
                mod.turns_remaining = action.value2 > 0 ? action.value2 : 1;
                mod.controller = game_state.active_player_id;
                mod.source_instance_id = source_instance_id;
                game_state.active_modifiers.push_back(mod);

            } else if (action.type == EffectActionType::REVEAL_CARDS) {
            } else if (action.type == EffectActionType::REGISTER_DELAYED_EFFECT) {
            } else if (action.type == EffectActionType::RESET_INSTANCE) {
            } else {
                GenericCardSystem::resolve_action(game_state, action, source_instance_id);
            }
        }
    }

    void EffectResolver::resolve_reaction(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         Player& controller = game_state.players[action.target_player];
         CardInstance card = remove_from_hand(controller, action.source_instance_id);
         game_state.stack_zone.push_back(card);
         resolve_play_from_stack(game_state, card.instance_id, 999, SpawnSource::HAND_SUMMON, controller.id, card_db);
    }
}
