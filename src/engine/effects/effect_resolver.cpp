#include "effect_resolver.hpp"
#include "generated_effects.hpp"
#include "../card_system/generic_card_system.hpp"
#include "../card_system/card_registry.hpp"
#include "../card_system/target_utils.hpp"
#include "../mana/mana_system.hpp"
#include "../flow/reaction_system.hpp" // Added
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    // Helper to find and remove card from hand
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
                    // Pass in Block Phase means "No Block" -> Transition to battle resolution
                    // Instead of immediate execution, we queue the resolution action logic.
                    // But effectively, we are ending the BLOCK phase.

                    // Logic check: Blocked or Unblocked?
                    // If is_blocked, it's a battle between creatures.
                    // If not blocked, check target.
                    //    If target is creature -> Battle between creatures.
                    //    If target is player -> Shield Break.

                    // We use PendingEffects to drive the ActionGenerator.

                    bool is_blocked = game_state.current_attack.is_blocked;
                    int target_id = game_state.current_attack.target_instance_id;

                    if (is_blocked || target_id != -1) {
                         // Creature Battle
                         game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                    } else {
                         // Player Attack (Shield Break)
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

                    // Phase stays BLOCK or moves to ATTACK?
                    // Usually we stay in a resolution phase until queue is empty, then go back to ATTACK.
                    // But here we rely on ActionGenerator to pick up PendingEffects.
                }
                break;
            case ActionType::MANA_CHARGE:
                resolve_mana_charge(game_state, action);
                break;
            case ActionType::MOVE_CARD:
                // For now, in Phase::MANA, this implies MANA_CHARGE
                // In future, we decode destination from action.
                if (game_state.current_phase == Phase::MANA) {
                    resolve_mana_charge(game_state, action);
                } else {
                    // Generic handler (stub for now, or implement if needed)
                    // For now, let's assume it might be used for other moves later.
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

        // Update loop check after action resolution
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

        // Execute Revolution Change Swap
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

        // Ninja Strike / Strike Back check on Block
        // "Whenever ... blocks or is blocked"
        // Active Player (Attacker) might use Ninja Strike because their creature "is blocked"?
        // Or Defender because they "blocked"?
        // Ninja Strike text: "Whenever ... blocks OR IS BLOCKED".

        // Check Defender's Reactions (e.g. they blocked)
        bool reaction_opened = ReactionSystem::check_and_open_window(
             game_state, card_db, "ON_BLOCK_OR_ATTACK", defender.id
        );

        // Check Attacker's Reactions (e.g. they were blocked)
        bool reaction_opened_attacker = ReactionSystem::check_and_open_window(
             game_state, card_db, "ON_BLOCK_OR_ATTACK", game_state.active_player_id
        );

        // Instead of executing battle immediately, we queue the battle resolution.
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

    // Renamed and Refactored from execute_battle
    void EffectResolver::resolve_battle_outcome(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        // Clear the pending effect if this was triggered by one
        // We look for the first RESOLVE_BATTLE effect
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
        
        if (attacker_it == active.battle_zone.end()) {
             // Attacker gone? Remove pending effects associated?
             // Actually, if attacker is gone, battle fizzles.
             return;
        }
        CardInstance attacker = *attacker_it;

        // Resolve Creature Battle (Blocked or Target Creature)
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
             // Attack Creature (Unblocked)
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

        // Return to ATTACK phase (end of battle)
        game_state.current_phase = Phase::ATTACK;
    }

    void EffectResolver::resolve_break_shield(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         // Clear the pending effect if this was triggered by one
         // If action.slot_index is provided and valid, use it. Otherwise search for first BREAK_SHIELD.
         // ActionGenerator sets slot_index.
         if (action.slot_index >= 0 && action.slot_index < (int)game_state.pending_effects.size()) {
             if (game_state.pending_effects[action.slot_index].type == EffectType::BREAK_SHIELD) {
                 game_state.pending_effects.erase(game_state.pending_effects.begin() + action.slot_index);
             } else {
                 // Fallback search
                 for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
                    if (game_state.pending_effects[i].type == EffectType::BREAK_SHIELD) {
                        game_state.pending_effects.erase(game_state.pending_effects.begin() + i);
                        break;
                    }
                }
             }
         } else {
             // Fallback search
             for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
                if (game_state.pending_effects[i].type == EffectType::BREAK_SHIELD) {
                    game_state.pending_effects.erase(game_state.pending_effects.begin() + i);
                    break;
                }
            }
         }

         // Single Shield Break
         Player& active = game_state.get_active_player();
         Player& opponent = game_state.get_non_active_player();

         if (!opponent.shield_zone.empty()) {
            // Default to top shield if no target specified (legacy support)
            // Or use action.target_slot_index / action.target_instance_id

            // For now, let's just pop back as per old logic for consistency,
            // unless we want to support specific targeting.
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
             // Direct Attack Success
             if (active.id == 0) game_state.winner = GameResult::P1_WIN;
             else game_state.winner = GameResult::P2_WIN;
        }

        // If no more BREAK_SHIELD pending, return to ATTACK phase
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

    // Deprecated / Forwarder
    void EffectResolver::execute_battle(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
         // This should generally not be called anymore if we rely on queueing,
         // but if existing code calls it, we can forward to the new logic.
         // However, execute_battle did BOTH battle and break.

         // Let's implement it by inspecting state and calling the new helpers immediately.
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

             // Loop breaks immediately
             for(int i=0; i<break_count; ++i) {
                 // Fake an action
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

        // Handle Special Pending Effects that map to Actions
        // (Though ideally ActionGenerator should generate specific Actions for these)
        if (effect.type == EffectType::RESOLVE_BATTLE) {
             game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
             resolve_battle_outcome(game_state, card_db);
             return;
        }
        if (effect.type == EffectType::BREAK_SHIELD) {
             game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
             // Create a dummy action or use the passed one?
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
        // Queue INTERNAL_PLAY pending effect. The card remains in hand (where resolve_break_shield put it).
        // The ActionGenerator will then generate PLAY_CARD_INTERNAL, which handles the actual move to stack/resolution.
        
        // Remove the SHIELD_TRIGGER pending effect first
        if (!game_state.pending_effects.empty()) {
             int index = action.slot_index;
             if (index >= 0 && index < static_cast<int>(game_state.pending_effects.size())) {
                 if (game_state.pending_effects[index].type == EffectType::SHIELD_TRIGGER) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
                 }
             }
        }

        // Queue Internal Play
        // Source is HAND_SUMMON because S-Trigger is cast from hand (after being added to hand from shield zone).
        // Wait, Internal Play defaults to EFFECT_SUMMON in ActionGenerator for INTERNAL_PLAY type.
        // We might want to distinguish.
        // However, PLAY_CARD_INTERNAL logic uses resolve_play_from_stack which takes SpawnSource from Action.
        // ActionGenerator sets SpawnSource::EFFECT_SUMMON for INTERNAL_PLAY.
        // If we want HAND_SUMMON for S-Trigger, we might need a separate EffectType or update ActionGenerator logic
        // to check if source is in hand?
        // Let's rely on INTERNAL_PLAY for now. S-Trigger is "summoned without cost" which is effectively an effect summon anyway in many contexts.
        // But for "When you summon from hand" triggers, it matters.
        // S-Trigger IS from hand.

        // Let's modify ActionGenerator logic later if needed to check source zone?
        // Or we can just use EffectType::INTERNAL_PLAY and assume EFFECT_SUMMON is acceptable for now,
        // as S-Trigger is special.

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

        // INTERNAL PLAY: Pending Effect (Anywhere) -> Stack -> Resolve
        if (action.type == ActionType::PLAY_CARD_INTERNAL) {
            // Find the pending effect
            int index = action.slot_index;
            if (index < 0 || index >= static_cast<int>(game_state.pending_effects.size())) {
                return; // Error
            }

            // Should match type
            const auto& pe = game_state.pending_effects[index];
            if (pe.type != EffectType::INTERNAL_PLAY && pe.type != EffectType::META_COUNTER) {
                return;
            }

            int card_instance_id = action.source_instance_id;

            // Locate the card. It could be in Buffer, Hand, or anywhere depending on the effect.
            // Usually internal play moves it to stack first.
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

            // Check Buffer first (Mekraid, etc.)
            if (!find_and_set(game_state.effect_buffer)) {
                // Check Hand (Shield Trigger, Meta Counter)
                // Note: Shield Trigger adds to Hand first, then queues SHIELD_TRIGGER effect.
                // But wait, Shield Trigger uses USE_SHIELD_TRIGGER action which calls resolve_use_shield_trigger.
                // It does NOT use PLAY_CARD_INTERNAL yet. We plan to migrate it.
                // For now, assume Buffer or Hand.
                // We must check the controller's zones.
                Player& controller = game_state.players[pe.controller];
                if (!find_and_set(controller.hand)) {
                     if (!find_and_set(controller.graveyard)) { // Reanimate?
                         // ...
                     }
                }
            }

            if (card_ptr && source_zone) {
                CardInstance card = *card_ptr;
                source_zone->erase(std::remove_if(source_zone->begin(), source_zone->end(),
                    [card_instance_id](const CardInstance& c) { return c.instance_id == card_instance_id; }),
                    source_zone->end());

                // Move to Stack
                game_state.stack_zone.push_back(card);

                // Remove Pending Effect
                game_state.pending_effects.erase(game_state.pending_effects.begin() + index);

                // Call resolve_play_from_stack
                // Pass the controller from the PendingEffect
                resolve_play_from_stack(game_state, card.instance_id, 999, action.spawn_source, pe.controller, card_db);
            }
            return;
        }

        // DECLARE: Hand -> Stack
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

        // PAY_COST: Tap Mana
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

        // RESOLVE_PLAY: Stack -> Zone
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

                // Set turn_played for Just Diver / Summoning Sickness tracking
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

        // LEGACY PLAY_CARD (Atomic)
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

        // If cost reduction is massive (flagging free play/trampling), skip payment logic
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

        // Gatekeeper Logic: Determine destination
        if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
            card.summoning_sickness = true;
            if (def.keywords.speed_attacker) card.summoning_sickness = false;
            if (def.keywords.evolution) card.summoning_sickness = false;

            // Future expansion: Check SpawnSource for specific overrides
            // e.g. if (spawn_source == SpawnSource::EFFECT_PUT) ...

            card.is_tapped = false;

            // Set turn_played for Just Diver / Summoning Sickness tracking
            card.turn_played = game_state.turn_number;

            player.battle_zone.push_back(card);

            // Phase 5: Stats
            game_state.turn_stats.creatures_played_this_turn++;

            if (def.keywords.cip) {
                dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
            }
        } else if (def.type == CardType::SPELL) {
            card.is_tapped = false;
            player.graveyard.push_back(card);

            // Phase 5: Stats
            game_state.turn_stats.spells_cast_this_turn++;

            dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
        }
    }

    // Phase 5: New Resolve Effect with Execution Context
    void EffectResolver::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context) {
        if (!GenericCardSystem::check_condition(game_state, effect.condition, source_instance_id)) return;

        for (size_t i = 0; i < effect.actions.size(); ++i) {
            ActionDef action = effect.actions[i]; // Copy to modify values

            // Dynamic Value Substitution (Input)
            if (!action.input_value_key.empty()) {
                if (execution_context.count(action.input_value_key)) {
                    // Overwrite value1 or value (legacy str)
                    action.value1 = execution_context[action.input_value_key];
                    action.value = std::to_string(action.value1);
                }
            }

            // Handle Target Selection Interruption
            if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                EffectDef continuation;
                continuation.trigger = TriggerType::NONE;
                continuation.condition = ConditionDef{"NONE", 0, ""};
                // Propagate execution_context?
                // Currently context is local to this frame.
                // We should theoretically embed context in continuation or PendingEffect.
                // But PendingEffect doesn't store generic context yet.
                // For "Chain" actions, we might need to.
                // For MVP, assume context is consumed immediately or regenerated.
                // (To properly support cross-select variable persistence, we'd need to add context to PendingEffect)

                for (size_t j = i; j < effect.actions.size(); ++j) {
                    continuation.actions.push_back(effect.actions[j]);
                }
                GenericCardSystem::select_targets(game_state, action, source_instance_id, continuation);
                return; // Stop execution
            }

            // Phase 5: New Actions Implementation
            if (action.type == EffectActionType::COUNT_CARDS) {
                int count = 0;
                Player& active = game_state.get_active_player();

                // Helper to count in vector
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

                // Determine players to check based on filter.owner
                // Default to SELF if unspecified
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
                    // Fallback to SELF
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
                // Basic implementation for cost reduction modifier
                mod.reduction_amount = action.value1;
                mod.condition_filter = action.filter;
                mod.turns_remaining = action.value2 > 0 ? action.value2 : 1;
                mod.controller = game_state.active_player_id;
                mod.source_instance_id = source_instance_id;
                game_state.active_modifiers.push_back(mod);

            } else if (action.type == EffectActionType::REVEAL_CARDS) {
                // Stub
            } else if (action.type == EffectActionType::REGISTER_DELAYED_EFFECT) {
                // Stub
            } else if (action.type == EffectActionType::RESET_INSTANCE) {
                // Stub
            } else {
                GenericCardSystem::resolve_action(game_state, action, source_instance_id);
            }
        }
    }

    void EffectResolver::resolve_attack(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& active = game_state.get_active_player();
        auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        
        if (attacker_it == active.battle_zone.end()) return;
        attacker_it->is_tapped = true;

        game_state.current_attack.source_instance_id = action.source_instance_id;
        game_state.current_attack.is_blocked = false;
        game_state.current_attack.blocker_instance_id = -1;

        if (action.type == ActionType::ATTACK_PLAYER) {
            game_state.current_attack.target_instance_id = -1;
            game_state.current_attack.target_player = action.target_player;
        } else {
            game_state.current_attack.target_instance_id = action.target_instance_id;
        }

        bool revolution_change_triggered = false;
        CardInstance attacker = *attacker_it;
        const CardDefinition& attacker_def = card_db.at(attacker.card_id);

        for (const auto& card : active.hand) {
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);
                bool has_rev_change = def.keywords.revolution_change;

                if (has_rev_change) {
                    bool matches = true;
                    if (def.revolution_change_condition.has_value()) {
                        matches = TargetUtils::is_valid_target(attacker, attacker_def,
                                                               def.revolution_change_condition.value(),
                                                               game_state,
                                                               active.id, active.id);
                    }
                    if (matches) {
                        game_state.pending_effects.emplace_back(EffectType::ON_ATTACK_FROM_HAND, attacker.instance_id, active.id);
                        revolution_change_triggered = true;
                    }
                }
            }
        }

        if (!revolution_change_triggered) {
            // Check for Reactions (Ninja Strike, etc.)
            // Ninja Strike triggers on Attack or Block.
            // Check Opponent's Hand.
            bool reaction_opened = ReactionSystem::check_and_open_window(
                game_state, card_db, "ON_BLOCK_OR_ATTACK", game_state.get_non_active_player().id
            );

            // Also check Attacking Player's hand? Ninja Strike is usually defensive but rules allow offensive usage if conditions met.
            // Actually, Ninja Strike is "Whenever one of your creatures blocks or is blocked OR whenever your opponent attacks".
            // So on Attack: Opponent can use Ninja Strike. Attacker cannot (unless he blocks?).
            // Wait, Ninja Strike text: "Whenever your opponent attacks or blocks". NO.
            // Ninja Strike text: "Whenever your opponent attacks OR blocks". (JAP: 「相手のクリーチャーが攻撃またはブロックした時」)
            // Wait, standard Ninja Strike text: "相手のターン中に、シノビが攻撃またはブロックした時" ? No.
            // "ニンジャ・ストライク X（相手のクリーチャーが攻撃またはブロックした時、...）"
            // "Whenever an opponent's creature attacks or blocks".

            // So, when ACTIVE player attacks:
            //   NON-ACTIVE player can use Ninja Strike.

            if (!reaction_opened) {
                 game_state.current_phase = Phase::BLOCK;
            } else {
                 // The pending effect REACTION_WINDOW will pause the flow.
                 // We do NOT change phase yet, or we assume we are still in ATTACK phase but handling stack.
                 // Actually, Ninja Strike happens "At the end of the attack step" or "During attack"?
                 // It's a triggered ability.
                 // If we open a window, ActionGenerator will produce actions.
                 // After window closes, we proceed to BLOCK phase?
                 // Wait, Ninja Strike timing is "At the start of attack step" effectively?
                 // No, it's a Trigger. "When attacks".
                 // So we queue it.
                 // But ReactionSystem adds a REACTION_WINDOW pending effect.
                 // We need to ensure that when that resolves/is passed, we move to BLOCK phase.
                 // That transition needs to happen.

                 // How do we track "After Reaction Window"?
                 // We can rely on PhaseManager or ActionGenerator.
                 // For now, let's leave current_phase as ATTACK.
                 // But wait, if we don't change phase, ActionGenerator might generate ATTACK actions again?
                 // No, ActionGenerator prioritizes PendingEffects.
                 // Once REACTION_WINDOW is cleared, what happens?
                 // We need to transition to BLOCK.
                 // We might need a state flag "attack_declared_reaction_checked".
                 // OR, we can just say: If we are in ATTACK phase and have an active attack,
                 // and no pending effects, we move to BLOCK.
                 // But ActionGenerator::generate_legal_actions does this?
                 // Currently, resolve_attack sets Phase::BLOCK directly.
                 // If we don't set it here, we stay in ATTACK.
                 // And ActionGenerator will see "Phase::ATTACK" and "No Pending Effects" (after window closes).
                 // It might try to generate attacks again.
                 // We need to move to BLOCK *after* the window closes.
                 // The REACTION_WINDOW pending effect should probably have a callback or we use a distinct phase?

                 // Simpler approach:
                 // We set Phase::BLOCK immediately, BUT because we pushed a PendingEffect,
                 // the ActionGenerator will generate actions for that PendingEffect (Reaction)
                 // BEFORE generating BLOCK actions.
                 // Block actions are generated in Phase::BLOCK.
                 // If we are in Phase::BLOCK, can we use Ninja Strike?
                 // Ninja Strike usage is "When attacks".
                 // If we move to BLOCK phase, the blockers are declared.
                 // If Ninja Strike puts a blocker into play, it needs to be before Block declaration.
                 // So we must resolve Ninja Strike BEFORE Block Phase starts (or at start of Block Phase).

                 // So setting Phase::BLOCK here is correct, PROVIDED ActionGenerator
                 // checks PendingEffects BEFORE Block actions.
                 // ActionGenerator typically checks PendingEffects first.

                 game_state.current_phase = Phase::BLOCK;
            }
        }
    }

    void EffectResolver::resolve_reaction(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
         Player& controller = game_state.players[action.target_player];

         // Ninja Strike Logic: Play for free (Cost 0) from Hand
         // The action.source_instance_id points to the card in hand

         CardInstance card = remove_from_hand(controller, action.source_instance_id);

         // Move to Stack first as per protocol for resolve_play_from_stack
         game_state.stack_zone.push_back(card);

         // Use Cost Reduction 999 to simulate "No Cost" / "Free Play"
         // This ensures ManaSystem::auto_tap_mana is skipped or pays 0
         resolve_play_from_stack(game_state, card.instance_id, 999, SpawnSource::HAND_SUMMON, controller.id, card_db);

         // The REACTION_WINDOW pending effect remains, allowing multiple Ninja Strikes.
    }

}
