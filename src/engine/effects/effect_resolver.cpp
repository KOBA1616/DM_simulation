#include "effect_resolver.hpp"
#include "generated_effects.hpp"
#include "../card_system/generic_card_system.hpp"
#include "../card_system/card_registry.hpp"
#include "../mana/mana_system.hpp"
#include <iostream>
#include <algorithm>

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
                    // Pass in Block Phase means "No Block" -> Resolve Battle
                    execute_battle(game_state, card_db);
                }
                // Otherwise, PhaseManager handles phase transition
                break;
            case ActionType::MANA_CHARGE:
                resolve_mana_charge(game_state, action);
                break;
            case ActionType::PLAY_CARD:
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
        // Handle Revolution Change, Ninja Strike, etc.
        if (game_state.pending_effects.empty()) return;

        // Find the pending effect
        int pe_index = -1;
        // In ActionGenerator, USE_ABILITY slot_index usually points to the hand index.
        // We need to know WHICH pending effect triggered this.
        // Currently Action struct doesn't link back to PendingEffect index for USE_ABILITY unless we use slot_index for that.
        // But ActionGenerator::generate_legal_actions for USE_ABILITY (to be implemented) should set slot_index to hand index?
        // Wait, for SHIELD_TRIGGER, slot_index IS the pending effect index.
        // For USE_ABILITY, let's assume slot_index is the hand index (to identify the card),
        // BUT we also need to clear the Pending Effect.

        // Let's adopt the convention: For USE_ABILITY, the pending effect is the one at the FRONT (or we scan).
        // Since Revolution Change is an interruption, it should be at the top of the stack.

        // Actually, we need to know specifically which PendingEffect this action is resolving.
        // Let's assume the ActionGenerator sets `target_slot_index` to the PendingEffect index?
        // Or we just look for the first matching PendingEffect type.

        // Let's implement the logic for REVOLUTION_CHANGE specifically for now.
        // We scan pending effects for ON_ATTACK_FROM_HAND.

        int target_pe_index = -1;
        for (int i = 0; i < (int)game_state.pending_effects.size(); ++i) {
             if (game_state.pending_effects[i].type == EffectType::ON_ATTACK_FROM_HAND) {
                 target_pe_index = i;
                 break;
             }
        }

        if (target_pe_index == -1) return;

        Player& player = game_state.get_active_player(); // Attacking player

        // Execute Revolution Change Swap
        // 1. Return Attacking Creature to Hand
        int attacker_id = game_state.current_attack.source_instance_id;

        auto attacker_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(),
            [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });

        if (attacker_it != player.battle_zone.end()) {
            CardInstance returned_creature = *attacker_it;
            // Untap it upon return? Rules say: "Return to hand."
            // Tapped status in hand is irrelevant usually, but good to reset.
            returned_creature.is_tapped = false;
            player.battle_zone.erase(attacker_it);
            player.hand.push_back(returned_creature);
        } else {
            // Attacker missing? Revolution Change fails?
            // Rules: If attacker leaves before resolution, change fails?
            // "You may switch that creature with this creature."
            // If that creature is gone, you can't switch.
            game_state.pending_effects.erase(game_state.pending_effects.begin() + target_pe_index);
            return;
        }

        // 2. Put Revolution Change Creature into Battle Zone
        try {
            CardInstance new_creature = remove_from_hand(player, action.source_instance_id);

            // Inherit State?
            // "It enters the battle zone tapped and attacking."
            new_creature.is_tapped = true;
            new_creature.summoning_sickness = false; // It's attacking, so effective speed attacker

            player.battle_zone.push_back(new_creature);

            // Update Global Attack State
            game_state.current_attack.source_instance_id = new_creature.instance_id;

            // Trigger CIP
            if (card_db.count(new_creature.card_id)) {
                const auto& def = card_db.at(new_creature.card_id);
                if (def.keywords.cip) {
                    dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, new_creature.instance_id);
                }
            }

        } catch (...) {
            // Error handling
        }

        // Remove the PendingEffect
        if (target_pe_index < (int)game_state.pending_effects.size()) {
             game_state.pending_effects.erase(game_state.pending_effects.begin() + target_pe_index);
        }

        // Transition to BLOCK Phase to continue the attack sequence (opponent gets to block)
        game_state.current_phase = Phase::BLOCK;
    }

    void EffectResolver::resolve_block(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        // Change target of current attack to blocker
        game_state.current_attack.is_blocked = true;
        game_state.current_attack.blocker_instance_id = action.source_instance_id;
        
        // Tap blocker
        Player& defender = game_state.get_non_active_player(); // NAP is blocking
        auto it = std::find_if(defender.battle_zone.begin(), defender.battle_zone.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        if (it != defender.battle_zone.end()) {
            it->is_tapped = true;
        }

        // Execute Battle immediately after block declaration
        execute_battle(game_state, card_db);
    }

    int EffectResolver::get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        if (card_db.find(creature.card_id) == card_db.end()) return 0;
        const auto& def = card_db.at(creature.card_id);
        int power = def.power;

        // Power Attacker
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

    void EffectResolver::execute_battle(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        Player& active = game_state.get_active_player();
        Player& opponent = game_state.get_non_active_player();
        
        int attacker_id = game_state.current_attack.source_instance_id;
        int target_id = game_state.current_attack.target_instance_id;
        bool is_blocked = game_state.current_attack.is_blocked;
        int blocker_id = game_state.current_attack.blocker_instance_id;

        // Find Attacker
        auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
            [attacker_id](const CardInstance& c) { return c.instance_id == attacker_id; });
        
        if (attacker_it == active.battle_zone.end()) return; // Attacker gone?
        CardInstance attacker = *attacker_it;

        if (is_blocked) {
            // Battle between Attacker and Blocker
            auto blocker_it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                [blocker_id](const CardInstance& c) { return c.instance_id == blocker_id; });
            
            if (blocker_it == opponent.battle_zone.end()) return; // Blocker gone?
            CardInstance blocker = *blocker_it;

            int attacker_power = get_creature_power(attacker, game_state, card_db);
            int blocker_power = get_creature_power(blocker, game_state, card_db);
            
            // Battle Logic
            bool destroy_attacker = (attacker_power <= blocker_power);
            bool destroy_blocker = (blocker_power <= attacker_power);

            // Slayer
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

        } else {
            // Not Blocked
            if (target_id == -1) {
                // Attack Player
                // Break Shield
                int break_count = get_breaker_count(attacker, card_db);
                
                for (int i = 0; i < break_count; ++i) {
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
                        // Direct Attack
                        if (active.id == 0) game_state.winner = GameResult::P1_WIN;
                        else game_state.winner = GameResult::P2_WIN;
                        break; // Game Over
                    }
                }
            } else {
                // Attack Creature
                auto defender_it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                    [target_id](const CardInstance& c) { return c.instance_id == target_id; });
                
                if (defender_it != opponent.battle_zone.end()) {
                    CardInstance defender = *defender_it;
                    int attacker_power = get_creature_power(attacker, game_state, card_db);
                    int defender_power = get_creature_power(defender, game_state, card_db);

                    bool destroy_attacker = (attacker_power <= defender_power);
                    bool destroy_defender = (defender_power <= attacker_power);

                    // Slayer
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
        }
        
        // Reset Phase to ATTACK
        game_state.current_phase = Phase::ATTACK;
    }

    void EffectResolver::resolve_pending_effect(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        if (game_state.pending_effects.empty()) return;
        
        int index = action.slot_index;
        // Safety check
        if (index < 0 || index >= static_cast<int>(game_state.pending_effects.size())) {
            // Fallback to back if index is invalid (e.g. from old logic)
            index = game_state.pending_effects.size() - 1;
        }

        PendingEffect effect = game_state.pending_effects[index];
        // std::cerr << "[EffectResolver] Resolving pending effect index=" << index << " type=" << (int)effect.type << " src=" << effect.source_instance_id << " controller=" << (int)effect.controller << " has_effect_def=" << (int)effect.effect_def.has_value() << " num_targets_needed=" << effect.num_targets_needed << " targets_selected=" << effect.target_instance_ids.size() << "\n";
        // If this pending effect carries a JSON EffectDef and requires targets,
        // wait until enough targets have been selected before popping it.
        if (effect.effect_def.has_value()) {
            if (effect.num_targets_needed > static_cast<int>(effect.target_instance_ids.size())) {
                // Not enough targets selected yet - keep the pending effect in the queue
                return;
            }
            // Enough targets have been selected - remove from queue and resolve
            game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
        } else {
            // copy pending effect then erase from queue for non-JSON-backed effects
            game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
        }

        // TODO: Implement specific effect logic based on effect.type and source card
        Player& controller = game_state.players[effect.controller];
        Player& opponent = game_state.players[1 - effect.controller];

        // Find source card definition
        // Source might be in Battle Zone, Graveyard, or just an ID if it was a spell that went to grave.
        // For simplicity, we assume we can look up ID from instance ID if we track it, or we need to store CardID in PendingEffect.
        // Let's assume we can find it in battle zone or graveyard.
        // Or better, let's look up CardID from instance_id across all zones? Expensive.
        // Let's just use the card_db and assume we know the ID? No.
        // We need to find the card instance to know its ID.
        
        CardID card_id = 0;
        // Search all common zones for source instance to find the CardID
        auto find_in_vec = [&](const std::vector<CardInstance>& vec)->CardID{
            for (const auto& c : vec) if (c.instance_id == effect.source_instance_id) return c.card_id;
            return (CardID)0;
        };

        card_id = find_in_vec(controller.battle_zone);
        if (card_id == 0) card_id = find_in_vec(controller.hand);
        if (card_id == 0) card_id = find_in_vec(controller.mana_zone);
        if (card_id == 0) card_id = find_in_vec(controller.graveyard);
        if (card_id == 0) card_id = find_in_vec(controller.shield_zone);

        if (card_id == 0) return; // Source gone?

        // If the pending effect carries an EffectDef (from JSON), resolve it using any selected targets
        if (effect.effect_def.has_value()) {
            dm::engine::GenericCardSystem::resolve_effect_with_targets(game_state, effect.effect_def.value(), effect.target_instance_ids, effect.source_instance_id);
            return;
        }

        // Prefer GenericCardSystem for resolving effects via JSON definitions
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
                    // If the card's JSON effects contain actions that require target selection,
                    // create a PendingEffect for selection instead of immediately resolving.
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

        // Fallback: existing generated-effects resolver
        GeneratedEffects::resolve(game_state, effect, card_id);
    }

    void EffectResolver::resolve_use_shield_trigger(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& st_player = game_state.players[action.target_player];
        
        try {
            CardInstance card = remove_from_hand(st_player, action.source_instance_id);
            const CardDefinition& def = card_db.at(card.card_id);
            
            // STATS TRACKING: Record Shield Trigger usage
            // Cost discount is full cost (since it's free)
            game_state.on_card_play(card.card_id, game_state.turn_number, true, def.cost, st_player.id);

            if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
                card.summoning_sickness = true;
                if (def.keywords.speed_attacker) card.summoning_sickness = false;
                if (def.keywords.evolution) card.summoning_sickness = false;
                st_player.battle_zone.push_back(card);
                
                // Trigger CIP via GenericCardSystem
                if (def.keywords.cip) {
                     dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
                }

            } else if (def.type == CardType::SPELL) {
                st_player.graveyard.push_back(card);
                
                // Trigger Spell Effect (Treat as CIP for spell logic in GenericCardSystem)
                // In GenericCardSystem, Spell effects are defined as TriggerType::ON_PLAY for simplicity.
                dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
            }
        } catch (...) {
            // Card not found or other error
        }

        // Remove the pending SHIELD_TRIGGER effect that triggered this
        if (!game_state.pending_effects.empty()) {
             int index = action.slot_index;
             if (index >= 0 && index < static_cast<int>(game_state.pending_effects.size())) {
                 if (game_state.pending_effects[index].type == EffectType::SHIELD_TRIGGER) {
                     game_state.pending_effects.erase(game_state.pending_effects.begin() + index);
                 }
             }
        }
    }

    void EffectResolver::resolve_mana_charge(GameState& game_state, const Action& action) {
        Player& player = game_state.get_active_player();
        try {
            CardInstance card = remove_from_hand(player, action.source_instance_id);
            // Mana enters tapped? Usually no, unless specified.
            // Standard DM: Mana enters untapped.
            card.is_tapped = false; 
            player.mana_zone.push_back(card);
        } catch (...) {
            // Handle error
        }
    }

    void EffectResolver::resolve_play_card(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& player = game_state.get_active_player();
        const CardDefinition& def = card_db.at(action.card_id);

        // --- STEP 1: DECLARATION (Hand -> Stack) ---
        // Verify card is in hand and move to Stack Zone
        auto it = std::find_if(player.hand.begin(), player.hand.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });

        if (it == player.hand.end()) return; // Error: Card not in hand

        CardInstance card = *it;
        player.hand.erase(it);
        game_state.stack_zone.push_back(card);

        // --- STEP 2: COST CALCULATION & PAYMENT ---
        // Use auto_tap_mana with GameState for Cost Modifiers
        if (!ManaSystem::auto_tap_mana(game_state, player, def, card_db)) {
            // Failed to pay cost - rollback (return to hand)
            game_state.stack_zone.pop_back();
            player.hand.push_back(card);
            return;
        }

        // STATS TRACKING: Record normal play
        // For now, assume cost discount is 0 (paid full cost) - TODO: Calculate actual paid cost
        game_state.on_card_play(action.card_id, game_state.turn_number, false, 0, player.id);

        // --- STEP 3: RESOLUTION (Stack -> Zone) ---
        // Remove from Stack Zone
        auto s_it = std::find_if(game_state.stack_zone.begin(), game_state.stack_zone.end(),
             [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        if (s_it != game_state.stack_zone.end()) {
            game_state.stack_zone.erase(s_it);
        }

        // Move to destination (Battle Zone or Graveyard)
        if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
            // Summoning Sickness
            card.summoning_sickness = true;
            if (def.keywords.speed_attacker) {
                card.summoning_sickness = false;
            }
            if (def.keywords.evolution) {
                card.summoning_sickness = false;
            }
            
            player.battle_zone.push_back(card);

            // CIP Effects (Enter the Battlefield): delegate to GenericCardSystem
            if (def.keywords.cip) {
                dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
            }
        } else if (def.type == CardType::SPELL) {
            // Go to graveyard
            player.graveyard.push_back(card);

            // Spell effects: Use GenericCardSystem
            dm::engine::GenericCardSystem::resolve_trigger(game_state, dm::core::TriggerType::ON_PLAY, card.instance_id);
        }
    }

    void EffectResolver::resolve_attack(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& active = game_state.get_active_player();
        
        // Tap attacker
        auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        
        if (attacker_it == active.battle_zone.end()) {
             std::cerr << "Attacker not found in BZ: " << action.source_instance_id << "\n";
             return;
        }
        attacker_it->is_tapped = true;

        // Setup Attack Context
        game_state.current_attack.source_instance_id = action.source_instance_id;
        game_state.current_attack.is_blocked = false;
        game_state.current_attack.blocker_instance_id = -1;

        if (action.type == ActionType::ATTACK_PLAYER) {
            game_state.current_attack.target_instance_id = -1;
            game_state.current_attack.target_player = action.target_player;
        } else {
            game_state.current_attack.target_instance_id = action.target_instance_id;
        }

        // Check for Revolution Change Triggers (ON_ATTACK_FROM_HAND)
        bool revolution_change_triggered = false;

        // Debug: Check Hand
        // std::cerr << "Checking Hand for Rev Change. Hand size: " << active.hand.size() << "\n";

        CardInstance attacker = *attacker_it; // Copy to be safe
        const CardDefinition& attacker_def = card_db.at(attacker.card_id);

        for (const auto& card : active.hand) {
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);
                bool has_rev_change = def.keywords.revolution_change;
                // std::cerr << "Card " << card.card_id << " has_rev_change=" << has_rev_change << "\n";
                if (has_rev_change) {
                    // Check conditions: Civilization or Race usually.
                    // Revolution Change: Fire (Switch with a Fire creature)
                    // We need to check if the attacker matches the requirement.
                    // This logic is specific to the card.
                    // IMPORTANT: We need to know IF it matches.
                    // For now, let's Push the PendingEffect if ANY RevChange card exists,
                    // and ActionGenerator will filter if it's actually usable.
                    // If ActionGenerator produces no actions (PASS only), then we resolve PASS immediately?

                    // To avoid infinite loops or empty checks, we only push if we haven't already?
                    // No, this is a one-time event per attack declaration.

                    game_state.pending_effects.emplace_back(EffectType::ON_ATTACK_FROM_HAND, attacker.instance_id, active.id);
                    revolution_change_triggered = true;
                    break; // Only need one pending effect to trigger the choice phase
                }
            }
        }

        // Transition to BLOCK Phase only if no interrupts
        if (!revolution_change_triggered) {
            game_state.current_phase = Phase::BLOCK;
        } else {
            // Stay in ATTACK phase but with pending effect?
            // Or use a transient phase?
            // ActionGenerator handles pending effects regardless of phase.
            // But if we stay in ATTACK phase, standard attack actions might be generated again?
            // No, ActionGenerator prioritizes PendingEffects (Step 0).
            // So as long as PendingEffect exists, standard actions are suppressed or secondary.
            // Wait, ActionGenerator DOES suppress other actions if PendingEffects exist.
            // "If !game_state.pending_effects.empty() ... return actions;"
            // So we are safe.
        }
    }

}
