#include "effect_resolver.hpp"
#include "generated_effects.hpp"
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
                resolve_block(game_state, action);
                break;
            case ActionType::SELECT_TARGET:
                resolve_select_target(game_state, action);
                break;
            default:
                break;
        }
    }

    void EffectResolver::resolve_select_target(GameState& game_state, const Action& action) {
        if (game_state.pending_effects.empty()) return;
        
        int index = action.slot_index;
        if (index >= 0 && index < static_cast<int>(game_state.pending_effects.size())) {
            game_state.pending_effects[index].target_instance_ids.push_back(action.target_instance_id);
        }
    }

    void EffectResolver::resolve_block(GameState& game_state, const Action& action) {
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
        // (Or should we wait for more steps? Standard DM: Block -> Battle)
        // Since we don't have "Blocker vs Slayer" timing complexities yet, execute battle.
        // But wait, resolve_attack was split. We need execute_battle.
        // Actually, if we are in BLOCK phase, we should transition back to ATTACK or resolve battle then transition.
        // Let's execute battle here.
        // But we need card_db.
        // resolve_block signature needs card_db? Or execute_battle needs it.
        // Let's change signature of resolve_block to take card_db if needed, or just call execute_battle which takes it.
        // Wait, resolve_action passes card_db.
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

            int attacker_power = card_db.at(attacker.card_id).power;
            int blocker_power = card_db.at(blocker.card_id).power;
            
            // Battle Logic
            bool destroy_attacker = (attacker_power <= blocker_power);
            bool destroy_blocker = (blocker_power <= attacker_power);

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
                }
            } else {
                // Attack Creature
                auto defender_it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                    [target_id](const CardInstance& c) { return c.instance_id == target_id; });
                
                if (defender_it != opponent.battle_zone.end()) {
                    CardInstance defender = *defender_it;
                    int attacker_power = card_db.at(attacker.card_id).power;
                    int defender_power = card_db.at(defender.card_id).power;

                    bool destroy_attacker = (attacker_power <= defender_power);
                    bool destroy_defender = (defender_power <= attacker_power);

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
        game_state.pending_effects.erase(game_state.pending_effects.begin() + index);

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
        // Search Battle Zone
        auto it_bz = std::find_if(controller.battle_zone.begin(), controller.battle_zone.end(),
            [&](const CardInstance& c) { return c.instance_id == effect.source_instance_id; });
        if (it_bz != controller.battle_zone.end()) card_id = it_bz->card_id;
        
        // Search Graveyard (for Spells)
        if (card_id == 0) {
             auto it_gy = std::find_if(controller.graveyard.begin(), controller.graveyard.end(),
                [&](const CardInstance& c) { return c.instance_id == effect.source_instance_id; });
             if (it_gy != controller.graveyard.end()) card_id = it_gy->card_id;
        }

        if (card_id == 0) return; // Source gone?

        // Use Generated Effects
        GeneratedEffects::resolve(game_state, effect, card_id);
    }

    void EffectResolver::resolve_use_shield_trigger(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& st_player = game_state.players[action.target_player];
        
        try {
            CardInstance card = remove_from_hand(st_player, action.source_instance_id);
            const CardDefinition& def = card_db.at(card.card_id);
            
            if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
                card.summoning_sickness = true;
                if (def.keywords.speed_attacker) card.summoning_sickness = false;
                if (def.keywords.evolution) card.summoning_sickness = false;
                st_player.battle_zone.push_back(card);
                
                // Trigger CIP
                if (def.keywords.cip) {
                    PendingEffect cip(EffectType::CIP, card.instance_id, st_player.id);
                    if (card.card_id == 5) cip.num_targets_needed = 1;
                    game_state.pending_effects.push_back(cip);
                }

            } else if (def.type == CardType::SPELL) {
                st_player.graveyard.push_back(card);
                
                // Trigger Spell Effect
                PendingEffect spell_effect(EffectType::CIP, card.instance_id, st_player.id);
                if (card.card_id == 5) spell_effect.num_targets_needed = 1;
                game_state.pending_effects.push_back(spell_effect);
            }
        } catch (...) {
            // Card not found or other error
        }

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

        // 1. Pay Cost (Auto-tap)
        if (!ManaSystem::auto_tap_mana(player, def, card_db)) {
            // Should not happen if action was legal
            return;
        }

        // 2. Move Card
        CardInstance card = remove_from_hand(player, action.source_instance_id);

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
            
            // CIP Effects (Enter the Battlefield) would go here
            if (def.keywords.cip) {
                PendingEffect cip(EffectType::CIP, card.instance_id, player.id);
                // Hardcode targeting requirement for Terror Pit (ID 5) or similar
                if (card.card_id == 5) { // Terror Pit
                     cip.num_targets_needed = 1;
                }
                game_state.pending_effects.push_back(cip);
            }
        } else if (def.type == CardType::SPELL) {
            // Spell effects would go here
            // For spells, effect happens immediately or goes to stack?
            // Spec says "EffectResolver (Stack Machine)". Spells go to stack.
            PendingEffect spell_effect(EffectType::CIP, card.instance_id, player.id); // Use CIP type for spell resolution for now
             if (card.card_id == 5) { // Terror Pit
                 spell_effect.num_targets_needed = 1;
            }
            game_state.pending_effects.push_back(spell_effect);
            
            // Go to graveyard
            player.graveyard.push_back(card);
        }
    }

    void EffectResolver::resolve_attack(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& active = game_state.get_active_player();
        
        // Tap attacker
        auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        
        if (attacker_it == active.battle_zone.end()) return;
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

        // Transition to BLOCK Phase
        game_state.current_phase = Phase::BLOCK;
    }

}
