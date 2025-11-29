#include "effect_resolver.hpp"
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
                // Do nothing, PhaseManager handles phase transition if pass is chosen
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
            default:
                break;
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
            // For now, just basic placement
        } else if (def.type == CardType::SPELL) {
            // Spell effects would go here
            
            // Go to graveyard
            player.graveyard.push_back(card);
        }
    }

    void EffectResolver::resolve_attack(GameState& game_state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Player& active = game_state.get_active_player();
        Player& opponent = game_state.get_non_active_player();

        // Tap attacker
        auto attacker_it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
            [action](const CardInstance& c) { return c.instance_id == action.source_instance_id; });
        
        if (attacker_it == active.battle_zone.end()) {
            return;
        }
        attacker_it->is_tapped = true;
        CardInstance attacker = *attacker_it;

        if (action.type == ActionType::ATTACK_PLAYER) {
            // Break Shield
            if (!opponent.shield_zone.empty()) {
                // Break one shield (random or first?)
                // Usually attacker chooses? Or just break 1.
                // Standard: Break 1.
                CardInstance shield = opponent.shield_zone.back();
                opponent.shield_zone.pop_back();
                
                // Shield Trigger Check would go here
                // For now, add to hand
                opponent.hand.push_back(shield);
            } else {
                // Direct Attack -> Win
                if (active.id == 0) {
                    game_state.winner = GameResult::P1_WIN;
                } else {
                    game_state.winner = GameResult::P2_WIN;
                }
            }
        } else if (action.type == ActionType::ATTACK_CREATURE) {
            // Battle logic
            auto defender_it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                [action](const CardInstance& c) { return c.instance_id == action.target_instance_id; });
            
            if (defender_it == opponent.battle_zone.end()) {
                return;
            }
            CardInstance defender = *defender_it;

            int attacker_power = card_db.at(attacker.card_id).power;
            int defender_power = card_db.at(defender.card_id).power;

            bool destroy_attacker = false;
            bool destroy_defender = false;

            if (attacker_power > defender_power) {
                destroy_defender = true;
            } else if (attacker_power < defender_power) {
                destroy_attacker = true;
            } else {
                destroy_attacker = true;
                destroy_defender = true;
            }

            if (destroy_attacker) {
                auto it = std::find_if(active.battle_zone.begin(), active.battle_zone.end(),
                    [attacker](const CardInstance& c) { return c.instance_id == attacker.instance_id; });
                if (it != active.battle_zone.end()) {
                    active.graveyard.push_back(*it);
                    active.battle_zone.erase(it);
                }
            }

            if (destroy_defender) {
                auto it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                    [defender](const CardInstance& c) { return c.instance_id == defender.instance_id; });
                if (it != opponent.battle_zone.end()) {
                    opponent.graveyard.push_back(*it);
                    opponent.battle_zone.erase(it);
                }
            }
        }
    }

}
