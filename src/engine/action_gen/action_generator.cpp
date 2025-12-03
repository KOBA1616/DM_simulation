#include "action_generator.hpp"
#include "../mana/mana_system.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ActionGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        std::vector<Action> actions;

        // 0. Pending Effects (The Stack)
        if (!game_state.pending_effects.empty()) {
            // Priority: AP -> NAP
            bool ap_has_effects = false;
            for (const auto& eff : game_state.pending_effects) {
                if (eff.controller == game_state.active_player_id) {
                    ap_has_effects = true;
                    break;
                }
            }
            
            PlayerID priority_player = ap_has_effects ? game_state.active_player_id : (1 - game_state.active_player_id);
            
            for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
                const auto& eff = game_state.pending_effects[i];
                if (eff.controller == priority_player) {
                    // Check if we need targets
                    if (eff.num_targets_needed > (int)eff.target_instance_ids.size()) {
                        // Generate SELECT_TARGET actions
                        // For now, assume target is opponent creature (e.g. Terror Pit)
                        // In real implementation, we need filter info from CardDefinition
                        const Player& opponent = game_state.players[1 - priority_player];
                        for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
                            Action select;
                            select.type = ActionType::SELECT_TARGET;
                            select.target_instance_id = opponent.battle_zone[j].instance_id;
                            select.slot_index = static_cast<int>(i); // Effect index
                            actions.push_back(select);
                        }
                        // If no targets, maybe pass? Or auto-resolve?
                        // Spec says "Maximize fulfillment". If no targets, effect might fizzle or skip targeting.
                        if (actions.empty()) {
                             Action resolve;
                             resolve.type = ActionType::RESOLVE_EFFECT;
                             resolve.slot_index = static_cast<int>(i);
                             actions.push_back(resolve);
                        }
                    } else if (eff.type == EffectType::SHIELD_TRIGGER) {
                        Action use;
                        use.type = ActionType::USE_SHIELD_TRIGGER;
                        use.source_instance_id = eff.source_instance_id;
                        use.target_player = eff.controller;
                        use.slot_index = static_cast<int>(i); // Store index
                        actions.push_back(use);
                        
                        Action pass;
                        pass.type = ActionType::RESOLVE_EFFECT;
                        pass.slot_index = static_cast<int>(i); // Store index
                        actions.push_back(pass);
                    } else {
                        Action resolve;
                        resolve.type = ActionType::RESOLVE_EFFECT;
                        resolve.slot_index = static_cast<int>(i); // Store index
                        actions.push_back(resolve);
                    }
                }
            }
            return actions;
        }

        const Player& active_player = game_state.players[game_state.active_player_id];
        const Player& opponent = game_state.players[1 - game_state.active_player_id];

        switch (game_state.current_phase) {
            case Phase::MANA:
                // 1. Charge Mana (any card from hand)
                for (size_t i = 0; i < active_player.hand.size(); ++i) {
                    const auto& card = active_player.hand[i];
                    Action action;
                    action.type = ActionType::MANA_CHARGE;
                    action.card_id = card.card_id;
                    action.source_instance_id = card.instance_id;
                    action.slot_index = static_cast<int>(i);
                    actions.push_back(action);
                }
                // 2. Pass
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::MAIN:
                // 1. Play Card
                for (size_t i = 0; i < active_player.hand.size(); ++i) {
                    const auto& card = active_player.hand[i];
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (ManaSystem::can_pay_cost(active_player, def, card_db)) {
                            Action action;
                            action.type = ActionType::PLAY_CARD;
                            action.card_id = card.card_id;
                            action.source_instance_id = card.instance_id;
                            action.slot_index = static_cast<int>(i);
                            actions.push_back(action);
                        }
                    }
                }

                // 2. Pass
                // Transition to ATTACK phase via PhaseManager::next_phase()
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::ATTACK:
                // 1. Attack with creatures
                for (size_t i = 0; i < active_player.battle_zone.size(); ++i) {
                    const auto& card = active_player.battle_zone[i];

                    bool can_attack = !card.is_tapped;
                    if (can_attack) {
                        // Check Summoning Sickness vs Speed Attacker / Evolution
                        if (card.summoning_sickness) {
                            if (card_db.count(card.card_id)) {
                                const auto& def = card_db.at(card.card_id);
                                if (!def.keywords.speed_attacker && !def.keywords.evolution) {
                                    can_attack = false;
                                }
                            } else {
                                can_attack = false; // Unknown card, default to sick
                            }
                        }
                    }

                    if (can_attack) {
                        // Attack Player
                        Action attack_player;
                        attack_player.type = ActionType::ATTACK_PLAYER;
                        attack_player.source_instance_id = card.instance_id;
                        attack_player.slot_index = static_cast<int>(i);
                        attack_player.target_player = opponent.id;
                        actions.push_back(attack_player);

                        // Attack Tapped Creatures
                        for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
                            const auto& opp_card = opponent.battle_zone[j];
                            if (opp_card.is_tapped) {
                                Action attack_creature;
                                attack_creature.type = ActionType::ATTACK_CREATURE;
                                attack_creature.source_instance_id = card.instance_id;
                                attack_creature.slot_index = static_cast<int>(i);
                                attack_creature.target_instance_id = opp_card.instance_id;
                                attack_creature.target_slot_index = static_cast<int>(j);
                                actions.push_back(attack_creature);
                            }
                        }
                    }
                }
                // 2. Pass (End Attack Phase)
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::BLOCK:
                // Generate Block actions for NAP
                // Note: In BLOCK phase, active_player is still the turn player, but NAP acts.
                // We need to check who is supposed to act.
                // Usually ActionGenerator generates actions for the "decision maker".
                // If MCTS calls this, it expects actions for the current decision maker.
                // We might need to handle this in MCTS or here.
                // For now, let's assume we generate actions for NAP if phase is BLOCK.
                {
                    const Player& defender = opponent; // NAP
                    for (size_t i = 0; i < defender.battle_zone.size(); ++i) {
                        const auto& card = defender.battle_zone[i];
                        if (!card.is_tapped) {
                            if (card_db.count(card.card_id)) {
                                const auto& def = card_db.at(card.card_id);
                                if (def.keywords.blocker) {
                                    Action block;
                                    block.type = ActionType::BLOCK;
                                    block.source_instance_id = card.instance_id;
                                    block.slot_index = static_cast<int>(i);
                                    actions.push_back(block);
                                }
                            }
                        }
                    }
                    // Pass (Don't Block)
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            default:
                break;
        }

        return actions;
    }

}
