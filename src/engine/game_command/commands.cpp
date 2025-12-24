#include "commands.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_event.hpp"
#include "engine/systems/card/card_registry.hpp" // Added for G-Neo lookup
#include <iostream>
#include <algorithm>

namespace dm::engine::game_command {

    // --- TransitionCommand ---

    void TransitionCommand::execute(core::GameState& state) {
        // --- G-Neo Handling ---
        // We must check for G-Neo BEFORE removing the card, as removal might trigger things
        // or we need access to the card in its original context.
        // However, we can use state.get_card_instance() to inspect it.

        bool should_replace = false;
        if (from_zone == core::Zone::BATTLE && to_zone != core::Zone::BATTLE) {
            core::CardInstance* card_ptr = state.get_card_instance(card_instance_id);
            if (card_ptr && !card_ptr->underlying_cards.empty()) {
                 // Access Global Card Registry (Singleton)
                 const auto& card_db = dm::engine::CardRegistry::get_all_definitions();
                 if (card_db.count(card_ptr->card_id)) {
                     const auto& def = card_db.at(card_ptr->card_id);
                     if (def.keywords.g_neo) {
                         should_replace = true;
                     }
                 }
            }
        }

        if (should_replace) {
             g_neo_activated = true;

             core::CardInstance* card_ptr = state.get_card_instance(card_instance_id);
             // Safety check
             if (!card_ptr) return;

             // Store underlying cards for Undo
             moved_underlying_cards = card_ptr->underlying_cards;

             // Move underlying to Graveyard
             auto& grave = state.players[owner_id].graveyard;
             for (const auto& under : card_ptr->underlying_cards) {
                 grave.push_back(under);

                 // Dispatch ZONE_ENTER for underlying cards entering Graveyard
                 if (state.event_dispatcher) {
                    core::GameEvent evt;
                    evt.type = core::EventType::ZONE_ENTER;
                    evt.card_id = under.card_id;
                    evt.instance_id = under.instance_id;
                    evt.player_id = owner_id;
                    evt.context["from_zone"] = static_cast<int>(core::Zone::BATTLE); // Effectively from under battle card
                    evt.context["to_zone"] = static_cast<int>(core::Zone::GRAVEYARD);
                    state.event_dispatcher(evt);
                 }
             }

             // Update the card in source vector (clear underlying)
             card_ptr->underlying_cards.clear();

             // Abort the move of the top card (Replacement Effect)
             return;
        }

        // --- Standard Move Logic using GameState Primitives ---

        auto removed_data = state.remove_card_from_zone(owner_id, from_zone, card_instance_id);
        if (!removed_data.first.has_value()) {
            // Card not found in source zone. Abort.
            return;
        }

        core::CardInstance card = removed_data.first.value();
        original_index = removed_data.second; // Store for undo

        // Update card state for destination
        if (to_zone == core::Zone::HAND || to_zone == core::Zone::DECK) {
            card.is_tapped = false;
            card.summoning_sickness = true; // Hand cards have sickness technically (cannot attack)
            card.underlying_cards.clear();
        }
        if (to_zone == core::Zone::BATTLE) {
            card.turn_played = state.turn_number;
            card.summoning_sickness = true;
        }

        // Insert into destination
        state.insert_card_to_zone(owner_id, to_zone, card, destination_index);

        // Phase 6: Event Dispatch
        if (state.event_dispatcher) {
            core::GameEvent evt;
            evt.type = core::EventType::ZONE_ENTER;
            evt.card_id = card.card_id;
            evt.instance_id = card_instance_id;
            evt.player_id = owner_id;

            // Context
            evt.context["instance_id"] = card_instance_id;
            evt.context["from_zone"] = static_cast<int>(from_zone);
            evt.context["to_zone"] = static_cast<int>(to_zone);
            evt.context["card_id"] = card.card_id;

            state.event_dispatcher(evt);
        }
    }

    void TransitionCommand::invert(core::GameState& state) {
        // --- G-Neo Undo Logic ---
        if (g_neo_activated) {
            // Restore underlying cards from Graveyard to Battle Zone (under the creature).
            // The creature (card_instance_id) is still in from_zone (BATTLE).

            core::CardInstance* card_ptr = state.get_card_instance(card_instance_id);
            if (!card_ptr) return; // Should be there

            // Restore underlying cards structure
            card_ptr->underlying_cards = moved_underlying_cards;

            // Remove specific instances from Graveyard
            for (const auto& moved_card : moved_underlying_cards) {
                state.remove_card_from_zone(owner_id, core::Zone::GRAVEYARD, moved_card.instance_id);
            }

            return;
        }

        // --- Standard Undo Logic ---
        // Move from `to_zone` back to `from_zone` at `original_index`.

        auto removed_data = state.remove_card_from_zone(owner_id, to_zone, card_instance_id);
        if (!removed_data.first.has_value()) {
            return;
        }

        core::CardInstance card = removed_data.first.value();

        // Note: We might want to restore original state flags (tapped, sick), but currently TransitionCommand
        // doesn't store them all. It assumes they are reset on move or managed by other commands.
        // For strict undo, we usually rely on the fact that `CardInstance` is a value type and we pushed a copy?
        // No, `remove_card_from_zone` returns the CURRENT state of the card in the `to_zone`.
        // If the card was modified in `to_zone` (e.g. tapped), moving it back might keep it tapped unless we reset.
        // However, `TransitionCommand` logic typically assumes the move is the only thing happening in this command step.
        // If other commands modified it, their `invert` should have run first.

        state.insert_card_to_zone(owner_id, from_zone, card, original_index);
    }

    // --- MutateCommand ---

    void MutateCommand::execute(core::GameState& state) {
        // Special case for global modifiers (instance_id might be ignored or -1)
        if (mutation_type == MutationType::ADD_PASSIVE_EFFECT) {
            if (passive_effect) {
                state.passive_effects.push_back(*passive_effect);
            }
            return;
        } else if (mutation_type == MutationType::ADD_COST_MODIFIER) {
            if (cost_modifier) {
                state.active_modifiers.push_back(*cost_modifier);
            }
            return;
        } else if (mutation_type == MutationType::ADD_PENDING_EFFECT) {
            if (pending_effect) {
                state.pending_effects.push_back(*pending_effect);
            }
            return;
        }

        core::CardInstance* card = state.get_card_instance(target_instance_id);
        if (!card) return;

        switch(mutation_type) {
            case MutationType::TAP:
                previous_bool_value = card->is_tapped;
                card->is_tapped = true;
                break;
            case MutationType::UNTAP:
                previous_bool_value = card->is_tapped;
                card->is_tapped = false;
                break;
            case MutationType::POWER_MOD:
                previous_int_value = card->power_modifier;
                card->power_modifier += int_value;
                break;
            case MutationType::ADD_KEYWORD:
                card->added_keywords.push_back(str_value);
                break;
            case MutationType::REMOVE_KEYWORD:
                {
                    auto it = std::find(card->added_keywords.begin(), card->added_keywords.end(), str_value);
                    if (it != card->added_keywords.end()) {
                        card->added_keywords.erase(it);
                        previous_bool_value = true;
                    } else {
                        previous_bool_value = false;
                    }
                }
                break;
            default: break;
        }
    }

    void MutateCommand::invert(core::GameState& state) {
        // Special case for global modifiers
        if (mutation_type == MutationType::ADD_PASSIVE_EFFECT) {
            if (!state.passive_effects.empty()) {
                state.passive_effects.pop_back();
            }
            return;
        } else if (mutation_type == MutationType::ADD_COST_MODIFIER) {
            if (!state.active_modifiers.empty()) {
                state.active_modifiers.pop_back();
            }
            return;
        } else if (mutation_type == MutationType::ADD_PENDING_EFFECT) {
            if (!state.pending_effects.empty()) {
                state.pending_effects.pop_back();
            }
            return;
        }

        core::CardInstance* card = state.get_card_instance(target_instance_id);
        if (!card) return;

        switch(mutation_type) {
            case MutationType::TAP:
            case MutationType::UNTAP:
                card->is_tapped = previous_bool_value;
                break;
            case MutationType::POWER_MOD:
                card->power_modifier = previous_int_value;
                break;
            case MutationType::ADD_KEYWORD:
                {
                    auto it = std::find(card->added_keywords.rbegin(), card->added_keywords.rend(), str_value);
                    if (it != card->added_keywords.rend()) {
                        card->added_keywords.erase(std::next(it).base());
                    }
                }
                break;
            case MutationType::REMOVE_KEYWORD:
                if (previous_bool_value) {
                    card->added_keywords.push_back(str_value);
                }
                break;
            default: break;
        }
    }

    // --- AttachCommand ---

    void AttachCommand::execute(core::GameState& state) {
        // Stub implementation for compilation
        (void)state;
    }

    void AttachCommand::invert(core::GameState& state) {
        (void)state;
    }

    // --- FlowCommand ---

    void FlowCommand::execute(core::GameState& state) {
        switch(flow_type) {
            case FlowType::PHASE_CHANGE:
                previous_value = static_cast<int>(state.current_phase);
                state.current_phase = static_cast<core::Phase>(new_value);
                // Dispatch Phase Change Event?
                break;
            case FlowType::TURN_CHANGE:
                previous_value = state.turn_number;
                state.turn_number = new_value;
                break;
            case FlowType::SET_ATTACK_SOURCE:
                previous_value = state.current_attack.source_instance_id;
                state.current_attack.source_instance_id = new_value;

                // Dispatch ATTACK_INITIATE event
                // Only if setting a valid source (initiating attack)
                if (new_value != -1 && state.event_dispatcher) {
                    core::GameEvent evt;
                    evt.type = core::EventType::ATTACK_INITIATE;
                    evt.instance_id = new_value;
                    evt.card_id = 0; // Unknown without lookup
                    evt.player_id = state.active_player_id;

                    evt.context["instance_id"] = new_value;
                    state.event_dispatcher(evt);
                }
                break;
            case FlowType::SET_ATTACK_TARGET:
                previous_value = state.current_attack.target_instance_id;
                state.current_attack.target_instance_id = new_value;
                break;
            case FlowType::SET_ATTACK_PLAYER:
                previous_value = state.current_attack.target_player_id;
                state.current_attack.target_player_id = new_value;
                break;
            case FlowType::SET_ACTIVE_PLAYER:
                previous_value = state.active_player_id;
                state.active_player_id = new_value;
                break;
            default: break;
        }
    }

    void FlowCommand::invert(core::GameState& state) {
        switch(flow_type) {
            case FlowType::PHASE_CHANGE:
                state.current_phase = static_cast<core::Phase>(previous_value);
                break;
            case FlowType::TURN_CHANGE:
                state.turn_number = previous_value;
                break;
            case FlowType::SET_ATTACK_SOURCE:
                state.current_attack.source_instance_id = previous_value;
                break;
            case FlowType::SET_ATTACK_TARGET:
                state.current_attack.target_instance_id = previous_value;
                break;
            case FlowType::SET_ATTACK_PLAYER:
                state.current_attack.target_player_id = previous_value;
                break;
            case FlowType::SET_ACTIVE_PLAYER:
                state.active_player_id = previous_value;
                break;
            default: break;
        }
    }

    // --- QueryCommand ---

    void QueryCommand::execute(core::GameState& state) {
        state.waiting_for_user_input = true;

        core::GameState::QueryContext ctx;
        // Check if there was a previous query? No, just increment ID.
        // Assuming single query at a time for now.
        int current_id = state.pending_query.query_id;
        ctx.query_id = current_id + 1;
        ctx.query_type = query_type;
        ctx.valid_targets = valid_targets;
        ctx.params = params;

        state.pending_query = ctx;
    }

    void QueryCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = false;
        // No need to clear pending_query struct, just flag false.
    }

    // --- DecideCommand ---

    void DecideCommand::execute(core::GameState& state) {
        was_waiting = state.waiting_for_user_input;
        if (was_waiting) {
            previous_query = state.pending_query;
        }

        if (state.waiting_for_user_input && state.pending_query.query_id == query_id) {
            state.waiting_for_user_input = false;
        }
    }

    void DecideCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = was_waiting;
        if (previous_query) {
             state.pending_query = *previous_query;
        }
    }

    // --- DeclareReactionCommand ---

    void DeclareReactionCommand::execute(core::GameState& state) {
        previous_status = state.status;
        previous_stack = state.reaction_stack;

        if (state.reaction_stack.empty()) return;

        auto& window = state.reaction_stack.back();

        if (!pass) {
            if (reaction_index < 0 || reaction_index >= (int)window.candidates.size()) {
                return;
            }
            window.used_candidate_indices.push_back(reaction_index);

            const auto& candidate = window.candidates[reaction_index];

            if (candidate.type == dm::engine::systems::ReactionType::SHIELD_TRIGGER) {
                 core::PendingEffect eff(core::EffectType::TRIGGER_ABILITY, candidate.instance_id, candidate.player_id);
                 state.pending_effects.push_back(eff);
            }
            else if (candidate.type == dm::engine::systems::ReactionType::REVOLUTION_CHANGE) {
                core::PendingEffect eff(core::EffectType::TRIGGER_ABILITY, candidate.instance_id, candidate.player_id);
                state.pending_effects.push_back(eff);
            }
        }

        if (pass) {
            state.reaction_stack.pop_back();
            if (state.reaction_stack.empty()) {
                state.status = core::GameState::Status::PLAYING;
            }
        }
    }

    void DeclareReactionCommand::invert(core::GameState& state) {
        state.status = previous_status;
        state.reaction_stack = previous_stack;
    }

    // --- StatCommand ---

    void StatCommand::execute(core::GameState& state) {
        switch (stat) {
            case StatType::CARDS_DRAWN:
                previous_value = state.turn_stats.cards_drawn_this_turn;
                state.turn_stats.cards_drawn_this_turn += amount;
                break;
            case StatType::CARDS_DISCARDED:
                previous_value = state.turn_stats.cards_discarded_this_turn;
                state.turn_stats.cards_discarded_this_turn += amount;
                break;
            case StatType::CREATURES_PLAYED:
                previous_value = state.turn_stats.creatures_played_this_turn;
                state.turn_stats.creatures_played_this_turn += amount;
                break;
            case StatType::SPELLS_CAST:
                previous_value = state.turn_stats.spells_cast_this_turn;
                state.turn_stats.spells_cast_this_turn += amount;
                break;
        }
    }

    void StatCommand::invert(core::GameState& state) {
        // Just restore the snapshot for safety
        switch (stat) {
            case StatType::CARDS_DRAWN:
                state.turn_stats.cards_drawn_this_turn = previous_value;
                break;
            case StatType::CARDS_DISCARDED:
                state.turn_stats.cards_discarded_this_turn = previous_value;
                break;
            case StatType::CREATURES_PLAYED:
                state.turn_stats.creatures_played_this_turn = previous_value;
                break;
            case StatType::SPELLS_CAST:
                state.turn_stats.spells_cast_this_turn = previous_value;
                break;
        }
    }

    // --- GameResultCommand ---

    void GameResultCommand::execute(core::GameState& state) {
        previous_result = state.winner;
        state.winner = result;
    }

    void GameResultCommand::invert(core::GameState& state) {
        state.winner = previous_result;
    }

}
