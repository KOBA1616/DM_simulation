#include "commands.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_event.hpp"
#include <iostream>
#include <algorithm>

namespace dm::engine::game_command {

    // --- TransitionCommand ---

    void TransitionCommand::execute(core::GameState& state) {
        // Logic similar to ZoneUtils::move_card
        // But simplified for primitive operation

        // 1. Find card and remove from source zone
        core::Player& owner = state.players[owner_id];
        std::vector<core::CardInstance>* source_vec = nullptr;
        std::vector<core::CardInstance>* dest_vec = nullptr;

        // Helper to get vector
        auto get_vec = [&](core::Zone z) -> std::vector<core::CardInstance>* {
            switch(z) {
                case core::Zone::HAND: return &owner.hand;
                case core::Zone::MANA: return &owner.mana_zone;
                case core::Zone::BATTLE: return &owner.battle_zone;
                case core::Zone::GRAVEYARD: return &owner.graveyard;
                case core::Zone::SHIELD: return &owner.shield_zone;
                case core::Zone::DECK: return &owner.deck;
                case core::Zone::BUFFER: return &owner.effect_buffer;
                // Stack is global in current GameState, not per player.
                // However, TransitionCommand expects owner_id.
                // If Stack is global, we need to handle it separately or assume owner_id logic applies to stack items?
                // Currently stack_zone is std::vector<CardInstance> in GameState.
                default: return nullptr;
            }
        };

        if (from_zone == core::Zone::STACK) {
             source_vec = &state.stack_zone;
        } else {
             source_vec = get_vec(from_zone);
        }

        if (to_zone == core::Zone::STACK) {
             dest_vec = &state.stack_zone;
        } else {
             dest_vec = get_vec(to_zone);
        }

        if (!source_vec || !dest_vec) return; // Error

        // Find
        auto it = std::find_if(source_vec->begin(), source_vec->end(),
            [&](const core::CardInstance& c){ return c.instance_id == card_instance_id; });

        if (it == source_vec->end()) {
             return;
        }

        // Store original index for undo
        original_index = std::distance(source_vec->begin(), it);

        core::CardInstance card = *it;
        source_vec->erase(it);

        // Add to dest
        if (destination_index == -1 || destination_index >= (int)dest_vec->size()) {
            dest_vec->push_back(card);
        } else {
            dest_vec->insert(dest_vec->begin() + destination_index, card);
        }

        // Phase 6: Event Dispatch
        // We need to dispatch ZONE_ENTER or similar.
        if (state.event_dispatcher) {
            core::GameEvent evt(core::EventType::ZONE_ENTER, card_instance_id, -1, owner_id);
            // Context
            evt.context["instance_id"] = card_instance_id;
            evt.context["from_zone"] = static_cast<int>(from_zone);
            evt.context["to_zone"] = static_cast<int>(to_zone);
            evt.context["card_id"] = card.card_id;

            state.event_dispatcher(evt);
        }
    }

    void TransitionCommand::invert(core::GameState& state) {
        // Reverse operation
        // Move FROM to_zone BACK TO from_zone at original_index

        core::Player& owner = state.players[owner_id];
        std::vector<core::CardInstance>* source_vec = nullptr; // Note: Invert swaps source/dest
        std::vector<core::CardInstance>* dest_vec = nullptr;

        auto get_vec = [&](core::Zone z) -> std::vector<core::CardInstance>* {
            switch(z) {
                case core::Zone::HAND: return &owner.hand;
                case core::Zone::MANA: return &owner.mana_zone;
                case core::Zone::BATTLE: return &owner.battle_zone;
                case core::Zone::GRAVEYARD: return &owner.graveyard;
                case core::Zone::SHIELD: return &owner.shield_zone;
                case core::Zone::DECK: return &owner.deck;
                default: return nullptr;
            }
        };

        // Current location (where it was moved TO) is now the source
        if (to_zone == core::Zone::STACK) {
             source_vec = &state.stack_zone;
        } else {
             source_vec = get_vec(to_zone);
        }

        // Original location (where it came FROM) is now the dest
        if (from_zone == core::Zone::STACK) {
             dest_vec = &state.stack_zone;
        } else {
             dest_vec = get_vec(from_zone);
        }

        if (!source_vec || !dest_vec) return;

        // Find card in current location
        auto it = std::find_if(source_vec->begin(), source_vec->end(),
            [&](const core::CardInstance& c){ return c.instance_id == card_instance_id; });

        if (it == source_vec->end()) return;

        core::CardInstance card = *it;
        source_vec->erase(it);

        // Restore to original index
        if (original_index >= 0 && original_index <= (int)dest_vec->size()) {
            dest_vec->insert(dest_vec->begin() + original_index, card);
        } else {
            dest_vec->push_back(card);
        }
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
                previous_int_value = card->power_mod;
                card->power_mod += int_value;
                break;
            // TODO: Keywords
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
                card->power_mod = previous_int_value;
                break;
            default: break;
        }
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
                    core::GameEvent evt(core::EventType::ATTACK_INITIATE, new_value, -1, state.active_player_id);
                    evt.context["instance_id"] = new_value;
                    state.event_dispatcher(evt);
                }
                break;
            case FlowType::SET_ATTACK_TARGET:
                previous_value = state.current_attack.target_instance_id;
                state.current_attack.target_instance_id = new_value;
                break;
            case FlowType::SET_ATTACK_PLAYER:
                previous_value = state.current_attack.target_player;
                state.current_attack.target_player = new_value;
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
                state.current_attack.target_player = previous_value;
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
        ctx.query_id = state.pending_query ? state.pending_query->query_id + 1 : 1;
        ctx.query_type = query_type;
        ctx.valid_target_ids = valid_targets;
        ctx.params = params;

        state.pending_query = ctx;
    }

    void QueryCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = false;
        state.pending_query = std::nullopt;
    }

    // --- DecideCommand ---

    void DecideCommand::execute(core::GameState& state) {
        was_waiting = state.waiting_for_user_input;
        previous_query = state.pending_query;

        if (state.pending_query && state.pending_query->query_id == query_id) {
            state.waiting_for_user_input = false;
            state.pending_query = std::nullopt;
        }
    }

    void DecideCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = was_waiting;
        state.pending_query = previous_query;
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
