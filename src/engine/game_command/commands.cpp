#include "commands.hpp"
#include "engine/utils/zone_utils.hpp"
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

        if (it == source_vec->end()) return; // Not found

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
                // Note: simple assignment works if only one command modified it.
                // But power_mod is additive. If multiple commands modified it,
                // we should subtract int_value instead of restoring previous absolute value?
                // `execute` did += int_value. `invert` should do -= int_value?
                // But `previous_int_value` stores the snapshot.
                // If we assume a linear history stack, restoring snapshot is fine.
                // But usually invert means "undo this delta".
                // Let's stick to snapshot restoration for now as it's safer against drift,
                // provided we undo in strict LIFO order.
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
                break;
            case FlowType::TURN_CHANGE:
                previous_value = state.turn_number;
                state.turn_number = new_value;
                break;
            case FlowType::SET_ATTACK_SOURCE:
                previous_value = state.current_attack.source_instance_id;
                state.current_attack.source_instance_id = new_value;
                break;
            case FlowType::SET_ATTACK_TARGET:
                previous_value = state.current_attack.target_instance_id;
                state.current_attack.target_instance_id = new_value;
                break;
            case FlowType::SET_ATTACK_PLAYER:
                previous_value = state.current_attack.target_player;
                state.current_attack.target_player = new_value;
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
        // Ideally we should restore the previous query if we are undoing a nested query,
        // but for now assume only one active query.
    }

    // --- DecideCommand ---

    void DecideCommand::execute(core::GameState& state) {
        // DECIDE resolves the query.
        // In a real flow, this would likely trigger a callback or resume the engine.
        // For the primitive, it just clears the waiting state and records the decision.
        // The engine loop is responsible for reading the decision and proceeding.

        was_waiting = state.waiting_for_user_input;
        previous_query = state.pending_query;

        // Verify ID?
        if (state.pending_query && state.pending_query->query_id == query_id) {
            state.waiting_for_user_input = false;
            state.pending_query = std::nullopt;

            // In a full event system, this would dispatch a "DECISION_MADE" event
            // or return control to the yielder.
            // For now, it just updates state to "Ready".
        }
    }

    void DecideCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = was_waiting;
        state.pending_query = previous_query;
    }

}
