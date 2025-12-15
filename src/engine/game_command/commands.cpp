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
                // Add support for BUFFER if needed, mapped to Zone::HYPER_SPATIAL or similar hack if not in enum
                // But Phase 00 defines 5 primitives.
                // Currently Zone enum does not have BUFFER.
                // Assuming BUFFER is handled via separate command or Zone enum extension.
                // For now, standard zones.
                default: return nullptr;
            }
        };

        source_vec = get_vec(from_zone);
        dest_vec = get_vec(to_zone);

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
        source_vec = get_vec(to_zone);
        // Original location (where it came FROM) is now the dest
        dest_vec = get_vec(from_zone);

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
        // Special case for Global Mutations (target_instance_id usually ignored or -1)
        if (mutation_type == MutationType::ADD_MODIFIER) {
            if (modifier_payload.has_value()) {
                added_index = state.active_modifiers.size();
                state.active_modifiers.push_back(modifier_payload.value());
            }
            return;
        }
        if (mutation_type == MutationType::ADD_PASSIVE) {
            if (passive_payload.has_value()) {
                added_index = state.passive_effects.size();
                state.passive_effects.push_back(passive_payload.value());
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
        // Special case for Global Mutations
        if (mutation_type == MutationType::ADD_MODIFIER) {
            if (added_index != -1 && added_index < (int)state.active_modifiers.size()) {
                // If we undo in strict LIFO order, it should be the last one.
                // But to be safe if added_index is valid, we check.
                // Since this vector is append-only usually (except expiration),
                // and undo happens within the turn before expiration...
                // Using pop_back() is the standard "Undo Push".
                state.active_modifiers.pop_back();
            }
            return;
        }
        if (mutation_type == MutationType::ADD_PASSIVE) {
            if (added_index != -1 && added_index < (int)state.passive_effects.size()) {
                state.passive_effects.pop_back();
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
                break;
            case FlowType::TURN_CHANGE:
                previous_value = state.turn_number;
                state.turn_number = new_value;
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
