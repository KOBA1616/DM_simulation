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
        if (mutation_type == MutationType::ADD_GLOBAL_MODIFIER) {
            core::CostModifier mod;
            mod.reduction_amount = int_value;
            mod.condition_filter = filter;
            mod.turns_remaining = duration;
            mod.source_instance_id = source_id;
            mod.controller = controller;

            state.active_modifiers.push_back(mod);
            added_index = state.active_modifiers.size() - 1;
            return;
        }

        if (mutation_type == MutationType::ADD_PASSIVE_EFFECT) {
            core::PassiveEffect eff;
            // Map str_value or int_value to PassiveType?
            // Usually we rely on specialized parsing, but here MutateCommand is low level.
            // We need to assume str_value or a dedicated field carried the type.
            // But we don't have a `PassiveType` field in MutateCommand.
            // We can hack it: int_value holds the PassiveType enum value?
            // The `int_value` field is already used for 'value' (e.g. +1000 power).

            // Wait, MutateCommand definition has `int_value`.
            // For POWER, int_value is power amount.
            // For LOCK_SPELL, int_value is not used (or duration).
            // We need to know the PassiveType.
            // Let's assume we store PassiveType in `previous_int_value` temporarily or similar? No.
            // In C++, we can't easily extend without recompile.
            // We should have added `PassiveType` to MutateCommand or use `str_value` to parse?
            // Parsing string every time is slow.

            // Revisit implementation:
            // I should have added `int secondary_value` or similar.
            // Or just interpret `int_value` differently based on context?
            // No, for POWER_MOD we need `int_value`.

            // Let's rely on `str_value` for now since we have it.
            // e.g. "POWER", "LOCK_SPELL", "BLOCKER"
            // And map inside here.

            if (str_value == "POWER") eff.type = core::PassiveType::POWER_MODIFIER;
            else if (str_value == "LOCK_SPELL") eff.type = core::PassiveType::CANNOT_USE_SPELLS;
            else if (str_value == "BLOCKER") eff.type = core::PassiveType::BLOCKER_GRANT;
            else if (str_value == "KEYWORD") eff.type = core::PassiveType::KEYWORD_GRANT;
            // ... add others as needed

            eff.value = int_value; // Power value
            if (eff.type == core::PassiveType::KEYWORD_GRANT) {
                // For KEYWORD_GRANT, we need the keyword string.
                // But str_value is used for "KEYWORD".
                // We are missing a field.
                // Wait, MutateCommand has `str_value`.
                // If str_value="KEYWORD", where is the actual keyword?
                // Maybe encoded in str_value like "KEYWORD:SPEED_ATTACKER"?
            }

            // Better approach:
            // Use `int_value` for `PassiveType` cast if `int_value` is not needed for amount.
            // But POWER needs amount.
            // MutateCommand needs to be robust.

            // Let's encode type in `str_value` if it's not POWER.
            // If it is POWER, type is implied by context? No.

            // Actually, `ModifierHandler` previously did:
            // if str_val == "POWER": eff.type = POWER_MODIFIER
            // So relying on `str_value` to determine type is consistent with Handler.

            eff.target_filter = filter;
            eff.condition = condition;
            eff.source_instance_id = source_id;
            eff.controller = controller;
            eff.turns_remaining = duration;

            // For KEYWORD_GRANT, we need to store the keyword.
            // `ModifierHandler` didn't handle KEYWORD_GRANT, `GrantKeywordHandler` did.
            // `GrantKeywordHandler` used `ctx.action.str_val` as the keyword.
            // So if we use MutateCommand for `GrantKeywordHandler`, `str_value` should be the keyword.
            // But we need to know it IS a keyword grant.
            // Maybe we add `MutationType::ADD_PASSIVE_KEYWORD`?
            // Or `ADD_PASSIVE_EFFECT` implies generic, and we use `extra_info`.

            // Let's support `str_value` as the identifier.
            // If str_value is "POWER", we look at int_value.
            // If str_value is "BLOCKER", it's a keyword.
            // If str_value is "SPEED_ATTACKER", it's a keyword.
            // Wait, "POWER" is not a keyword.

            // Let's use `str_value` as the "Passive Key".
            // If `str_value` == "POWER_MODIFIER", `int_value` is used.
            // If `str_value` == "CANNOT_USE_SPELLS", etc.
            // If it's a keyword, we might prefix or just match?

            if (str_value == "POWER_MODIFIER" || str_value == "POWER") {
                eff.type = core::PassiveType::POWER_MODIFIER;
                eff.str_value = "";
            } else if (str_value == "CANNOT_USE_SPELLS" || str_value == "LOCK_SPELL") {
                eff.type = core::PassiveType::CANNOT_USE_SPELLS;
            } else {
                // Assume keyword grant
                eff.type = core::PassiveType::KEYWORD_GRANT;
                eff.str_value = str_value;
            }

            state.passive_effects.push_back(eff);
            added_index = state.passive_effects.size() - 1;
            return;
        }

        if (mutation_type == MutationType::REMOVE_GLOBAL_MODIFIER) {
             // Invert of ADD.
             // But we usually don't call REMOVE directly as a command,
             // unless we are implementing "Remove Modifier" effect?
             // Usually modifiers expire.
             // But `invert` needs it.
             return;
        }

        // Target-specific mutations
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
        if (mutation_type == MutationType::ADD_GLOBAL_MODIFIER) {
            if (added_index >= 0 && added_index < (int)state.active_modifiers.size()) {
                // Warning: This assumes LIFO or strict indexing.
                // If other modifiers were added after this one, `erase` shifts them.
                // If we are strictly undoing the last command, it is the last element.
                // But in a tree search, we might undo deep?
                // MCTS usually undoes in strict reverse order.
                // So `pop_back` should work if this was the last one added.
                // But let's use the index if we trust it wasn't reordered.
                // Actually, `std::vector` erase is safe if index is valid.
                state.active_modifiers.erase(state.active_modifiers.begin() + added_index);
            } else {
                 // Fallback: pop back if it matches?
                 if (!state.active_modifiers.empty()) {
                     state.active_modifiers.pop_back();
                 }
            }
            return;
        }

        if (mutation_type == MutationType::ADD_PASSIVE_EFFECT) {
            if (added_index >= 0 && added_index < (int)state.passive_effects.size()) {
                state.passive_effects.erase(state.passive_effects.begin() + added_index);
            } else {
                 if (!state.passive_effects.empty()) state.passive_effects.pop_back();
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
