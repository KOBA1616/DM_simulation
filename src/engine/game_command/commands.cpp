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

    // --- DeclareReactionCommand ---

    void DeclareReactionCommand::execute(core::GameState& state) {
        previous_status = state.status;
        previous_stack = state.reaction_stack;

        if (state.reaction_stack.empty()) return;

        auto& window = state.reaction_stack.back();

        // Validation: Verify candidate index
        if (!pass) {
            if (reaction_index < 0 || reaction_index >= (int)window.candidates.size()) {
                // Invalid index
                return;
            }
            // Mark candidate as used
            window.used_candidate_indices.push_back(reaction_index);

            // Execute Reaction Logic (The hard part)
            const auto& candidate = window.candidates[reaction_index];

            // NOTE: The Command implementation shouldn't execute complex logic directly?
            // Ideally it just queues a PendingEffect or performs state mutation.
            // S-Trigger -> Add "Play Card" Pending Effect.
            // Revolution Change -> Add "Swap and Play" Pending Effect or perform swap immediately.

            // Design Decision:
            // Since this is "Declare", we should queue the resolution.
            // S-Trigger resolution is: Play the card (free).
            // This is "Use Ability".

            if (candidate.type == dm::engine::systems::ReactionType::SHIELD_TRIGGER) {
                 // Add a PendingEffect to play this card
                 core::PendingEffect play_eff(core::EffectType::TRIGGER_ABILITY, candidate.instance_id, candidate.player_id);
                 // Wait, Phase 6 uses Instructions. We should queue an Instruction?
                 // Or we use the existing PendingEffect structure which works with the new engine wrapper.
                 // Let's use PendingEffect.

                 // EffectDef needs to be synthesized or looked up?
                 // Standard S-Trigger: Play this card.
                 // We can use a special "Resolve S-Trigger" effect type?
                 // Or just Queue "PLAY_CARD" action?
                 // But PLAY_CARD is an Action, not Effect.
                 // PendingEffect usually wraps an EffectDef.
                 // The "Trigger" here is the Shield Trigger capability.
                 // The "Effect" is "You may cast this spell for no cost".

                 // For now, let's create a PendingEffect that holds the instruction "PLAY_CARD".
                 // But PendingEffect stores EffectDef.
                 // We need to support 'Generated Actions' via PendingEffect.
                 // Let's assume there is a TRIGGER_RESOLUTION system that picks this up.
                 // Or we explicitly add `state.pending_effects.push_back(...)` here.

                 // Simplification for Phase 6 MVP:
                 // Queue a PendingEffect with source_id = candidate.card_id/instance_id.
                 // The engine loop handles "Resolving" the pending effect by playing it.
                 // But we need to signal "Play for free".
                 // Let's use a flag or cost modifier?
                 // S-Trigger is "Play for 0".

                 // For now, just mark the candidate as chosen.
                 // The *Caller* (Game Loop) will see the `DeclareReactionCommand` success
                 // and execute the logic.
                 // BUT `execute` is where state changes happen.

                 // If we strictly follow GameCommand pattern, THIS command must apply the change.
                 // So we must Queue the PendingEffect here.

                 core::PendingEffect eff(core::EffectType::TRIGGER_ABILITY, candidate.instance_id, candidate.player_id);
                 // We need to attach an EffectDef that says "Play Self".
                 // That's tricky.
                 // Instead, let's look at how ON_PLAY triggers are handled.
                 // They queue `TRIGGER_ABILITY`.

                 // Actually, Shield Trigger is "Use Card".
                 // We can queue a pending effect that, when resolved, calls `resolve_play_card`.

                 // For now, let's assume `TRIGGER_ABILITY` is sufficient and the Resolver knows
                 // that if a card in Hand triggers, and it has Shield Trigger, it means Play it.
                 // Or we introduce `EffectType::SHIELD_TRIGGER_RESOLVE`.
                 eff.type = core::EffectType::TRIGGER_ABILITY;
                 // We rely on the engine to interpret this correctly or we add metadata.

                 state.pending_effects.push_back(eff);
            }
            else if (candidate.type == dm::engine::systems::ReactionType::REVOLUTION_CHANGE) {
                // Perform Swap
                // 1. Return attacker to hand
                // 2. Put this card into battle zone
                // This is a complex atomic operation.
                // We should probably issue child commands?
                // But `execute` is synchronous.
                // We can mutate state directly here (since this IS the command).

                // Swap logic:
                // Find attacker.
                // Move attacker to Hand.
                // Move candidate to Battle Zone (tapped and attacking).
                // Copy state (tapped, attacking)? Revolution Change inherits state?
                // Yes, "switch" implies inheriting status.

                // For MVP, just queue a PendingEffect "REVOLUTION_CHANGE_RESOLVE".
                core::PendingEffect eff(core::EffectType::TRIGGER_ABILITY, candidate.instance_id, candidate.player_id);
                // We need to encode "Revolution Change" intent.
                // Maybe a custom EffectDef inside PendingEffect?
                state.pending_effects.push_back(eff);
            }
        }

        // Close logic
        // If Pass, or if single-use window (S-Trigger allows multiple usually).
        // Spec 6.1: "Wait until all pass".
        // But for S-Trigger, usually you declare one by one or batch?
        // Let's assume simplistic "One Declaration per step" or "Pass ends turn".

        // If Pass was declared, we are done with this player?
        // S-Trigger: You can use multiple triggers.
        // So passing means "I'm done with triggers".
        if (pass) {
            // Remove window or mark player as passed?
            // For now, simple: Pass = Done.
            state.reaction_stack.pop_back();
            if (state.reaction_stack.empty()) {
                state.status = core::GameState::Status::PLAYING;
            }
        } else {
            // Used one. Do we close?
            // S-Trigger: Keep open.
            // Revolution Change: Usually one per attack.
            // Ninja Strike: Keep open?

            // Let's assume for MVP: Using one action closes the window to process that action.
            // (Then maybe reopen later? No, S-Trigger processing is immediate).
            // Actually, usually triggers stack.
            // So we queue the effect and KEEP THE WINDOW OPEN.
            // Unless the rule says otherwise.
            // S-Trigger: You reveal all, then use.
            // We are in the "Use" phase.
            // Let's keep window open.
        }
    }

    void DeclareReactionCommand::invert(core::GameState& state) {
        state.status = previous_status;
        state.reaction_stack = previous_stack;
        // Also need to remove the pending effect added?
        // previous_stack restoration handles window state.
        // But `pending_effects` mutation needs inversion.
        // Since we didn't store index/count, this is risky.
        // Ideally we store "added_effect_index" member.
        // For MVP/Prototype, we skip perfect Undo for Reaction Execution logic
        // until we robustify it.
    }

}
