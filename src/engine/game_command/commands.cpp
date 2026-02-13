#include "commands.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_event.hpp"
#include "engine/systems/card/card_registry.hpp" // Added for G-Neo lookup
#include <iostream>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace dm::engine::game_command {

    // --- HistoryCommand ---

    void HistoryCommand::execute(core::GameState& state) {
        state.played_cards_history_this_game[player_id].push_back({card_id, turn_played});
        // Note: Global stats updates from on_card_play are not reversible here and are assumed
        // to be analytical only, or handled separately.
        // Turn stats should be handled by StatCommand if needed, but on_card_play does it too.
        // For now, we only sync the history vector which is part of state.
    }

    void HistoryCommand::invert(core::GameState& state) {
        auto& history = state.played_cards_history_this_game[player_id];
        if (!history.empty()) {
            history.pop_back();
        }
    }

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
                case core::Zone::STACK: return &owner.stack;
                default: return nullptr;
            }
        };

        if (from_zone == core::Zone::STACK) {
             // source_vec = nullptr; // Old
             source_vec = get_vec(core::Zone::STACK);
        } else {
             source_vec = get_vec(from_zone);
        }

        if (to_zone == core::Zone::STACK) {
             // dest_vec = nullptr; // Old
             dest_vec = get_vec(core::Zone::STACK);
        } else {
             dest_vec = get_vec(to_zone);
        }

        if (!source_vec || !dest_vec) return; // Error

        // Find
        auto it = std::find_if(source_vec->begin(), source_vec->end(),
            [&](const core::CardInstance& c){ return c.instance_id == card_instance_id; });

        if (it == source_vec->end()) {
             // Log missing instance for diagnostics
             try {
                 std::ofstream lout("logs/transition_debug.txt", std::ios::app);
                 if (lout) {
                     lout << "[Transition] MISSING in source: id=" << card_instance_id
                          << " owner=" << owner_id << " from=" << static_cast<int>(from_zone)
                          << " to=" << static_cast<int>(to_zone) << "\n";
                     lout.close();
                 }
             } catch(...) {}
             return;
        }

        // Store original index for undo
        original_index = std::distance(source_vec->begin(), it);

        core::CardInstance card = *it;

        // Ensure ownership is updated if moving between players (though TransitionCommand is usually intra-player,
        // the owner_id implies the destination owner context).
        // Since owner_id is passed to TransitionCommand, it effectively sets the controller.
        card.owner = owner_id;

        // Phase 6: Event Dispatch (ZONE_LEAVE)
        if (state.event_dispatcher) {
            core::GameEvent evt;
            evt.type = core::EventType::ZONE_LEAVE;
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

        // --- G-Neo Handling ---
        // Requirement: "G-Neo Creature... when it leaves the field, if there is a card under this creature,
        // instead of leaving, all cards under it are put in the graveyard."

        bool should_replace = false;
        if (from_zone == core::Zone::BATTLE && to_zone != core::Zone::BATTLE && !card.underlying_cards.empty()) {
             // Access Global Card Registry (Singleton)
             const auto& card_db = dm::engine::CardRegistry::get_all_definitions();
             if (card_db.count(card.card_id)) {
                 const auto& def = card_db.at(card.card_id);
                 if (def.keywords.g_neo) {
                     should_replace = true;
                 }
             }
        }

        if (should_replace) {
             g_neo_activated = true;

             // Store underlying cards for Undo
             moved_underlying_cards = card.underlying_cards;

             // Move underlying to Graveyard
             auto& grave = state.players[owner_id].graveyard;
             for (const auto& under : card.underlying_cards) {
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
             it->underlying_cards.clear();

             // Abort the move of the top card (Replacement Effect)
             return;
        }

        source_vec->erase(it);

        // Add to dest
        if (destination_index == -1 || destination_index >= (int)dest_vec->size()) {
            dest_vec->push_back(card);
        } else {
            dest_vec->insert(dest_vec->begin() + destination_index, card);
        }

        // Phase 6: Event Dispatch (ZONE_ENTER)
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
        // Log successful transition and owner zone counts for diagnostics
        try {
            std::ofstream lout("logs/transition_debug.txt", std::ios::app);
            if (lout) {
                lout << "[Transition] MOVED id=" << card_instance_id
                     << " from=" << static_cast<int>(from_zone)
                     << " to=" << static_cast<int>(to_zone)
                     << " owner=" << owner_id;

                // Append simple zone counts for quick inspection
                lout << " counts=[battle=" << owner.battle_zone.size()
                     << ",hand=" << owner.hand.size()
                     << ",mana=" << owner.mana_zone.size()
                     << ",shield=" << owner.shield_zone.size()
                     << ",deck=" << owner.deck.size()
                     << ",grave=" << owner.graveyard.size() << "]\n";

                lout.close();
            }
        } catch(...) {}
    }

    void TransitionCommand::invert(core::GameState& state) {
        // Reverse operation

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
                case core::Zone::BUFFER: return &owner.effect_buffer;
                case core::Zone::STACK: return &owner.stack;
                default: return nullptr;
            }
        };

        // --- G-Neo Undo Logic ---
        if (g_neo_activated) {
            // Restore underlying cards from Graveyard to Battle Zone (under the creature).
            // The creature (card_instance_id) is still in from_zone (BATTLE).

            dest_vec = get_vec(from_zone);
            if (!dest_vec) return;

            // Find the creature
            auto it = std::find_if(dest_vec->begin(), dest_vec->end(),
                [&](const core::CardInstance& c){ return c.instance_id == card_instance_id; });

            if (it == dest_vec->end()) return;

            // Restore underlying cards
            auto& grave = state.players[owner_id].graveyard;
            it->underlying_cards = moved_underlying_cards;

            // Remove from Graveyard
            // Note: We need to find and remove specific instances.
            // Assuming they are at the end of graveyard if pushed recently?
            // Safer to find by instance ID.
            for (const auto& moved_card : moved_underlying_cards) {
                auto git = std::find_if(grave.begin(), grave.end(),
                    [&](const core::CardInstance& c){ return c.instance_id == moved_card.instance_id; });
                if (git != grave.end()) {
                    grave.erase(git);
                }
            }

            // No need to move the top card, as it never moved.
            return;
        }

        // --- Standard Undo Logic ---

        // Current location (where it was moved TO) is now the source
        if (to_zone == core::Zone::STACK) {
             source_vec = get_vec(core::Zone::STACK);
        } else {
             source_vec = get_vec(to_zone);
        }

        // Original location (where it came FROM) is now the dest
        if (from_zone == core::Zone::STACK) {
             dest_vec = get_vec(core::Zone::STACK);
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

    // --- AddCardCommand ---

    void AddCardCommand::execute(core::GameState& state) {
        if(player_id >= state.players.size()) return;
        core::Player& p = state.players[player_id];
        std::vector<core::CardInstance>* vec = nullptr;

        switch(to_zone) {
            case core::Zone::HAND: vec = &p.hand; break;
            case core::Zone::MANA: vec = &p.mana_zone; break;
            case core::Zone::BATTLE: vec = &p.battle_zone; break;
            case core::Zone::SHIELD: vec = &p.shield_zone; break;
            case core::Zone::GRAVEYARD: vec = &p.graveyard; break;
            case core::Zone::DECK: vec = &p.deck; break;
            case core::Zone::BUFFER: vec = &p.effect_buffer; break;
            case core::Zone::STACK: vec = &p.stack; break;
            default: break;
        }

        if(vec) {
            // Update the owner of the card instance to match the destination player
            card.owner = player_id;

            vec->push_back(card);
            added_index = vec->size() - 1;

            // Also update card_owner_map if instance_id is within range
            if (card.instance_id >= 0) {
                 state.set_card_owner(card.instance_id, player_id);
            }
        }
    }

    void AddCardCommand::invert(core::GameState& state) {
        if(player_id >= state.players.size() || added_index < 0) return;
        core::Player& p = state.players[player_id];
        std::vector<core::CardInstance>* vec = nullptr;

        switch(to_zone) {
            case core::Zone::HAND: vec = &p.hand; break;
            case core::Zone::MANA: vec = &p.mana_zone; break;
            case core::Zone::BATTLE: vec = &p.battle_zone; break;
            case core::Zone::SHIELD: vec = &p.shield_zone; break;
            case core::Zone::GRAVEYARD: vec = &p.graveyard; break;
            case core::Zone::DECK: vec = &p.deck; break;
            case core::Zone::BUFFER: vec = &p.effect_buffer; break;
            case core::Zone::STACK: vec = &p.stack; break;
            default: break;
        }

        if(vec && added_index < (int)vec->size()) {
             // Verify instance ID matches to be safe?
             // For now just remove at index.
             // If other commands pushed after this, they should have been undone first.
             vec->erase(vec->begin() + added_index);
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
                // Event Dispatch
                if (!previous_bool_value && state.event_dispatcher) {
                    core::GameEvent evt;
                    evt.type = core::EventType::TAP_CARD;
                    evt.instance_id = target_instance_id;
                    evt.player_id = card->owner;
                    state.event_dispatcher(evt);
                }
                break;
            case MutationType::UNTAP:
                previous_bool_value = card->is_tapped;
                card->is_tapped = false;
                // Event Dispatch
                if (previous_bool_value && state.event_dispatcher) {
                    core::GameEvent evt;
                    evt.type = core::EventType::UNTAP_CARD;
                    evt.instance_id = target_instance_id;
                    evt.player_id = card->owner;
                    state.event_dispatcher(evt);
                }
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
                try {
                    std::filesystem::create_directories("logs");
                    std::ofstream lout("logs/phase_transitions.txt", std::ios::app);
                    if (lout) {
                        lout << "{\"event\":\"phase_change\",";
                        lout << "\"turn\":" << state.turn_number << ",";
                        lout << "\"player\":" << state.active_player_id << ",";
                        lout << "\"phase\":" << static_cast<int>(state.current_phase) << "}" << std::endl;
                        lout.close();
                    }
                } catch(...) {}
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
            case FlowType::SET_PLAYED_WITHOUT_MANA:
                previous_value = state.turn_stats.played_without_mana;
                state.turn_stats.played_without_mana = new_value;
                break;
            case FlowType::SET_MANA_CHARGED:
                previous_bool_value = state.turn_stats.mana_charged_by_player[state.active_player_id];
                state.turn_stats.mana_charged_by_player[state.active_player_id] = (new_value != 0);
                // DEBUG: Log the flag change
                try {
                    std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
                    if (ofs) {
                        ofs << "[FlowCommand] SET_MANA_CHARGED from " 
                            << (previous_bool_value ? "TRUE" : "FALSE")
                            << " to " << (state.turn_stats.mana_charged_by_player[state.active_player_id] ? "TRUE" : "FALSE")
                            << " (new_value=" << new_value << " pid=" << (int)state.active_player_id << ")\n";
                    }
                } catch(...) {}
                break;
            case FlowType::RESET_TURN_STATS:
                previous_turn_stats = state.turn_stats;
                state.turn_stats = core::TurnStats{};
                // DEBUG: Log turn stats reset
                try {
                    std::ofstream ofs("logs/reset_turn_stats_debug.txt", std::ios::app);
                    if (ofs) {
                        ofs << "[FlowCommand] RESET_TURN_STATS turn=" << state.turn_number
                            << " active_pid=" << (int)state.active_player_id
                            << " mana_charged_p0=" << (state.turn_stats.mana_charged_by_player[0]?"1":"0")
                            << " mana_charged_p1=" << (state.turn_stats.mana_charged_by_player[1]?"1":"0")
                            << "\n";
                    }
                } catch(...) {}
                break;
            case FlowType::CLEANUP_STEP:
                {
                     auto& mods = state.active_modifiers;
                     for (const auto& m : mods) {
                         if (m.turns_remaining == 1) removed_modifiers.push_back(m);
                     }
                     mods.erase(std::remove_if(mods.begin(), mods.end(), [](core::CostModifier& m) {
                         if (m.turns_remaining > 0) m.turns_remaining--;
                         return m.turns_remaining == 0;
                     }), mods.end());

                     auto& passives = state.passive_effects;
                     for (const auto& p : passives) {
                         if (p.turns_remaining == 1) removed_passives.push_back(p);
                     }
                     passives.erase(std::remove_if(passives.begin(), passives.end(), [](core::PassiveEffect& p) {
                         if (p.turns_remaining > 0) p.turns_remaining--;
                         return p.turns_remaining == 0;
                     }), passives.end());
                }
                break;
            case FlowType::SET_BLOCKING_CREATURE:
                previous_value = state.current_attack.blocking_creature_id;
                previous_bool_value = state.current_attack.blocked;
                state.current_attack.blocking_creature_id = new_value;

                if (new_value != -1) state.current_attack.blocked = true;
                else state.current_attack.blocked = false;
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
            case FlowType::SET_PLAYED_WITHOUT_MANA:
                state.turn_stats.played_without_mana = previous_value;
                break;
            case FlowType::SET_MANA_CHARGED:
                state.turn_stats.mana_charged_by_player[state.active_player_id] = previous_bool_value;
                break;
            case FlowType::RESET_TURN_STATS:
                state.turn_stats = previous_turn_stats;
                break;
            case FlowType::CLEANUP_STEP:
                // Restore removed modifiers and passives
                for (const auto& m : removed_modifiers) {
                    state.active_modifiers.push_back(m);
                }
                for (const auto& p : removed_passives) {
                    state.passive_effects.push_back(p);
                }
                break;
            case FlowType::SET_BLOCKING_CREATURE:
                state.current_attack.blocking_creature_id = previous_value;
                state.current_attack.blocked = previous_bool_value;
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
        ctx.options = options;

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
#include "commands.hpp"
#include <algorithm>
#include <random>

namespace dm::engine::game_command {

    void ShuffleCommand::execute(core::GameState& state) {
        if (player_id >= state.players.size()) return;
        auto& deck = state.players[player_id].deck;

        // Store original order for undo
        original_deck_order.clear();
        for (const auto& card : deck) {
            original_deck_order.push_back(card.instance_id);
        }

        std::shuffle(deck.begin(), deck.end(), state.rng);
    }

    void ShuffleCommand::invert(core::GameState& state) {
        if (player_id >= state.players.size()) return;
        auto& deck = state.players[player_id].deck;

        // Restore original order
        if (deck.size() != original_deck_order.size()) {
            // Something went wrong or deck size changed (unlikely within undo stack constraints)
            // Fallback: try to reconstruct as best as possible or do nothing.
            // Ideally this shouldn't happen if undo stack is consistent.
            return;
        }

        std::vector<core::CardInstance> restored_deck;
        restored_deck.reserve(deck.size());

        // Reconstruct the deck based on original_deck_order
        // We have the cards in 'deck' (shuffled), we need to reorder them.
        for (int id : original_deck_order) {
            auto it = std::find_if(deck.begin(), deck.end(), [id](const core::CardInstance& c) {
                return c.instance_id == id;
            });
            if (it != deck.end()) {
                restored_deck.push_back(*it);
            } else {
                 // Card missing?
            }
        }

        if (restored_deck.size() == deck.size()) {
            deck = std::move(restored_deck);
        }
    }

}
