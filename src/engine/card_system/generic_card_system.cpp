#include "../effects/effect_resolver.hpp"
#include "generic_card_system.hpp"
#include "card_registry.hpp"
#include "target_utils.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    // Helper to find card instance
    static CardInstance* find_instance(GameState& game_state, int instance_id) {
        for (auto& p : game_state.players) {
            for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
        }
        // Also check effect buffer
        for (auto& c : game_state.effect_buffer) if (c.instance_id == instance_id) return &c;

        return nullptr;
    }

    // Helper to generate a new unique instance id
    static int generate_instance_id(GameState& game_state) {
        int max_id = -1;
        for (auto &p : game_state.players) {
            for (auto &c : p.hand) if (c.instance_id > max_id) max_id = c.instance_id;
            for (auto &c : p.battle_zone) if (c.instance_id > max_id) max_id = c.instance_id;
            for (auto &c : p.mana_zone) if (c.instance_id > max_id) max_id = c.instance_id;
            for (auto &c : p.shield_zone) if (c.instance_id > max_id) max_id = c.instance_id;
            for (auto &c : p.graveyard) if (c.instance_id > max_id) max_id = c.instance_id;
            for (auto &c : p.deck) if (c.instance_id > max_id) max_id = c.instance_id;
        }
        return max_id + 1;
    }

    void GenericCardSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id) {
        CardInstance* instance = find_instance(game_state, source_instance_id);
        if (!instance) return;

        const CardData* data = CardRegistry::get_card_data(instance->card_id);
        if (!data) return;

        for (const auto& effect : data->effects) {
            if (effect.trigger == trigger) {
                resolve_effect(game_state, effect, source_instance_id);
            }
        }
    }

    void GenericCardSystem::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id) {
        // Delegate to EffectResolver's new signature with context
        // But since this method is static and doesn't have the context, we call EffectResolver's static method?
        // Actually, GenericCardSystem::resolve_effect is often called from outside EffectResolver (e.g. initial trigger).
        // To support variables, we should probably start a fresh context or use EffectResolver.

        // Use EffectResolver::resolve_effect which initializes a context
        EffectResolver::resolve_effect(game_state, effect, source_instance_id, {});
    }

    // Forward declaration of private helper
    // static void resolve_mekraid_internal(GameState& game_state, int card_instance_id, PlayerID player_id);

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        if (!check_condition(game_state, effect.condition, source_instance_id)) return;

        for (const auto& action : effect.actions) {
            // If action expects target selection, use provided targets
            if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                // Only handle actions that operate on targets (DESTROY, RETURN_TO_HAND, TAP, UNTAP)
                if (action.type == EffectActionType::DESTROY) {
                    for (int tid : targets) {
                        for (auto &p : game_state.players) {
                            auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                            if (it != p.battle_zone.end()) {
                                p.graveyard.push_back(*it);
                                p.battle_zone.erase(it);
                                break;
                            }
                        }
                    }
                } else if (action.type == EffectActionType::RETURN_TO_HAND) {
                    for (int tid : targets) {
                        bool found = false;
                        // Check Battle Zones
                        for (auto &p : game_state.players) {
                            auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                            if (it != p.battle_zone.end()) {
                                p.hand.push_back(*it);
                                p.battle_zone.erase(it);
                                // Reset state
                                p.hand.back().is_tapped = false;
                                p.hand.back().summoning_sickness = true;
                                found = true;
                                break;
                            }
                        }
                        // Check Buffer
                        if (!found) {
                            auto it = std::find_if(game_state.effect_buffer.begin(), game_state.effect_buffer.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                            if (it != game_state.effect_buffer.end()) {
                                Player& active = game_state.get_active_player(); // Buffer usually belongs to active
                                active.hand.push_back(*it);
                                game_state.effect_buffer.erase(it);
                                active.hand.back().is_tapped = false;
                                active.hand.back().summoning_sickness = true;
                                found = true;
                            }
                        }
                    }
                } else if (action.type == EffectActionType::TAP) {
                    for (int tid : targets) {
                        CardInstance* inst = find_instance(game_state, tid);
                        if (inst) inst->is_tapped = true;
                    }
                } else if (action.type == EffectActionType::UNTAP) {
                    for (int tid : targets) {
                        CardInstance* inst = find_instance(game_state, tid);
                        if (inst) inst->is_tapped = false;
                    }
                } else if (action.type == EffectActionType::PLAY_FROM_BUFFER) {
                    // This handles PLAY_FROM_BUFFER with selected targets
                    Player& active = game_state.get_active_player();

                    for (int tid : targets) {
                         auto it = std::find_if(game_state.effect_buffer.begin(), game_state.effect_buffer.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });

                         if (it != game_state.effect_buffer.end()) {
                             // Do NOT remove from buffer yet.
                             // PLAY_CARD_INTERNAL will find it in buffer and move it to stack.
                             // Just queue the pending effect.
                             game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, tid, active.id);
                         }
                    }
                } else if (action.type == EffectActionType::SELECT_FROM_BUFFER) {
                     // No-op (acknowledgment that selection happened for this action)
                } else if (action.type == EffectActionType::COST_REFERENCE && action.str_val == "FINISH_HYPER_ENERGY") {
                    // Handle Hyper Energy Finalization
                    // 1. Tap the selected targets
                    for (int tid : targets) {
                        for (auto &p : game_state.players) {
                             auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (it != p.battle_zone.end()) {
                                 it->is_tapped = true;
                             }
                        }
                    }

                    // 2. Delegate Play Logic to EffectResolver
                    int taps = action.value1;
                    int reduction = taps * 2;
                    // Hyper Energy is typically used by the active player (owner of the card).
                    EffectResolver::resolve_play_from_stack(game_state, source_instance_id, reduction, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db);
                }
                // other action types could be added as needed
            } else {
                resolve_action(game_state, action, source_instance_id);
            }
        }
    }

    bool GenericCardSystem::check_condition(GameState& game_state, const ConditionDef& condition, int source_instance_id) {
        if (condition.type == "NONE") return true;
        // Implement other conditions like MANA_ARMED
        return true;
    }

    std::vector<int> GenericCardSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id, const EffectDef& continuation) {
        // Push a pending effect to ask for selection
        PlayerID controller = game_state.active_player_id; // Default

        PendingEffect pending(EffectType::NONE, source_instance_id, controller);
        pending.resolve_type = ResolveType::TARGET_SELECT;
        pending.filter = action.filter;

        // If filter zones are empty, check ActionDef source_zone/target_choice for legacy compatibility
        if (pending.filter.zones.empty()) {
             if (action.target_choice == "ALL_ENEMY") {
                 pending.filter.owner = "OPPONENT";
                 pending.filter.zones = {"BATTLE_ZONE"};
             }
             // Add more legacy mappings if needed
        }

        // Count
        if (action.filter.count.has_value()) {
            pending.num_targets_needed = action.filter.count.value();
        } else {
            pending.num_targets_needed = 1;
        }

        pending.optional = action.optional;

        // Store the continuation EffectDef for resumption
        pending.effect_def = continuation;

        game_state.pending_effects.push_back(pending);

        // Return empty. The logic will resume when 'resolve_effect_with_targets' is called
        return {};
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {

        // 1. Check if we need to select targets first
        // If we are here, it means resolve_effect called us directly for a non-selection action,
        // OR resolve_effect_with_targets called us for a non-selection action.
        // If an action requires selection, resolve_effect should have caught it and deferred.
        // BUT if resolve_action is called standalone (e.g. from tests or elsewhere), we need to handle it.
        if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
             EffectDef ed;
             ed.trigger = TriggerType::NONE;
             ed.condition = ConditionDef{"NONE", 0, ""};
             ed.actions = { action };
             select_targets(game_state, action, source_instance_id, ed);
             return;
        }

        Player& active = game_state.get_active_player();

        switch (action.type) {
            case EffectActionType::DRAW_CARD: {
                int count = action.value1 > 0 ? action.value1 : std::stoi(action.value.empty() ? "1" : action.value);
                for (int i = 0; i < count; ++i) {
                    if (active.deck.empty()) {
                        game_state.winner = (active.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                        return;
                    }
                    CardInstance c = active.deck.back();
                    active.deck.pop_back();
                    active.hand.push_back(c);
                    // Phase 5: Stats
                    game_state.turn_stats.cards_drawn_this_turn++;
                }
                break;
            }
            case EffectActionType::ADD_MANA: {
                int count = action.value1 > 0 ? action.value1 : std::stoi(action.value.empty() ? "1" : action.value);
                for (int i = 0; i < count; ++i) {
                    if (active.deck.empty()) break;
                    CardInstance c = active.deck.back();
                    active.deck.pop_back();
                    c.is_tapped = false; // Usually untapped
                    active.mana_zone.push_back(c);
                }
                break;
            }
            case EffectActionType::TAP: {
                 if (action.target_choice == "ALL_ENEMY") {
                     int enemy = 1 - game_state.active_player_id;
                     for (auto& c : game_state.players[enemy].battle_zone) {
                         c.is_tapped = true;
                     }
                 }
                 break;
            }
            case EffectActionType::UNTAP: {
                 if (action.target_choice == "ALL_SELF") {
                     int self = game_state.active_player_id;
                     for (auto& c : game_state.players[self].battle_zone) {
                         c.is_tapped = false;
                     }
                 }
                 break;
            }
            case EffectActionType::DESTROY: {
                 // Non-selection destroy? (e.g. self-destruct)
                 // Or fallback for legacy.
                 break;
            }
            case EffectActionType::RETURN_TO_HAND: {
                if (action.target_choice == "ALL_ENEMY") {
                     int enemy_idx = 1 - game_state.active_player_id;
                     auto& bz = game_state.players[enemy_idx].battle_zone;
                     for (auto& c : bz) {
                         game_state.players[enemy_idx].hand.push_back(c);
                         // Reset state
                         game_state.players[enemy_idx].hand.back().is_tapped = false;
                         game_state.players[enemy_idx].hand.back().summoning_sickness = true;
                         game_state.players[enemy_idx].hand.back().power_mod = 0;
                     }
                     bz.clear();
                }
                break;
            }
            case EffectActionType::SEARCH_DECK_BOTTOM: {
                int look = action.value1 > 0 ? action.value1 : 1;
                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (active.deck.empty()) break;
                    looked.push_back(active.deck.back());
                    active.deck.pop_back();
                }

                // Inline filter check
                auto inline_matches = [&](const CardInstance& ci, const FilterDef& f, int owner_id) -> bool {
                    const dm::core::CardData* cd = dm::engine::CardRegistry::get_card_data(ci.card_id);
                    if (!cd) return false;
                    if (!f.types.empty()) {
                        bool ok = false;
                        for (const auto &t : f.types) if (t == cd->type) { ok = true; break; }
                        if (!ok) return false;
                    }
                    if (!f.civilizations.empty()) {
                        bool ok = false;
                        for (const auto &civ : f.civilizations) if (civ == cd->civilization) { ok = true; break; }
                        if (!ok) return false;
                    }
                    if (f.min_power.has_value() && cd->power < f.min_power.value()) return false;
                    if (f.max_power.has_value() && cd->power > f.max_power.value()) return false;
                    if (!f.races.empty()) {
                        bool ok = false;
                        for (const auto &r : f.races) {
                            for (const auto &cr : cd->races) if (r == cr) { ok = true; break; }
                            if (ok) break;
                        }
                        if (!ok) return false;
                    }
                    if (f.is_tapped.has_value()) if (ci.is_tapped != f.is_tapped.value()) return false;
                    if (f.owner.has_value()) {
                        std::string o = f.owner.value();
                        if (o == "SELF" && owner_id != game_state.get_active_player().id) return false;
                        if (o == "OPPONENT" && owner_id == game_state.get_active_player().id) return false;
                    }
                    if (f.max_cost.has_value() && cd->cost > f.max_cost.value()) return false;
                    if (f.min_cost.has_value() && cd->cost < f.min_cost.value()) return false;
                    return true;
                };

                // Auto-select first matching card for simplicity (MVP)
                int chosen_idx = -1;
                for (size_t i = 0; i < looked.size(); ++i) {
                    if (inline_matches(looked[i], action.filter, active.id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    // Add chosen to hand
                    active.hand.push_back(looked[chosen_idx]);
                }

                // Remaining cards go to bottom of deck.
                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    active.deck.insert(active.deck.begin(), looked[i]);
                }
                break;
            }
            case EffectActionType::MEKRAID: {
                int look = action.value1 > 0 ? action.value1 : 3;
                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (active.deck.empty()) break;
                    looked.push_back(active.deck.back());
                    active.deck.pop_back();
                }

                auto inline_matches = [&](const CardInstance& ci, const FilterDef& f, int owner_id) -> bool {
                    const dm::core::CardData* cd = dm::engine::CardRegistry::get_card_data(ci.card_id);
                    if (!cd) return false;
                    // MEKRAID Logic: Filter by Race/Civ AND Cost <= max_cost (if specified in filter)
                    if (!f.types.empty()) {
                        bool ok = false;
                        for (const auto &t : f.types) if (t == cd->type) { ok = true; break; }
                        if (!ok) return false;
                    }
                    if (!f.civilizations.empty()) {
                        bool ok = false;
                        for (const auto &civ : f.civilizations) if (civ == cd->civilization) { ok = true; break; }
                        if (!ok) return false;
                    }
                    if (!f.races.empty()) {
                        bool ok = false;
                        for (const auto &r : f.races) {
                            for (const auto &cr : cd->races) if (r == cr) { ok = true; break; }
                            if (ok) break;
                        }
                        if (!ok) return false;
                    }
                    if (f.max_cost.has_value() && cd->cost > f.max_cost.value()) return false;

                    return true;
                };

                int chosen_idx = -1;
                for (size_t i = 0; i < looked.size(); ++i) {
                    if (inline_matches(looked[i], action.filter, active.id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    // Refactored to use PendingEffect (Stack) for proper gatekeeper/trigger handling
                    CardInstance card = looked[chosen_idx];

                    // Move chosen card to Buffer first so PendingEffect can find it
                    game_state.effect_buffer.push_back(card);

                    // Queue Internal Play
                    game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, card.instance_id, active.id);
                }

                // Rest to bottom
                // Note: The chosen card is NOT in 'looked' anymore? No, 'looked' is local copy.
                // We popped from deck.
                // If we put chosen into Buffer, we shouldn't put it back in deck.
                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    active.deck.insert(active.deck.begin(), looked[i]);
                }
                break;
            }
            case EffectActionType::LOOK_TO_BUFFER: {
                // Move top N cards from source_zone (default: DECK) to effect_buffer
                int count = action.value1 > 0 ? action.value1 : std::stoi(action.value.empty() ? "1" : action.value);

                std::vector<CardInstance>* source = nullptr;
                if (action.source_zone == "DECK" || action.source_zone.empty()) {
                    source = &active.deck;
                } else if (action.source_zone == "HAND") {
                    source = &active.hand;
                }
                // Add other zones if needed

                if (source) {
                    for (int i = 0; i < count; ++i) {
                        if (source->empty()) break;
                        CardInstance c = source->back();
                        source->pop_back();
                        // When moving to buffer, we usually reveal it (or at least it's known to the logic)
                        // is_face_down should be false unless it's a secret look?
                        // For Gachinko Judge, it's face up. For Search, it's known to owner (face up to them).
                        // Let's assume face up for now.
                        c.is_face_down = false;
                        game_state.effect_buffer.push_back(c);
                    }
                }
                break;
            }
            case EffectActionType::SELECT_FROM_BUFFER: {
                // Trigger target selection from buffer
                // If we reached here in resolve_action, it means scope was NOT TARGET_SELECT?
                // OR we are calling resolve_action for this specific logic?
                // But SELECT_FROM_BUFFER usually implies target selection.
                // If it was called via resolve_effect, it should have been caught by the check at top of loop.
                // If it was called via resolve_effect_with_targets, it's ignored/no-op?

                // If we are here, and scope IS TARGET_SELECT, we already returned at top of function.
                // If scope is NOT TARGET_SELECT, then maybe it's just a "Select All" or "Random"?
                // If so, we should handle it.
                break;
            }
            case EffectActionType::PLAY_FROM_BUFFER: {
                // Play cards designated by targets (if any) or ALL in buffer?
                // Usually used after SELECT_FROM_BUFFER, so targets are in `targets` argument of resolve_effect_with_targets
                // But here we are in resolve_action which is called if NO targets were needed OR after they are selected?
                // Wait, if SELECT_FROM_BUFFER was called, it created a PendingEffect.
                // When that resolves, it calls resolve_effect_with_targets.
                // resolve_effect_with_targets calls resolve_action for each action in the effect.

                // We need to know WHICH cards were selected.
                // The `targets` are passed to `resolve_effect_with_targets` but NOT `resolve_action`.
                // We need to change `resolve_action` signature or logic.
                // However, `resolve_effect_with_targets` handles `DESTROY`, `RETURN_TO_HAND` etc. by looking at targets.
                // It iterates targets and does something.
                // For `PLAY_FROM_BUFFER`, we need to do the same.

                // BUT `resolve_effect_with_targets` calls `resolve_action` in the `else` block!
                // So `PLAY_FROM_BUFFER` here won't know the targets.

                // We must handle `PLAY_FROM_BUFFER` inside `resolve_effect_with_targets`!
                break;
            }
            case EffectActionType::MOVE_BUFFER_TO_ZONE: {
                // Move remaining cards in buffer to destination
                std::vector<CardInstance>* dest = nullptr;
                if (action.destination_zone == "DECK_BOTTOM") {
                    dest = &active.deck;
                    // Usually we put them on bottom. `deck` is a vector, back is top.
                    // So insert at begin.
                    // Also random order? Or specific?
                    // "Put the rest on the bottom of your deck in any order."
                    // We just dump them.
                    for (auto& c : game_state.effect_buffer) {
                        active.deck.insert(active.deck.begin(), c);
                    }
                    game_state.effect_buffer.clear();
                }
                else if (action.destination_zone == "GRAVEYARD") {
                    dest = &active.graveyard;
                    for (auto& c : game_state.effect_buffer) {
                        active.graveyard.push_back(c);
                    }
                    game_state.effect_buffer.clear();
                }
                else if (action.destination_zone == "HAND") {
                    dest = &active.hand;
                     for (auto& c : game_state.effect_buffer) {
                        active.hand.push_back(c);
                    }
                    game_state.effect_buffer.clear();
                }
                // Add others as needed
                break;
            }
            default:
                break;
        }
    }

}
