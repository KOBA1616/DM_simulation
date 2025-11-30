#include "generic_card_system.hpp"
#include "card_registry.hpp"
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
        if (!check_condition(game_state, effect.condition, source_instance_id)) return;

        for (const auto& action : effect.actions) {
            resolve_action(game_state, action, source_instance_id);
        }
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id) {
        if (!check_condition(game_state, effect.condition, source_instance_id)) return;

        for (const auto& action : effect.actions) {
            // If action expects target selection, use provided targets
            if (action.scope == TargetScope::TARGET_SELECT) {
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
                        for (auto &p : game_state.players) {
                            auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                            if (it != p.battle_zone.end()) {
                                p.hand.push_back(*it);
                                p.battle_zone.erase(it);
                                break;
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

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {
        // Handle simple actions immediately
        // Handle complex actions (Target Selection) by pushing PendingEffect?
        // For now, let's implement simple auto-target actions.

        Player& active = game_state.get_active_player();
        // Player& opponent = game_state.get_non_active_player();

        switch (action.type) {
            case EffectActionType::DRAW_CARD: {
                int count = action.value1;
                for (int i = 0; i < count; ++i) {
                    if (active.deck.empty()) {
                        game_state.winner = (active.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                        return;
                    }
                    CardInstance c = active.deck.back();
                    active.deck.pop_back();
                    active.hand.push_back(c);
                }
                break;
            }
            case EffectActionType::ADD_MANA: {
                int count = action.value1;
                for (int i = 0; i < count; ++i) {
                    if (active.deck.empty()) break;
                    CardInstance c = active.deck.back();
                    active.deck.pop_back();
                    c.is_tapped = false; // Usually untapped
                    active.mana_zone.push_back(c);
                }
                break;
            }
            case EffectActionType::DESTROY: {
                // If action requires target selection, create a pending effect and return
                if (action.scope == TargetScope::TARGET_SELECT) {
                    PendingEffect pe(EffectType::DESTRUCTION, source_instance_id, game_state.get_active_player().id);
                    pe.num_targets_needed = action.filter.count.has_value() ? action.filter.count.value() : action.value1;
                    EffectDef ed;
                    ed.trigger = TriggerType::NONE;
                    ed.condition = ConditionDef{"NONE", 0, ""};
                    ed.actions = { action };
                    pe.effect_def = ed;
                    game_state.pending_effects.push_back(pe);
                    return;
                }

                // Otherwise, immediate selection and destroy
                auto targets = select_targets(game_state, action, source_instance_id);
                for (int tid : targets) {
                    // find the instance and move to graveyard
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
                break;
            }
            case EffectActionType::RETURN_TO_HAND: {
                auto targets = select_targets(game_state, action, source_instance_id);
                for (int tid : targets) {
                    for (auto &p : game_state.players) {
                        auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                            [tid](const CardInstance& c){ return c.instance_id == tid; });
                        if (it != p.battle_zone.end()) {
                            p.hand.push_back(*it);
                            p.battle_zone.erase(it);
                            break;
                        }
                    }
                }
                break;
            }
            case EffectActionType::TAP: {
                auto targets = select_targets(game_state, action, source_instance_id);
                for (int tid : targets) {
                    CardInstance* inst = find_instance(game_state, tid);
                    if (inst) inst->is_tapped = true;
                }
                break;
            }
            case EffectActionType::MODIFY_POWER: {
                // Modify power of target(s) by value1 (can be negative)
                if (action.scope == TargetScope::TARGET_SELECT) {
                    PendingEffect pe(EffectType::DESTRUCTION, source_instance_id, game_state.get_active_player().id);
                    pe.num_targets_needed = action.filter.count.has_value() ? action.filter.count.value() : action.value1;
                    EffectDef ed;
                    ed.trigger = TriggerType::NONE;
                    ed.condition = ConditionDef{"NONE", 0, ""};
                    ed.actions = { action };
                    pe.effect_def = ed;
                    game_state.pending_effects.push_back(pe);
                    return;
                }

                auto targets = select_targets(game_state, action, source_instance_id);
                for (int tid : targets) {
                    CardInstance* inst = find_instance(game_state, tid);
                    if (inst) {
                        inst->power_mod += action.value1;
                    }
                }
                break;
            }
            case EffectActionType::SUMMON_TOKEN: {
                int token_id = action.value1;
                int count = action.value2 > 0 ? action.value2 : 1;
                for (int i = 0; i < count; ++i) {
                    int iid = generate_instance_id(game_state);
                    CardInstance token((CardID)token_id, iid);
                    token.summoning_sickness = true;
                    game_state.get_active_player().battle_zone.push_back(token);
                }
                break;
            }
            case EffectActionType::LOOK_AND_ADD: {
                int look = action.value1 > 0 ? action.value1 : 1;
                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (active.deck.empty()) break;
                    looked.push_back(active.deck.back());
                    active.deck.pop_back();
                }

                // Find first matching card according to filter
                int chosen_idx = -1;
                // Inline filter check (similar to select_targets.matches_filter)
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
                    return true;
                };

                for (size_t i = 0; i < looked.size(); ++i) {
                    if (inline_matches(looked[i], action.filter, active.id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    // Add chosen to hand
                    active.hand.push_back(looked[chosen_idx]);
                    // Return others to deck preserving original order (top remains top)
                    for (int i = (int)looked.size() - 1; i >= 0; --i) {
                        if (i == chosen_idx) continue;
                        active.deck.push_back(looked[i]);
                    }
                } else {
                    // No match: return all to deck in reverse to preserve order
                    for (int i = (int)looked.size() - 1; i >= 0; --i) active.deck.push_back(looked[i]);
                }
                break;
            }
            case EffectActionType::UNTAP: {
                auto targets = select_targets(game_state, action, source_instance_id);
                for (int tid : targets) {
                    CardInstance* inst = find_instance(game_state, tid);
                    if (inst) inst->is_tapped = false;
                }
                break;
            }
            case EffectActionType::BREAK_SHIELD: {
                // Break opponent shields up to value1 times
                Player& opponent = game_state.get_non_active_player();
                int times = action.value1 > 0 ? action.value1 : 1;
                for (int i = 0; i < times; ++i) {
                    if (opponent.shield_zone.empty()) {
                        // Direct attack win condition handled elsewhere
                        break;
                    }
                    CardInstance shield = opponent.shield_zone.back();
                    opponent.shield_zone.pop_back();
                    opponent.hand.push_back(shield);

                    // Create pending effect for shield trigger
                    game_state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield.instance_id, opponent.id);
                }
                break;
            }
            default:
                break;
        }
    }

    std::vector<int> GenericCardSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id) {
        std::vector<int> out;
        // Basic implementation: support PLAYER_SELF, PLAYER_OPPONENT, ALL_FILTERED, TARGET_SELECT with count
        Player& active = game_state.get_active_player();
        Player& opponent = game_state.get_non_active_player();

        auto matches_filter = [&](const CardInstance& ci, const FilterDef& f, int owner_id) -> bool {
            const dm::core::CardData* cd = dm::engine::CardRegistry::get_card_data(ci.card_id);
            if (!cd) return false;

            // types
            if (!f.types.empty()) {
                bool ok = false;
                for (const auto &t : f.types) {
                    if (t == cd->type) { ok = true; break; }
                }
                if (!ok) return false;
            }

            // civilizations
            if (!f.civilizations.empty()) {
                bool ok = false;
                for (const auto &civ : f.civilizations) {
                    if (civ == cd->civilization) { ok = true; break; }
                }
                if (!ok) return false;
            }

            // power range
            if (f.min_power.has_value() && cd->power < f.min_power.value()) return false;
            if (f.max_power.has_value() && cd->power > f.max_power.value()) return false;

            // races (any overlap)
            if (!f.races.empty()) {
                bool ok = false;
                for (const auto &r : f.races) {
                    for (const auto &cr : cd->races) {
                        if (r == cr) { ok = true; break; }
                    }
                    if (ok) break;
                }
                if (!ok) return false;
            }

            // tapped state
            if (f.is_tapped.has_value()) {
                if (ci.is_tapped != f.is_tapped.value()) return false;
            }

            // owner filter: f.owner can be "SELF"/"OPPONENT"/"BOTH"
            if (f.owner.has_value()) {
                std::string o = f.owner.value();
                if (o == "SELF" && owner_id != game_state.get_active_player().id) return false;
                if (o == "OPPONENT" && owner_id == game_state.get_active_player().id) return false;
                // BOTH allows either
            }

            return true;
        };

        if (action.scope == TargetScope::PLAYER_SELF) {
            // select from active player's battle zone
            for (auto &c : active.battle_zone) {
                if (action.filter.count.has_value()) {
                    if ((int)out.size() >= action.filter.count.value()) break;
                }
                if (matches_filter(c, action.filter, active.id)) out.push_back(c.instance_id);
            }
        } else if (action.scope == TargetScope::PLAYER_OPPONENT) {
            for (auto &c : opponent.battle_zone) {
                if (action.filter.count.has_value()) {
                    if ((int)out.size() >= action.filter.count.value()) break;
                }
                if (matches_filter(c, action.filter, opponent.id)) out.push_back(c.instance_id);
            }
        } else if (action.scope == TargetScope::ALL_FILTERED) {
            for (auto &p : game_state.players) {
                for (auto &c : p.battle_zone) {
                    if (action.filter.count.has_value()) {
                        if ((int)out.size() >= action.filter.count.value()) break;
                    }
                    if (matches_filter(c, action.filter, p.id)) out.push_back(c.instance_id);
                }
            }
        } else if (action.scope == TargetScope::TARGET_SELECT) {
            // No auto-selection: rely on pending effect flow. For now, return empty.
        }

        return out;
    }

}
