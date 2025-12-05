#include "../effects/effect_resolver.hpp"
#include "generic_card_system.hpp"
#include "card_registry.hpp"
#include "target_utils.hpp"
#include <algorithm>
#include <iostream>
#include <set>

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

    // Helper to determine controller of an instance
    static PlayerID get_controller(const GameState& game_state, int instance_id) {
        for (const auto& p : game_state.players) {
            for (const auto& c : p.battle_zone) if (c.instance_id == instance_id) return p.id;
            for (const auto& c : p.hand) if (c.instance_id == instance_id) return p.id;
            for (const auto& c : p.mana_zone) if (c.instance_id == instance_id) return p.id;
            for (const auto& c : p.shield_zone) if (c.instance_id == instance_id) return p.id;
            for (const auto& c : p.graveyard) if (c.instance_id == instance_id) return p.id;
            for (const auto& c : p.deck) if (c.instance_id == instance_id) return p.id;
        }
        // If in buffer, assume active player (usually)
        // But for triggered effects, source might be anywhere.
        // If not found, return active player as fallback or throw?
        // Let's fallback to active player.
        return game_state.active_player_id;
    }

    void GenericCardSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id) {
        CardInstance* instance = find_instance(game_state, source_instance_id);
        if (!instance) {
            return;
        }

        const CardData* data = CardRegistry::get_card_data(instance->card_id);
        if (!data) {
            return;
        }

        bool triggered = false;
        for (const auto& effect : data->effects) {
            if (effect.trigger == trigger) {
                triggered = true;
                resolve_effect(game_state, effect, source_instance_id);
            }
        }
    }

    void GenericCardSystem::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id) {
        std::map<std::string, int> empty_context;
        resolve_effect_with_context(game_state, effect, source_instance_id, empty_context);
    }

    void GenericCardSystem::resolve_effect_with_context(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context) {
        if (!check_condition(game_state, effect.condition, source_instance_id)) {
            return;
        }

        // Pass context by reference to allow updates
        for (const auto& action : effect.actions) {
            resolve_action(game_state, action, source_instance_id, execution_context);
        }
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
         std::map<std::string, int> empty_context;
         resolve_effect_with_targets(game_state, effect, targets, source_instance_id, card_db, empty_context);
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context) {
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
                        // Check Mana Zone (rare, but possible if filters allow)
                        if (!found) {
                            for (auto &p : game_state.players) {
                                auto it = std::find_if(p.mana_zone.begin(), p.mana_zone.end(),
                                    [tid](const CardInstance& c){ return c.instance_id == tid; });
                                if (it != p.mana_zone.end()) {
                                    p.hand.push_back(*it);
                                    p.mana_zone.erase(it);
                                    p.hand.back().is_tapped = false;
                                    p.hand.back().summoning_sickness = true;
                                    found = true;
                                    break;
                                }
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
                    Player& active = game_state.get_active_player();
                    for (int tid : targets) {
                         auto it = std::find_if(game_state.effect_buffer.begin(), game_state.effect_buffer.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });

                         if (it != game_state.effect_buffer.end()) {
                             game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, tid, active.id);
                         }
                    }
                } else if (action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {
                    for (int tid : targets) {
                        for (auto &p : game_state.players) {
                             auto it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (it != p.shield_zone.end()) {
                                 p.graveyard.push_back(*it);
                                 p.shield_zone.erase(it);
                                 break;
                             }
                        }
                    }
                } else if (action.type == EffectActionType::SEARCH_DECK) {
                    Player& active = game_state.get_active_player();
                    for (int tid : targets) {
                         auto it = std::find_if(active.deck.begin(), active.deck.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });
                         if (it != active.deck.end()) {
                             active.hand.push_back(*it);
                             active.deck.erase(it);
                         }
                    }
                    std::shuffle(active.deck.begin(), active.deck.end(), game_state.rng);

                } else if (action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                    // Handle Targeted SEND_TO_DECK_BOTTOM
                    // Uses controller of the source card usually, or the player who owns the target.
                    // Since targets are instances, we find where they are.
                    for (int tid : targets) {
                        for (auto &p : game_state.players) {
                             // Check Hand
                             auto it = std::find_if(p.hand.begin(), p.hand.end(),
                                 [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (it != p.hand.end()) {
                                 p.deck.insert(p.deck.begin(), *it);
                                 p.hand.erase(it);
                                 continue;
                             }
                             // Check Battle Zone
                             auto bit = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                 [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (bit != p.battle_zone.end()) {
                                 p.deck.insert(p.deck.begin(), *bit);
                                 p.battle_zone.erase(bit);
                                 continue;
                             }
                        }
                    }

                } else if (action.type == EffectActionType::SELECT_FROM_BUFFER) {
                     // No-op
                } else if (action.type == EffectActionType::COST_REFERENCE && action.str_val == "FINISH_HYPER_ENERGY") {
                    for (int tid : targets) {
                        for (auto &p : game_state.players) {
                             auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (it != p.battle_zone.end()) {
                                 it->is_tapped = true;
                             }
                        }
                    }
                    int taps = action.value1;
                    int reduction = taps * 2;
                    EffectResolver::resolve_play_from_stack(game_state, source_instance_id, reduction, SpawnSource::HAND_SUMMON, game_state.active_player_id, card_db);
                }
            } else {
                resolve_action(game_state, action, source_instance_id, execution_context);
            }
        }
    }

    bool GenericCardSystem::check_condition(GameState& game_state, const ConditionDef& condition, int source_instance_id) {
        if (condition.type == "NONE") return true;
        return true;
    }

    std::vector<int> GenericCardSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id, const EffectDef& continuation, std::map<std::string, int>& execution_context) {
        PlayerID controller = get_controller(game_state, source_instance_id);

        PendingEffect pending(EffectType::NONE, source_instance_id, controller);
        pending.resolve_type = ResolveType::TARGET_SELECT;
        pending.filter = action.filter;

        if (pending.filter.zones.empty()) {
             if (action.target_choice == "ALL_ENEMY") {
                 pending.filter.owner = "OPPONENT";
                 pending.filter.zones = {"BATTLE_ZONE"};
             }
        }

        if (action.filter.count.has_value()) {
            pending.num_targets_needed = action.filter.count.value();
        } else {
            pending.num_targets_needed = 1;
        }

        // Variable Linking: Override count if input_value_key is present in context
        if (!action.input_value_key.empty()) {
            if (execution_context.count(action.input_value_key)) {
                pending.num_targets_needed = execution_context[action.input_value_key];
            }
        }

        pending.optional = action.optional;
        pending.effect_def = continuation;
        pending.execution_context = execution_context; // Save context

        game_state.pending_effects.push_back(pending);

        return {};
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {
        std::map<std::string, int> empty;
        resolve_action(game_state, action, source_instance_id, empty);
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) {

        // 1. Dispatch Target Selection
        if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
             EffectDef ed;
             ed.trigger = TriggerType::NONE;
             ed.condition = ConditionDef{"NONE", 0, ""};
             ed.actions = { action };
             select_targets(game_state, action, source_instance_id, ed, execution_context);
             return;
        }

        // 2. SEARCH_DECK
        if (action.type == EffectActionType::SEARCH_DECK) {
             EffectDef ed;
             ed.trigger = TriggerType::NONE;
             ed.condition = ConditionDef{"NONE", 0, ""};

             ActionDef move_act;
             move_act.type = EffectActionType::RETURN_TO_HAND;
             if (action.destination_zone == "MANA_ZONE") {
                 move_act.type = EffectActionType::SEND_TO_MANA;
             }

             ActionDef shuffle_act;
             shuffle_act.type = EffectActionType::SHUFFLE_DECK;

             ed.actions = { move_act, shuffle_act };

             ActionDef mod_action = action;
             if (mod_action.filter.zones.empty()) {
                 mod_action.filter.zones = {"DECK"};
             }
             if (!mod_action.filter.owner.has_value()) {
                 mod_action.filter.owner = "SELF";
             }
             select_targets(game_state, mod_action, source_instance_id, ed, execution_context);
             return;
        }

        PlayerID controller_id = get_controller(game_state, source_instance_id);
        Player& controller = game_state.players[controller_id];

        // Variable Linking: Resolve Value
        int val1 = action.value1;
        if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
            val1 = execution_context[action.input_value_key];
        }
        // Fallback to string value parsing if needed (legacy)
        if (val1 == 0 && !action.value.empty()) {
             try { val1 = std::stoi(action.value); } catch (...) {}
        }
        if (val1 == 0) val1 = 1; // Default to 1

        switch (action.type) {
            case EffectActionType::COUNT_CARDS: {
                int count = 0;
                // Use TargetUtils-like logic or manual scan
                // Filter is in action.filter
                const auto& f = action.filter;

                auto check_zone = [&](const std::vector<CardInstance>& zone, int owner_id) {
                     for (const auto& card : zone) {
                         const CardData* cd = CardRegistry::get_card_data(card.card_id);
                         if (!cd) continue;

                         if (!f.types.empty()) {
                             bool match = false;
                             for(auto& t : f.types) if(t == cd->type) match = true;
                             if(!match) continue;
                         }
                         if (!f.civilizations.empty()) {
                             bool match = false;
                             for(auto& c : f.civilizations) if(c == cd->civilization) match = true;
                             if(!match) continue;
                         }
                         if (!f.races.empty()) {
                             bool match = false;
                             for(auto& r : f.races) {
                                 for(auto& cr : cd->races) if(r == cr) match = true;
                             }
                             if(!match) continue;
                         }
                         if (f.owner.has_value()) {
                             if (f.owner == "SELF" && owner_id != controller_id) continue;
                             if (f.owner == "OPPONENT" && owner_id == controller_id) continue;
                         }

                         count++;
                     }
                };

                for (const auto& z : f.zones) {
                    if (z == "BATTLE_ZONE") {
                        check_zone(game_state.players[0].battle_zone, 0);
                        check_zone(game_state.players[1].battle_zone, 1);
                    } else if (z == "GRAVEYARD") {
                        check_zone(game_state.players[0].graveyard, 0);
                        check_zone(game_state.players[1].graveyard, 1);
                    } else if (z == "MANA_ZONE") {
                         check_zone(game_state.players[0].mana_zone, 0);
                         check_zone(game_state.players[1].mana_zone, 1);
                    } else if (z == "HAND") {
                         check_zone(game_state.players[0].hand, 0);
                         check_zone(game_state.players[1].hand, 1);
                    } else if (z == "SHIELD_ZONE") {
                         check_zone(game_state.players[0].shield_zone, 0);
                         check_zone(game_state.players[1].shield_zone, 1);
                    }
                }

                if (!action.output_value_key.empty()) {
                    execution_context[action.output_value_key] = count;
                }
                break;
            }
            case EffectActionType::GET_GAME_STAT: {
                int result = 0;
                if (action.str_val == "MANA_CIVILIZATION_COUNT") {
                    std::set<std::string> civs;
                    for (const auto& c : controller.mana_zone) {
                        const CardData* cd = CardRegistry::get_card_data(c.card_id);
                        if (cd) {
                             std::string s = cd->civilization;
                             size_t pos = 0;
                             while ((pos = s.find('/')) != std::string::npos) {
                                 civs.insert(s.substr(0, pos));
                                 s.erase(0, pos + 1);
                             }
                             if (!s.empty()) civs.insert(s);
                        }
                    }
                    result = (int)civs.size();
                } else if (action.str_val == "SHIELD_COUNT") {
                    result = (int)controller.shield_zone.size();
                } else if (action.str_val == "HAND_COUNT") {
                    result = (int)controller.hand.size();
                }

                if (!action.output_value_key.empty()) {
                    execution_context[action.output_value_key] = result;
                }
                break;
            }
            case EffectActionType::SHUFFLE_DECK: {
                std::shuffle(controller.deck.begin(), controller.deck.end(), game_state.rng);
                break;
            }
            case EffectActionType::ADD_SHIELD: {
                std::vector<CardInstance>* source = &controller.deck;
                if (action.source_zone == "HAND") source = &controller.hand;
                else if (action.source_zone == "GRAVEYARD") source = &controller.graveyard;

                if (!source->empty()) {
                    CardInstance c = source->back();
                    source->pop_back();
                    c.is_face_down = true;
                    controller.shield_zone.push_back(c);
                }
                break;
            }
            case EffectActionType::SEND_SHIELD_TO_GRAVE: {
                if (!controller.shield_zone.empty()) {
                    CardInstance c = controller.shield_zone.back();
                    controller.shield_zone.pop_back();
                    controller.graveyard.push_back(c);
                }
                break;
            }
            case EffectActionType::DRAW_CARD: {
                int count = val1;
                for (int i = 0; i < count; ++i) {
                    if (controller.deck.empty()) {
                        game_state.winner = (controller.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                        return;
                    }
                    CardInstance c = controller.deck.back();
                    controller.deck.pop_back();
                    controller.hand.push_back(c);
                    // Update turn stats only if controller is active player
                    if (controller.id == game_state.active_player_id) {
                        game_state.turn_stats.cards_drawn_this_turn++;
                    }
                }
                break;
            }
            case EffectActionType::ADD_MANA: {
                int count = val1;
                for (int i = 0; i < count; ++i) {
                    if (controller.deck.empty()) break;
                    CardInstance c = controller.deck.back();
                    controller.deck.pop_back();
                    c.is_tapped = false;
                    controller.mana_zone.push_back(c);
                }
                break;
            }
            case EffectActionType::TAP: {
                 if (action.target_choice == "ALL_ENEMY") {
                     int enemy = 1 - controller.id;
                     for (auto& c : game_state.players[enemy].battle_zone) {
                         c.is_tapped = true;
                     }
                 }
                 break;
            }
            case EffectActionType::UNTAP: {
                 if (action.target_choice == "ALL_SELF") {
                     for (auto& c : controller.battle_zone) {
                         c.is_tapped = false;
                     }
                 }
                 break;
            }
            case EffectActionType::DESTROY: {
                 break;
            }
            case EffectActionType::RETURN_TO_HAND: {
                if (action.target_choice == "ALL_ENEMY") {
                     int enemy_idx = 1 - controller.id;
                     auto& bz = game_state.players[enemy_idx].battle_zone;
                     for (auto& c : bz) {
                         game_state.players[enemy_idx].hand.push_back(c);
                         game_state.players[enemy_idx].hand.back().is_tapped = false;
                         game_state.players[enemy_idx].hand.back().summoning_sickness = true;
                         game_state.players[enemy_idx].hand.back().power_mod = 0;
                     }
                     bz.clear();
                }
                break;
            }
            case EffectActionType::SEARCH_DECK_BOTTOM: {
                int look = val1;
                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (controller.deck.empty()) break;
                    looked.push_back(controller.deck.back());
                    controller.deck.pop_back();
                }

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
                        if (o == "SELF" && owner_id != controller.id) return false;
                        if (o == "OPPONENT" && owner_id == controller.id) return false;
                    }
                    if (f.max_cost.has_value() && cd->cost > f.max_cost.value()) return false;
                    if (f.min_cost.has_value() && cd->cost < f.min_cost.value()) return false;
                    return true;
                };

                int chosen_idx = -1;
                for (size_t i = 0; i < looked.size(); ++i) {
                    if (inline_matches(looked[i], action.filter, controller.id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    controller.hand.push_back(looked[chosen_idx]);
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
                break;
            }
            case EffectActionType::MEKRAID: {
                int look = val1;
                if (look == 1) look = 3;

                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (controller.deck.empty()) break;
                    looked.push_back(controller.deck.back());
                    controller.deck.pop_back();
                }

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
                    if (inline_matches(looked[i], action.filter, controller.id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    CardInstance card = looked[chosen_idx];
                    game_state.effect_buffer.push_back(card);
                    game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, card.instance_id, controller.id);
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
                break;
            }
            case EffectActionType::LOOK_TO_BUFFER: {
                int count = val1;
                std::vector<CardInstance>* source = nullptr;
                if (action.source_zone == "DECK" || action.source_zone.empty()) {
                    source = &controller.deck;
                } else if (action.source_zone == "HAND") {
                    source = &controller.hand;
                }

                if (source) {
                    for (int i = 0; i < count; ++i) {
                        if (source->empty()) break;
                        CardInstance c = source->back();
                        source->pop_back();
                        c.is_face_down = false;
                        game_state.effect_buffer.push_back(c);
                    }
                }
                break;
            }
            case EffectActionType::MOVE_BUFFER_TO_ZONE: {
                std::vector<CardInstance>* dest = nullptr;
                if (action.destination_zone == "DECK_BOTTOM") {
                    dest = &controller.deck;
                    for (auto& c : game_state.effect_buffer) {
                        controller.deck.insert(controller.deck.begin(), c);
                    }
                    game_state.effect_buffer.clear();
                }
                else if (action.destination_zone == "GRAVEYARD") {
                    dest = &controller.graveyard;
                    for (auto& c : game_state.effect_buffer) {
                        controller.graveyard.push_back(c);
                    }
                    game_state.effect_buffer.clear();
                }
                else if (action.destination_zone == "HAND") {
                    dest = &controller.hand;
                     for (auto& c : game_state.effect_buffer) {
                        controller.hand.push_back(c);
                    }
                    game_state.effect_buffer.clear();
                }
                break;
            }
            default:
                break;
        }
    }

}
