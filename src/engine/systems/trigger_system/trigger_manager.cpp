#include "trigger_manager.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/effect_system.hpp" // For resolve_trigger (temporary linkage) or just common logic
#include "core/card_def.hpp"
#include <iostream>

namespace dm::engine::systems {

    using namespace core;

    void TriggerManager::subscribe(EventType type, EventCallback callback) {
        listeners[type].push_back(callback);
    }

    void TriggerManager::dispatch(const GameEvent& event, GameState& state) {
        // std::cout << "DEBUG: TriggerManager::dispatch " << (int)event.type << std::endl;
        auto it = listeners.find(event.type);
        if (it != listeners.end()) {
            std::vector<EventCallback> callbacks = it->second;
            for (const auto& callback : callbacks) {
                callback(event, state);
            }
        }
    }

    // Helper to map GameEvent to TriggerType
    // Returns TriggerType::NONE if no mapping exists
    static TriggerType map_event_to_trigger(const GameEvent& event) {
        if (event.type == EventType::ZONE_ENTER) {
            if (event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::BATTLE) {
                return TriggerType::ON_PLAY;
            }
            if (event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::HAND &&
                event.context.count("from_zone") && event.context.at("from_zone") == (int)Zone::DECK) {
                return TriggerType::ON_DRAW;
            }
            // Destruction Logic: Moved TO Graveyard FROM Battle Zone
             if (event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::GRAVEYARD &&
                 event.context.count("from_zone") && event.context.at("from_zone") == (int)Zone::BATTLE) {
                 return TriggerType::ON_DESTROY;
             }
        }
        if (event.type == EventType::ATTACK_INITIATE) {
            return TriggerType::ON_ATTACK;
        }
        if (event.type == EventType::BLOCK_INITIATE) {
             return TriggerType::ON_BLOCK;
        }
        if (event.type == EventType::SHIELD_BREAK) {
            return TriggerType::AT_BREAK_SHIELD;
        }
        if (event.type == EventType::PLAY_CARD) {
             if (event.context.count("is_spell") && event.context.at("is_spell") == 1) {
                 return TriggerType::ON_CAST_SPELL;
             }
        }
        return TriggerType::NONE;
    }

    void TriggerManager::check_triggers(const GameEvent& event, GameState& state,
                                        const std::map<CardID, CardDefinition>& card_db) {
        TriggerType trigger_type = map_event_to_trigger(event);
        if (trigger_type == TriggerType::NONE) return;

        // Standard self-trigger logic + iteration over active cards
        // Zones to check for potential listeners
        std::vector<Zone> zones_to_check = {Zone::BATTLE};

        for (PlayerID pid : {state.active_player_id, static_cast<PlayerID>(1 - state.active_player_id)}) {
            const auto& battle_zone = state.get_zone(pid, Zone::BATTLE);
            for (int instance_id : battle_zone) {
                if (instance_id < 0) continue;
                const auto* card_ptr = state.get_card_instance(instance_id);
                if (!card_ptr) continue;

                // Get Definition
                const CardDefinition* def = nullptr;
                if (card_db.count(card_ptr->card_id)) {
                    def = &card_db.at(card_ptr->card_id);
                }
                if (!def) continue;

                // Collect effects
                std::vector<EffectDef> active_effects;
                active_effects.insert(active_effects.end(), def->effects.begin(), def->effects.end());

                for (const auto& effect : active_effects) {
                    if (effect.trigger == trigger_type) {
                        bool condition_met = false;
                        TargetScope scope = effect.trigger_scope;

                        // Default Scope (NONE) => SELF (This Creature)
                        if (scope == TargetScope::NONE) {
                            if (event.instance_id == instance_id) {
                                condition_met = true;
                            }
                        } else {
                            // Expanded Scope Logic
                            PlayerID controller = state.card_owner_map[instance_id];
                            PlayerID event_player = event.player_id;

                            // Try to infer event player from instance if missing
                            if (event_player == 255 && event.instance_id != -1) {
                                if (event.instance_id < (int)state.card_owner_map.size())
                                    event_player = state.card_owner_map[event.instance_id];
                            }

                            bool player_match = false;
                            if (scope == TargetScope::PLAYER_SELF) {
                                if (event_player == controller) player_match = true;
                            } else if (scope == TargetScope::PLAYER_OPPONENT) {
                                if (event_player != 255 && event_player != controller) player_match = true;
                            } else if (scope == TargetScope::ALL_PLAYERS || scope == TargetScope::ALL_FILTERED) {
                                player_match = true;
                            } else if (scope == TargetScope::SELF) {
                                if (event.instance_id == instance_id) player_match = true;
                            }

                            if (player_match) {
                                // Check if trigger_filter has any actual filtering criteria
                                bool has_filter = !effect.trigger_filter.zones.empty() ||
                                                 !effect.trigger_filter.types.empty() ||
                                                 !effect.trigger_filter.civilizations.empty() ||
                                                 !effect.trigger_filter.races.empty() ||
                                                 effect.trigger_filter.owner.has_value() ||
                                                 effect.trigger_filter.min_cost.has_value() ||
                                                 effect.trigger_filter.max_cost.has_value() ||
                                                 effect.trigger_filter.min_power.has_value() ||
                                                 effect.trigger_filter.max_power.has_value() ||
                                                 effect.trigger_filter.is_tapped.has_value() ||
                                                 effect.trigger_filter.is_blocker.has_value() ||
                                                 effect.trigger_filter.is_evolution.has_value();
                                
                                if (has_filter) {
                                    const CardInstance* source_card = state.get_card_instance(event.instance_id);
                                    if (source_card && card_db.count(source_card->card_id)) {
                                         if (TargetUtils::is_valid_target(*source_card, card_db.at(source_card->card_id),
                                                                          effect.trigger_filter, state, controller, controller, false, nullptr)) {
                                              condition_met = true;
                                         }
                                    }
                                } else {
                                    condition_met = true;
                                }
                            }
                        }

                        if (condition_met) {
                            // Match!
                            PlayerID controller = state.card_owner_map[instance_id];
                            PendingEffect pending(EffectType::TRIGGER_ABILITY, instance_id, controller);
                            pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                            pending.effect_def = effect; // Copy effect
                            pending.optional = true;
                            pending.chain_depth = state.turn_stats.current_chain_depth + 1;

                            state.pending_effects.push_back(pending);
                        }
                    }
                }
            }
        }
    }

    bool TriggerManager::check_reactions(const GameEvent& event, GameState& state,
                                         const std::map<CardID, CardDefinition>& card_db) {
        std::vector<ReactionCandidate> candidates;

        // 1. Shield Trigger
        if (event.type == EventType::ZONE_ENTER) {
            if (event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::HAND &&
                event.context.count("from_zone") && event.context.at("from_zone") == (int)Zone::SHIELD) {

                int instance_id = event.context.at("instance_id");
                const CardInstance* card = state.get_card_instance(instance_id);
                if (card && card_db.count(card->card_id)) {
                    const auto& def = card_db.at(card->card_id);
                    if (def.keywords.shield_trigger) {
                        ReactionCandidate c;
                        c.card_id = card->card_id;
                        c.instance_id = instance_id;
                        c.player_id = state.card_owner_map[instance_id];
                        c.type = ReactionType::SHIELD_TRIGGER;
                        candidates.push_back(c);
                    }
                }
            }
        }

        // 2. Revolution Change
        if (event.type == EventType::ATTACK_INITIATE) {
            PlayerID att_pid = event.player_id;
            const Player& player = state.players[att_pid];
            int attacker_id = event.instance_id;
            const CardInstance* attacker = state.get_card_instance(attacker_id);

            if (attacker) {
                for (const auto& hand_card : player.hand) {
                    if (!card_db.count(hand_card.card_id)) continue;
                    const auto& def = card_db.at(hand_card.card_id);

                    if (def.keywords.revolution_change) {
                        if (def.revolution_change_condition.has_value()) {
                            bool match = TargetUtils::is_valid_target(*attacker, card_db.at(attacker->card_id),
                                                                    def.revolution_change_condition.value(),
                                                                    state, att_pid, att_pid);
                            if (match) {
                                ReactionCandidate c;
                                c.card_id = hand_card.card_id;
                                c.instance_id = hand_card.instance_id;
                                c.player_id = att_pid;
                                c.type = ReactionType::REVOLUTION_CHANGE;
                                candidates.push_back(c);
                            }
                        }
                    }
                }
            }
        }

        if (!candidates.empty()) {
            ReactionWindow window(candidates);
            state.reaction_stack.push_back(window);
            state.status = GameState::Status::WAITING_FOR_REACTION;
            return true;
        }
        return false;
    }

    void TriggerManager::clear() {
        listeners.clear();
    }

    void TriggerManager::setup_event_handling(core::GameState& state,
                                              std::shared_ptr<TriggerManager> trigger_manager,
                                              std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> card_db) {
        // Capture &state by pointer to allow mutable access inside lambda
        core::GameState* state_ptr = &state;

        state.event_dispatcher = [trigger_manager, card_db, state_ptr](const core::GameEvent& event) {
            if (!state_ptr) return; // Should not happen if lifecycle is managed correctly

            trigger_manager->dispatch(event, *state_ptr);
            if (card_db) {
                trigger_manager->check_triggers(event, *state_ptr, *card_db);
                trigger_manager->check_reactions(event, *state_ptr, *card_db);
            }
        };
    }

}
