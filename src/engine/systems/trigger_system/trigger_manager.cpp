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
        }
        if (event.type == EventType::ATTACK_INITIATE) {
            return TriggerType::ON_ATTACK;
        }
        if (event.type == EventType::ZONE_ENTER) {
             // Destruction Logic: Moved TO Graveyard FROM Battle Zone
             if (event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::GRAVEYARD &&
                 event.context.count("from_zone") && event.context.at("from_zone") == (int)Zone::BATTLE) {
                 return TriggerType::ON_DESTROY;
             }
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
                        bool scope_match = false;
                        PlayerID owner = state.card_owner_map[instance_id];

                        switch (effect.trigger_scope) {
                            case TargetScope::NONE:
                                // Default behavior: trigger on self
                                scope_match = (event.instance_id == instance_id);
                                break;
                            case TargetScope::SELF:
                            case TargetScope::PLAYER_SELF:
                                // Matches if event is caused by owner's card
                                scope_match = (event.player_id == owner);
                                break;
                            case TargetScope::PLAYER_OPPONENT:
                                // Matches if event is caused by opponent's card
                                scope_match = (event.player_id != owner);
                                break;
                            case TargetScope::ALL_PLAYERS:
                            case TargetScope::ALL_FILTERED:
                                scope_match = true;
                                break;
                            default:
                                scope_match = (event.instance_id == instance_id);
                                break;
                        }

                        if (scope_match) {
                            bool filter_match = true;

                            // Apply filter if specified (and we are not just defaulting to self)
                            // If scope is NONE, we usually skip filter, but the user requested filter support always.
                            // However, legacy cards have empty filters. TargetUtils::is_valid_target handles empty/default filters by returning true?
                            // Let's assume if it has filter criteria, we check it.

                            if (scope_match) {
                                // For triggers, the "target" of the filter is the card that caused the event.
                                const CardInstance* event_card = state.get_card_instance(event.instance_id);

                                // Some events might not have a card instance (e.g. Turn Start?), but map_event_to_trigger handles card events.
                                // For now, we only filter if we have a card.
                                if (event_card && card_db.count(event_card->card_id)) {
                                     // Check if filter is set?
                                     // We blindly pass it to TargetUtils.
                                     // Note: trigger_filter is FilterDef.
                                     // TargetUtils::is_valid_target checks if card matches filter.

                                     // We only check if there IS a filter (implicit or explicit).
                                     // FilterDef default has empty vectors etc.
                                     // TargetUtils handles default empty filter as "match all"?
                                     // Yes, usually.

                                     filter_match = TargetUtils::is_valid_target(*event_card, card_db.at(event_card->card_id), effect.trigger_filter, state, owner, owner);
                                }
                            }

                            if (filter_match) {
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
