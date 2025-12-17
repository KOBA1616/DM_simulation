#include "trigger_manager.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/generic_card_system.hpp" // For resolve_trigger (temporary linkage) or just common logic
#include "core/card_def.hpp"

namespace dm::engine::systems {

    using namespace core;

    void TriggerManager::subscribe(EventType type, EventCallback callback) {
        listeners[type].push_back(callback);
    }

    void TriggerManager::dispatch(const GameEvent& event, GameState& state) {
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
             // Context check for spell?
             // If we dispatch PLAY_CARD for both creatures and spells, we need to distinguish.
             // Usually ON_PLAY is for Creatures entering Battle Zone.
             // ON_CAST is for Spells.
             // We can check the card definition in check_triggers or here if we have context.
             // For now, let's assume PLAY_CARD maps to ON_CAST_SPELL if it's a spell, but map_event_to_trigger returns one type.
             // We'll handle dual mapping or context checks in check_triggers logic more robustly.
             // But for simple mapping:
             if (event.context.count("is_spell") && event.context.at("is_spell") == 1) {
                 return TriggerType::ON_CAST_SPELL;
             }
             // Creatures use ZONE_ENTER (Stack -> Battle) for ON_PLAY usually.
        }
        return TriggerType::NONE;
    }

    void TriggerManager::check_triggers(const GameEvent& event, GameState& state,
                                        const std::map<CardID, CardDefinition>& card_db) {
        // Phase 6 Engine Overhaul: Event-Driven Trigger System
        // This replaces the scattered logic in EffectResolver/GenericCardSystem.

        TriggerType trigger_type = map_event_to_trigger(event);
        if (trigger_type == TriggerType::NONE) return;

        // Determine which cards to check.
        // For standard triggers (ON_PLAY, ON_ATTACK, ON_DESTROY), the source is usually the event source itself.
        // But some effects are "Whenever ANOTHER creature..." (Passive/Triggered hybrid).
        // For MVP Phase 6 Step 1, we focus on "Self-Triggered" abilities.

        // However, generic triggers usually require scanning the board.
        // E.g. "Whenever a creature attacks" -> TriggerType::ON_ATTACK on non-active creatures?
        // Current JSON data primarily uses ON_ATTACK for the attacker itself.
        // Let's assume Self-Trigger for now, as that covers 90% of cases.
        // For "Whenever another...", that is usually handled by `GenericCardSystem` iterating ALL cards.

        // We replicate GenericCardSystem's iteration logic here to be the single source of truth.

        // Zones to check for potential listeners
        // Usually effects trigger from Battle Zone.
        // Some (Ninja Strike) trigger from Hand, but that is a Reaction, handled in check_reactions.
        std::vector<Zone> zones_to_check = {Zone::BATTLE};

        // Also check Effect Buffer? (e.g. for temporary effects) - Usually not needed for standard triggers.

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
                // Add metamorph/other conditional effects logic here if needed (GenericCardSystem handles this)
                // For direct access, we might miss dynamically granted effects.
                // Ideally, CardInstance should store granted triggers.

                for (const auto& effect : active_effects) {
                    if (effect.trigger == trigger_type) {
                        // Condition: Is this card the source of the event?
                        // "Self" triggers
                        if (event.source_id == instance_id) {
                            // Match!
                            // Add to Pending Effects
                            PlayerID controller = state.card_owner_map[instance_id];
                            PendingEffect pending(EffectType::TRIGGER_ABILITY, instance_id, controller);
                            pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                            pending.effect_def = effect; // Copy effect
                            pending.optional = true;
                            pending.chain_depth = state.turn_stats.current_chain_depth + 1;

                            state.pending_effects.push_back(pending);
                        }
                        // "Another" triggers logic would go here (check filter vs event.source_id)
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
            int attacker_id = event.source_id;
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

}
