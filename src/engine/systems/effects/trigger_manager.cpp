#include "trigger_manager.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/systems/effects/effect_system.hpp"
#include "engine/systems/effects/reaction_system.hpp"
#include "engine/systems/effects/reaction_window.hpp"
#include "core/card_def.hpp"
#include <iostream>

namespace {
using namespace dm::core;

static bool zone_name_matches(const std::string& name, int zone_value) {
    if (name == "BATTLE" || name == "BATTLE_ZONE") return zone_value == (int)Zone::BATTLE;
    if (name == "HAND") return zone_value == (int)Zone::HAND;
    if (name == "MANA" || name == "MANA_ZONE") return zone_value == (int)Zone::MANA;
    if (name == "SHIELD" || name == "SHIELD_ZONE") return zone_value == (int)Zone::SHIELD;
    if (name == "GRAVEYARD") return zone_value == (int)Zone::GRAVEYARD;
    if (name == "DECK") return zone_value == (int)Zone::DECK;
    return false;
}

static bool matches_trigger_descriptor(const TriggerDescriptor& td, const GameEvent& event) {
    // timing_mode: PRE は event.context 側で is_pre=1 か timing_mode=0 を要求
    const std::string timing = td.timing_mode.empty() ? "POST" : td.timing_mode;
    if (timing == "PRE") {
        const bool is_pre = (event.context.count("is_pre") && event.context.at("is_pre") == 1) ||
                            (event.context.count("timing_mode") && event.context.at("timing_mode") == 0);
        if (!is_pre) return false;
    } else {
        // POST は既定。明示PREマーカーがあるイベントではマッチしない。
        if (event.context.count("is_pre") && event.context.at("is_pre") == 1) return false;
    }

    if (!td.trigger_zones.empty()) {
        int to_zone = event.context.count("to_zone") ? event.context.at("to_zone") : -1;
        int from_zone = event.context.count("from_zone") ? event.context.at("from_zone") : -1;
        bool zone_ok = false;
        for (const auto& z : td.trigger_zones) {
            if (zone_name_matches(z, to_zone) || zone_name_matches(z, from_zone)) {
                zone_ok = true;
                break;
            }
        }
        if (!zone_ok) return false;
    }

    return true;
}

static bool already_pending_once(const GameState& state, const PendingEffect& pending) {
    // 再発防止: multiplicity=ONCE は同一pendingが重複積みされないよう保護する。
    for (const auto& p : state.pending_effects) {
        if (p.type != EffectType::TRIGGER_ABILITY) continue;
        if (p.source_instance_id != pending.source_instance_id) continue;
        if (p.controller != pending.controller) continue;
        if (p.resolve_type != pending.resolve_type) continue;
        nlohmann::json ja;
        nlohmann::json jb;
        dm::core::to_json(ja, p.effect_def);
        dm::core::to_json(jb, pending.effect_def);
        if (ja == jb) return true;
    }
    return false;
}
} // namespace

namespace dm::engine::systems {

    using namespace core;
    using namespace dm::engine::effects;

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
    // 再発防止: 新しいTriggerTypeを追加したら必ずここのマッピングも更新すること
    static TriggerType map_event_to_trigger(const GameEvent& event) {
        if (event.type == EventType::ZONE_ENTER) {
            auto to   = event.context.count("to_zone")   ? event.context.at("to_zone")   : -1;
            auto from = event.context.count("from_zone") ? event.context.at("from_zone") : -1;

            // バトルゾーン参入 → ON_PLAY
            if (to == (int)Zone::BATTLE) {
                return TriggerType::ON_PLAY;
            }
            // 山札→手札 = ドロー
            if (to == (int)Zone::HAND && from == (int)Zone::DECK) {
                return TriggerType::ON_DRAW;
            }
            // バトルゾーン→墓地 = 破壊（ON_DESTROY）
            if (to == (int)Zone::GRAVEYARD && from == (int)Zone::BATTLE) {
                return TriggerType::ON_DESTROY;
            }
            // 手札→墓地 = 捨て（ON_DISCARD）
            if (to == (int)Zone::GRAVEYARD && from == (int)Zone::HAND) {
                return TriggerType::ON_DISCARD;
            }
            // シールドゾーン追加
            if (to == (int)Zone::SHIELD) {
                return TriggerType::ON_SHIELD_ADD;
            }
        }
        // バトルゾーンからの離脱（ON_EXIT）— 破壊以外も含む広範なトリガー
        if (event.type == EventType::ZONE_LEAVE) {
            auto from = event.context.count("from_zone") ? event.context.at("from_zone") : -1;
            if (from == (int)Zone::BATTLE) {
                return TriggerType::ON_EXIT;
            }
        }
        if (event.type == EventType::ATTACK_INITIATE) {
            return TriggerType::ON_ATTACK;
        }
        if (event.type == EventType::BLOCK_INITIATE) {
            return TriggerType::ON_BLOCK;
        }
        if (event.type == EventType::BATTLE_WIN) {
            return TriggerType::ON_BATTLE_WIN;
        }
        if (event.type == EventType::BATTLE_LOSE) {
            return TriggerType::ON_BATTLE_LOSE;
        }
        if (event.type == EventType::SHIELD_BREAK) {
            return TriggerType::AT_BREAK_SHIELD;
        }
        if (event.type == EventType::PLAY_CARD) {
            if (event.context.count("is_spell") && event.context.at("is_spell") == 1) {
                return TriggerType::ON_CAST_SPELL;
            }
        }
        if (event.type == EventType::TURN_END) {
            return TriggerType::ON_TURN_END;
        }
        if (event.type == EventType::TAP_CARD) {
            return TriggerType::ON_TAP;
        }
        if (event.type == EventType::UNTAP_CARD) {
            return TriggerType::ON_UNTAP;
        }
        return TriggerType::NONE;
    }

    void TriggerManager::check_triggers(const GameEvent& event, GameState& state,
                                        const std::map<CardID, CardDefinition>& card_db) {
        TriggerType trigger_type = map_event_to_trigger(event);
        if (trigger_type == TriggerType::NONE) return;

        // 再発防止: ON_PLAYはpipelineのCHECK_CREATURE_ENTER_TRIGGERSで処理済み。
        // TransitionCommand→ZONE_ENTER経由でTriggerManagerが発火すると、
        // TriggerSystem::resolve_triggerと合わせてON_PLAYが二重発火する。
        // ON_CAST_SPELLもCHECK_SPELL_CAST_TRIGGERSで処理されるため同様にスキップ。
        // ON_EXIT/ON_DISCARDはZONE_LEAVEから派生するが、map_event_to_triggerがON_EXIT返す場合に
        // 二重処理が発生しないよう確認済み（ZONE_LEAVEは直接TriggerSystem経由では呼ばれない）。
        if (trigger_type == TriggerType::ON_PLAY ||
            trigger_type == TriggerType::ON_CAST_SPELL) {
            return;
        }

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
                    const TriggerDescriptor* td_ptr =
                        (effect.trigger_descriptor.has_value())
                            ? &effect.trigger_descriptor.value()
                            : nullptr;

                    // フェーズ2: TriggerDescriptor の trigger_list（OR結合）を優先して照合
                    bool trigger_match = false;
                    if (td_ptr && !td_ptr->trigger_list.empty()) {
                        for (const auto& t : td_ptr->trigger_list) {
                            if (t == trigger_type) { trigger_match = true; break; }
                        }
                    } else {
                        trigger_match = (effect.trigger == trigger_type);
                    }

                    if (trigger_match && td_ptr) {
                        trigger_match = matches_trigger_descriptor(*td_ptr, event);
                    }

                    if (trigger_match) {
                        bool condition_met = false;
                        TargetScope scope = effect.trigger_scope;

                        // Default Scope (NONE) => SELF (This Creature)
                        if (scope == TargetScope::NONE) {
                            if (event.instance_id == instance_id) {
                                condition_met = true;
                            }
                        } else {
                            // Expanded Scope Logic
                            PlayerID controller = state.get_card_owner(instance_id);
                            PlayerID event_player = event.player_id;

                            // Try to infer event player from instance if missing
                            if (event_player == 255 && event.instance_id != -1) {
                                if (event.instance_id < (int)state.card_owner_map.size())
                                    event_player = state.get_card_owner(event.instance_id);
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
                                         if (dm::engine::utils::TargetUtils::is_valid_target(*source_card, card_db.at(source_card->card_id),
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
                            PlayerID controller = state.get_card_owner(instance_id);
                            PendingEffect pending(EffectType::TRIGGER_ABILITY, instance_id, controller);
                            pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                            pending.effect_def = effect; // Copy effect
                            pending.optional = true;
                            pending.chain_depth = state.turn_stats.current_chain_depth + 1;

                            const bool once_only = td_ptr &&
                                (td_ptr->multiplicity.empty() || td_ptr->multiplicity == "ONCE");
                            if (!once_only || !already_pending_once(state, pending)) {
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
                        c.player_id = state.get_card_owner(instance_id);
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
                            bool match = dm::engine::utils::TargetUtils::is_valid_target(*attacker, card_db.at(attacker->card_id),
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

        // 3. Ninja Strike / Strike Back (via ReactionSystem)
        std::string trigger_event = "";
        if (event.type == EventType::ATTACK_INITIATE) trigger_event = "ON_ATTACK";
        else if (event.type == EventType::BLOCK_INITIATE) trigger_event = "ON_BLOCK";
        else if (event.type == EventType::ZONE_ENTER && event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::SHIELD) trigger_event = "ON_SHIELD_ADD";

        if (!trigger_event.empty()) {
             // Check both players for reactions
             for (PlayerID pid : {state.active_player_id, static_cast<PlayerID>(1 - state.active_player_id)}) {
                  auto extra = ReactionSystem::get_reaction_candidates(state, card_db, trigger_event, pid);
                  candidates.insert(candidates.end(), extra.begin(), extra.end());
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
