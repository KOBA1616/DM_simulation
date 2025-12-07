#include "pending_strategy.hpp"
#include "../../card_system/target_utils.hpp"
#include "../../flow/reaction_system.hpp"
#include <algorithm>

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> PendingEffectStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;

        // Determine decision maker for pending effects
        // Usually active player, but if pending effect is controlled by opponent (e.g. they need to select target),
        // we must check who controls the pending effects.
        // The original logic checked if AP has any pending effects.

        PlayerID decision_maker = game_state.active_player_id;
        bool ap_has = false;
        for (const auto& eff : game_state.pending_effects) {
            if (eff.controller == game_state.active_player_id) { ap_has = true; break; }
        }
        decision_maker = ap_has ? game_state.active_player_id : (1 - game_state.active_player_id);

        // If the context active_player_id is strictly the turn player, we might need to be careful.
        // But here we locally determine `decision_maker`.

        for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
            const auto& eff = game_state.pending_effects[i];
            if (eff.controller != decision_maker) continue;

            if (eff.resolve_type == ResolveType::TARGET_SELECT) {
                if (eff.target_instance_ids.size() >= eff.num_targets_needed) {
                    Action resolve;
                    resolve.type = ActionType::RESOLVE_EFFECT;
                    resolve.slot_index = static_cast<int>(i);
                    actions.push_back(resolve);
                    continue;
                }

                const auto& filter = eff.filter;
                bool found_target = false;

                for (const auto& zone_str : filter.zones) {
                    std::vector<int> players_to_check;
                    if (filter.owner.has_value()) {
                        if (filter.owner.value() == "SELF") players_to_check.push_back(decision_maker);
                        else if (filter.owner.value() == "OPPONENT") players_to_check.push_back(1 - decision_maker);
                        else if (filter.owner.value() == "BOTH") {
                            players_to_check.push_back(decision_maker);
                            players_to_check.push_back(1 - decision_maker);
                        }
                    } else {
                        players_to_check.push_back(decision_maker);
                    }

                    for (int pid : players_to_check) {
                        const auto& target_player = game_state.players[pid];
                        const std::vector<CardInstance>* zone_ptr = nullptr;

                        if (zone_str == "BATTLE_ZONE") zone_ptr = &target_player.battle_zone;
                        else if (zone_str == "MANA_ZONE") zone_ptr = &target_player.mana_zone;
                        else if (zone_str == "HAND") zone_ptr = &target_player.hand;
                        else if (zone_str == "GRAVEYARD") zone_ptr = &target_player.graveyard;
                        else if (zone_str == "SHIELD_ZONE") zone_ptr = &target_player.shield_zone;
                        else if (zone_str == "DECK") zone_ptr = &target_player.deck;
                        else if (zone_str == "EFFECT_BUFFER" || zone_str == "BUFFER") {
                            zone_ptr = &game_state.effect_buffer;
                        }

                        if (zone_ptr) {
                            for (const auto& card : *zone_ptr) {
                                if (card_db.count(card.card_id) == 0) continue;
                                const auto& def = card_db.at(card.card_id);

                                if (TargetUtils::is_valid_target(card, def, filter, game_state, decision_maker, (PlayerID)pid)) {
                                    // Specific Check for Just Diver on SELECTION
                                    if (decision_maker != pid) { // Choosing opponent's card
                                        if (TargetUtils::is_protected_by_just_diver(card, def, game_state, decision_maker)) {
                                            continue; // Cannot select this target
                                        }
                                    }

                                    Action select;
                                    select.type = ActionType::SELECT_TARGET;
                                    select.target_instance_id = card.instance_id;
                                    select.slot_index = static_cast<int>(i);
                                    actions.push_back(select);
                                    found_target = true;
                                }
                            }
                        }
                    }
                }

                if (eff.optional || !found_target) {
                    Action pass;
                    pass.type = ActionType::PASS;
                    pass.slot_index = static_cast<int>(i);
                    actions.push_back(pass);
                }
            }
            else if (eff.type == EffectType::SHIELD_TRIGGER) {
                Action use;
                use.type = ActionType::USE_SHIELD_TRIGGER;
                use.source_instance_id = eff.source_instance_id;
                use.target_player = eff.controller;
                use.slot_index = static_cast<int>(i);
                actions.push_back(use);

                Action pass;
                pass.type = ActionType::RESOLVE_EFFECT;
                pass.slot_index = static_cast<int>(i);
                actions.push_back(pass);
            }
            else if (eff.type == EffectType::ON_ATTACK_FROM_HAND) {
                const Player& player = game_state.players[eff.controller];
                auto attacker_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(),
                    [&](const CardInstance& c) { return c.instance_id == eff.source_instance_id; });

                if (attacker_it != player.battle_zone.end()) {
                    for (size_t k = 0; k < player.hand.size(); ++k) {
                        const auto& card = player.hand[k];
                        if (card_db.count(card.card_id)) {
                            const auto& def = card_db.at(card.card_id);
                            if (def.keywords.revolution_change) {
                                Action use;
                                use.type = ActionType::USE_ABILITY;
                                use.source_instance_id = card.instance_id;
                                use.slot_index = static_cast<int>(k);
                                actions.push_back(use);
                            }
                        }
                    }
                }

                Action pass;
                pass.type = ActionType::RESOLVE_EFFECT;
                pass.slot_index = static_cast<int>(i);
                actions.push_back(pass);
            }
            else if (eff.type == EffectType::RESOLVE_BATTLE) {
                 Action action;
                 action.type = ActionType::RESOLVE_BATTLE;
                 action.slot_index = static_cast<int>(i);
                 actions.push_back(action);
            }
            else if (eff.type == EffectType::BREAK_SHIELD) {
                 Action action;
                 action.type = ActionType::BREAK_SHIELD;
                 action.slot_index = static_cast<int>(i);
                 actions.push_back(action);
            }
            else if (eff.type == EffectType::INTERNAL_PLAY || eff.type == EffectType::META_COUNTER) {
                Action action;
                action.type = ActionType::PLAY_CARD_INTERNAL;
                action.source_instance_id = eff.source_instance_id;
                action.target_player = eff.controller;
                action.slot_index = static_cast<int>(i);

                if (eff.type == EffectType::META_COUNTER) {
                    action.spawn_source = SpawnSource::HAND_SUMMON;
                } else {
                    action.spawn_source = SpawnSource::EFFECT_SUMMON;
                }
                actions.push_back(action);
            }
             else if (eff.type == EffectType::REACTION_WINDOW) {
                 const auto& player = game_state.players[eff.controller];
                 std::string event_type = eff.reaction_context.has_value() ? eff.reaction_context->trigger_event : "";

                 for (size_t k = 0; k < player.hand.size(); ++k) {
                     const auto& card = player.hand[k];
                     if (!card_db.count(card.card_id)) continue;
                     const auto& def = card_db.at(card.card_id);

                     bool legal = false;
                     for (const auto& r : def.reaction_abilities) {
                         bool event_match = (r.condition.trigger_event == event_type);
                         if (!event_match) {
                             if (r.condition.trigger_event == "ON_BLOCK_OR_ATTACK") {
                                 if (event_type == "ON_ATTACK" || event_type == "ON_BLOCK") {
                                     event_match = true;
                                 }
                             }
                         }
                         if (event_match) {
                             if (ReactionSystem::check_condition(game_state, r, card, eff.controller, card_db)) {
                                 legal = true;
                                 break;
                             }
                         }
                     }

                     if (legal) {
                         Action act;
                         act.type = ActionType::DECLARE_REACTION;
                         act.source_instance_id = card.instance_id;
                         act.target_player = eff.controller;
                         act.slot_index = static_cast<int>(k);
                         actions.push_back(act);
                     }
                 }

                 Action pass;
                 pass.type = ActionType::RESOLVE_EFFECT;
                 pass.slot_index = static_cast<int>(i);
                 actions.push_back(pass);
             }
            else if (eff.num_targets_needed > (int)eff.target_instance_ids.size()) {
                 if (actions.empty()) {
                     Action resolve;
                     resolve.type = ActionType::RESOLVE_EFFECT;
                     resolve.slot_index = static_cast<int>(i);
                     actions.push_back(resolve);
                 }
            }
            else {
                Action resolve;
                resolve.type = ActionType::RESOLVE_EFFECT;
                resolve.slot_index = static_cast<int>(i);
                actions.push_back(resolve);
            }
        }
        return actions;
    }

}
