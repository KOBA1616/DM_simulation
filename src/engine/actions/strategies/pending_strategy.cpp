#include "pending_strategy.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/flow/reaction_system.hpp"
#include <algorithm>
#include <vector>

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> PendingEffectStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;

        PlayerID decision_maker = game_state.active_player_id;
        bool ap_has = false;
        for (const auto& eff : game_state.pending_effects) {
            if (eff.controller == game_state.active_player_id) { ap_has = true; break; }
        }
        decision_maker = ap_has ? game_state.active_player_id : (1 - game_state.active_player_id);

        // Step 2-1: Strict Spell Priority Logic
        /*
        bool has_spell_effect = false;
        for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
            const auto& eff = game_state.pending_effects[i];
            if (eff.controller != decision_maker) continue;

            bool is_spell = false;
            if (eff.type == EffectType::SHIELD_TRIGGER) {
                 is_spell = true;
            } else if (eff.type == EffectType::INTERNAL_PLAY) {
                 const CardInstance* card = game_state.get_card_instance(eff.source_instance_id);
                 if (card && card_db.count(card->card_id) && card_db.at(card->card_id).type == CardType::SPELL) {
                     is_spell = true;
                 }
            } else {
                 const CardInstance* card = game_state.get_card_instance(eff.source_instance_id);
                 if (card && card_db.count(card->card_id) && card_db.at(card->card_id).type == CardType::SPELL) {
                     is_spell = true;
                 }
            }
            if (is_spell) {
                has_spell_effect = true;
                break;
            }
        }
        */

        std::vector<size_t> spell_indices;
        std::vector<size_t> other_indices;

        for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
            const auto& eff = game_state.pending_effects[i];
            if (eff.controller != decision_maker) continue;

            bool is_spell = false;
            if (eff.type == EffectType::SHIELD_TRIGGER) {
                 is_spell = true;
            } else {
                 const CardInstance* card = game_state.get_card_instance(eff.source_instance_id);
                 if (card && card_db.count(card->card_id) && card_db.at(card->card_id).type == CardType::SPELL) {
                     is_spell = true;
                 }
            }

            if (is_spell) {
                spell_indices.push_back(i);
            } else {
                other_indices.push_back(i);
            }
        }

        const std::vector<size_t>& active_indices = (!spell_indices.empty()) ? spell_indices : other_indices;

        for (size_t i : active_indices) {
            const auto& eff = game_state.pending_effects[i];

            if (eff.resolve_type == ResolveType::TARGET_SELECT) {
                if (eff.target_instance_ids.size() >= (size_t)eff.num_targets_needed) {
                    Action resolve;
                    resolve.type = PlayerIntent::RESOLVE_EFFECT;
                    resolve.slot_index = static_cast<int>(i);
                    actions.push_back(resolve);
                    continue;
                }

                const auto& filter = eff.filter;
                bool found_target = false;

                // Collect valid candidates for sorting/filtering
                struct Candidate {
                    CardInstance card;
                    const CardDefinition* def;
                };
                std::vector<Candidate> candidates;

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
                            // Target the buffer of the player being checked in the loop (pid)
                            // Usually we filter by Owner="SELF" so we check decision_maker's buffer.
                            zone_ptr = &game_state.players[pid].effect_buffer;
                        }

                        if (zone_ptr) {
                            for (const auto& card : *zone_ptr) {
                                if (card_db.count(card.card_id) == 0) continue;
                                const auto& def = card_db.at(card.card_id);

                                // Check Top Card (Element)
                                if (TargetUtils::is_valid_target(card, def, filter, game_state, decision_maker, (PlayerID)pid, false, &eff.execution_context)) {
                                    bool protected_by_jd = false;
                                    if (decision_maker != pid) {
                                        if (TargetUtils::is_protected_by_just_diver(card, def, game_state, decision_maker)) {
                                            protected_by_jd = true;
                                        }
                                    }
                                    if (!protected_by_jd) {
                                        candidates.push_back({card, &def});
                                    }
                                }

                                // Check Underlying Cards (Card Selection Mode)
                                if (filter.is_card_designation.has_value() && filter.is_card_designation.value()) {
                                    for (const auto& under : card.underlying_cards) {
                                        if (card_db.count(under.card_id) == 0) continue;
                                        const auto& under_def = card_db.at(under.card_id);
                                        // Note: Underlying cards are checked independently
                                        if (TargetUtils::is_valid_target(under, under_def, filter, game_state, decision_maker, (PlayerID)pid, false, &eff.execution_context)) {
                                            candidates.push_back({under, &under_def});
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Step 3-1: Apply Selection Mode (MIN/MAX)
                if (!candidates.empty() && filter.selection_mode.has_value() && filter.selection_sort_key.has_value()) {
                    std::string mode = filter.selection_mode.value();
                    std::string key = filter.selection_sort_key.value();

                    if (mode == "MIN" || mode == "MAX") {
                        // Sort
                        std::sort(candidates.begin(), candidates.end(), [&](const Candidate& a, const Candidate& b) {
                            int va = 0, vb = 0;
                            if (key == "COST") {
                                va = a.def->cost; vb = b.def->cost;
                            } else if (key == "POWER") {
                                va = a.def->power; vb = b.def->power;
                            }

                            if (mode == "MIN") return va < vb;
                            else return va > vb;
                        });

                        // Keep only the best value (handle ties)
                        int best_val = (key == "COST") ? candidates[0].def->cost : candidates[0].def->power;

                        std::vector<Candidate> best_candidates;
                        for (const auto& c : candidates) {
                            int val = (key == "COST") ? c.def->cost : c.def->power;
                            if (val == best_val) best_candidates.push_back(c);
                            else break; // Sorted, so we can stop
                        }
                        candidates = best_candidates;
                    }
                }

                // Generate actions for remaining candidates
                for (const auto& cand : candidates) {
                    Action select;
                    select.type = PlayerIntent::SELECT_TARGET;
                    select.target_instance_id = cand.card.instance_id;
                    select.slot_index = static_cast<int>(i);
                    actions.push_back(select);
                    found_target = true;
                }

                if (eff.optional || !found_target) {
                    Action pass;
                    pass.type = PlayerIntent::PASS;
                    pass.slot_index = static_cast<int>(i);
                    actions.push_back(pass);
                }
            }
            else if (eff.type == EffectType::SHIELD_TRIGGER) {
                Action use;
                use.type = PlayerIntent::USE_SHIELD_TRIGGER;
                use.source_instance_id = eff.source_instance_id;
                use.target_player = eff.controller;
                use.slot_index = static_cast<int>(i);
                actions.push_back(use);

                Action pass;
                pass.type = PlayerIntent::RESOLVE_EFFECT;
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
                                // Validate Condition
                                bool valid_condition = true;
                                if (def.revolution_change_condition.has_value()) {
                                    valid_condition = TargetUtils::is_valid_target(*attacker_it, card_db.at(attacker_it->card_id), def.revolution_change_condition.value(), game_state, eff.controller, eff.controller);
                                }

                                if (valid_condition) {
                                    Action use;
                                    use.type = PlayerIntent::USE_ABILITY;
                                    use.source_instance_id = card.instance_id;
                                    use.target_instance_id = attacker_it->instance_id; // Set target to attacker
                                    use.slot_index = static_cast<int>(k);
                                    actions.push_back(use);
                                }
                            }
                        }
                    }
                }

                Action pass;
                pass.type = PlayerIntent::RESOLVE_EFFECT;
                pass.slot_index = static_cast<int>(i);
                actions.push_back(pass);
            }
            else if (eff.type == EffectType::RESOLVE_BATTLE) {
                 Action action;
                 action.type = PlayerIntent::RESOLVE_BATTLE;
                 action.slot_index = static_cast<int>(i);
                 actions.push_back(action);
            }
            else if (eff.type == EffectType::BREAK_SHIELD) {
                 Action action;
                 action.type = PlayerIntent::BREAK_SHIELD;
                 action.slot_index = static_cast<int>(i);
                 actions.push_back(action);
            }
            else if (eff.type == EffectType::INTERNAL_PLAY || eff.type == EffectType::META_COUNTER) {
                Action action;
                action.type = PlayerIntent::PLAY_CARD_INTERNAL;
                action.source_instance_id = eff.source_instance_id;
                action.target_player = eff.controller;
                action.slot_index = static_cast<int>(i);

                const CardInstance* card = game_state.get_card_instance(eff.source_instance_id);
                if (card) {
                    action.card_id = card->card_id;
                }

                if (eff.type == EffectType::META_COUNTER) {
                    action.spawn_source = SpawnSource::HAND_SUMMON;
                } else {
                    action.spawn_source = SpawnSource::EFFECT_SUMMON;
                }

                // Step 3-3: Check for Destination Override
                // If effect says "PLAY_FROM_ZONE" (or derived from it) and "destination_zone" is set...
                // PendingEffect doesn't directly store destination_zone.
                // It stores `EffectDef` in `effect_def`.
                // Or we can check the `EffectPrimitive` of the *ActionDef* if we had it.
                // But `INTERNAL_PLAY` is generated by `GenericCardSystem::resolve_trigger`? No, triggers don't set internal play directly.
                // `INTERNAL_PLAY` is a specific EffectType.
                // Usually comes from "Gatekeeper" actions.

                // If we want to override, we should inspect `eff.effect_def` if present.
                if (eff.effect_def.has_value()) {
                    for (const auto& act : eff.effect_def->actions) {
                        if (act.destination_zone == "DECK_BOTTOM") {
                            action.destination_override = 1; // 1 = Deck Bottom
                            break;
                        }
                    }
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
                         if (ReactionSystem::check_condition(game_state, r, card, eff.controller, card_db, event_type)) {
                             legal = true;
                             break;
                         }
                     }

                     if (legal) {
                         Action act;
                         act.type = PlayerIntent::DECLARE_REACTION;
                         act.source_instance_id = card.instance_id;
                         act.target_player = eff.controller;
                         act.slot_index = static_cast<int>(k);
                         actions.push_back(act);
                     }
                 }

                 Action pass;
                 pass.type = PlayerIntent::RESOLVE_EFFECT;
                 pass.slot_index = static_cast<int>(i);
                 actions.push_back(pass);
             }
             else if (eff.type == EffectType::SELECT_OPTION) {
                 // Generate SELECT_OPTION actions for each option
                 for (size_t opt_idx = 0; opt_idx < eff.options.size(); ++opt_idx) {
                     Action choice;
                     choice.type = PlayerIntent::SELECT_OPTION;
                     choice.slot_index = static_cast<int>(i); // The pending effect index
                     choice.target_slot_index = static_cast<int>(opt_idx); // The chosen option index
                     actions.push_back(choice);
                 }
             }
            else if (eff.type == EffectType::SELECT_NUMBER) {
                // Generate SELECT_NUMBER actions for range [0, num_targets_needed]
                // Note: num_targets_needed is used as the MAX value here.
                // If we need a MIN value, we should store it in PendingEffect (e.g., optional<int> min_val).
                // For now assuming min is 0.

                // If user specifies min count in filter, use it.
                int min_val = 0;
                if (eff.filter.count.has_value()) {
                     // Wait, filter.count is usually fixed count.
                     // If we use filter.min_cost/max_cost for min/max number?
                     // Or just define semantics: num_targets_needed is MAX.
                     // Let's check how SelectNumberHandler sets it up.
                     // It sets num_targets_needed = max.
                }

                for (int val = min_val; val <= eff.num_targets_needed; ++val) {
                    Action select;
                    select.type = PlayerIntent::SELECT_NUMBER;
                    select.slot_index = static_cast<int>(i); // The pending effect index
                    select.target_instance_id = val; // The chosen number (stored in target_instance_id)
                    actions.push_back(select);
                }
            }
            else if (eff.num_targets_needed > (int)eff.target_instance_ids.size()) {
                 if (actions.empty()) {
                     Action resolve;
                     resolve.type = PlayerIntent::RESOLVE_EFFECT;
                     resolve.slot_index = static_cast<int>(i);
                     actions.push_back(resolve);
                 }
            }
            else {
                Action resolve;
                resolve.type = PlayerIntent::RESOLVE_EFFECT;
                resolve.slot_index = static_cast<int>(i);
                actions.push_back(resolve);
            }
        }
        return actions;
    }

}
