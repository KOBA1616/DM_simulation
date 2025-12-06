#include "action_generator.hpp"
#include "engine/card_system/target_utils.hpp"
#include "engine/mana/mana_system.hpp"
#include "engine/flow/reaction_system.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ActionGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        std::vector<Action> actions;

        // Determine who is acting
        PlayerID decision_maker = game_state.active_player_id;

        // 0. Pending Effects (The Stack / Triggers) - Highest Priority
        if (!game_state.pending_effects.empty()) {
            bool ap_has = false;
            for (const auto& eff : game_state.pending_effects) {
                if (eff.controller == game_state.active_player_id) { ap_has = true; break; }
            }
            decision_maker = ap_has ? game_state.active_player_id : (1 - game_state.active_player_id);
            
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
                                        // is_valid_target allows general filtering (e.g. counting)
                                        // But here we are generating SELECT_TARGET actions for manual choice.
                                        // Just Diver prevents being CHOSEN by OPPONENT.
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
                     // Allow targeting specific shields?
                     // For now, we generate one action. If we want manual selection,
                     // we would loop through enemy shields and generate actions for each.
                     // But EffectResolver logic currently pops back.
                     // Let's implement fully later. For now, automatic single action.
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
                     // Generate DECLARE_REACTION actions for matching cards in hand.
                     // The pending effect's controller is the player who can react.

                     // We need to re-verify conditions or just trust they were checked?
                     // check_and_open_window already verified potential existence.
                     // Now we list legal cards.

                     const auto& player = game_state.players[eff.controller];
                     std::string event_type = eff.reaction_context.has_value() ? eff.reaction_context->trigger_event : "";

                     // Iterate hand
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
                             act.slot_index = static_cast<int>(k); // Use slot in hand? Or slot in pending?
                             // We need to know which pending effect triggered this window?
                             // Actually, since pending effects are processed top-down, we know it's index 'i'.
                             // We don't store 'i' in action usually for this purpose, but we can.
                             // But DECLARE_REACTION uses the card.
                             // Resolving this action will NOT remove the pending effect (window),
                             // unless we implement logic to do so.
                             // Wait, user can use multiple.

                             actions.push_back(act);
                         }
                     }

                     // Always offer PASS (close window)
                     Action pass;
                     pass.type = ActionType::RESOLVE_EFFECT; // Use RESOLVE_EFFECT to signal "Done with this pending effect"
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

        // 0.5. Check for Cards on Stack (Atomic Action Flow)
        // If there are cards on the stack (and no pending effects which were handled above),
        // we must process the stack card.
        // Currently we assume only 1 card on stack for PLAY sequence.
        // TODO: Handle multiple cards if we support stack-based chains later.
        if (!game_state.stack_zone.empty()) {
            const auto& stack_card = game_state.stack_zone.back();
            // Check state of the card?
            // We need to know if cost is paid.
            // But we don't track "is_cost_paid" on CardInstance easily without adding a field.
            // OR we rely on `game_state` having a mode?
            // Requirement says: DECLARE -> PAY -> RESOLVE.
            // If card is on stack, it means it was DECLARED.
            // Now we must PAY or RESOLVE.

            // For now, let's assume if it's on stack, it's waiting for payment or resolution.
            // If G-Zero was used or cost is 0, we can skip payment?
            // The Refactor plan says:
            // "PAY_COST: Calculate cost... if fail, return failure."
            // So we should generate PAY_COST action.

            // Is it possible we ALREADY paid?
            // If we use `PAY_COST` action, it should transition the state or mark the card as paid.
            // Since we don't have a "Paid" flag on CardInstance, we might need a temporary way to track this.
            // OR we generate PAY_COST, and if successful, EffectResolver IMMEDIATELY resolves it or moves it to a "Paid" state?
            // BUT ActionGenerator is called every step.

            // Alternative: `stack_zone` holds cards that are "Declared".
            // `PAY_COST` action takes card from Stack, taps mana.
            // IF successful, it triggers `RESOLVE_PLAY`.

            // Problem: If `PAY_COST` is an action, the agent must choose it.
            // Is there a choice in paying cost?
            // Yes, which mana to tap?
            // But currently `ManaSystem` auto-taps.
            // If auto-tap, then `PAY_COST` is deterministic.

            // To support the decomposed flow:
            // 1. `DECLARE_PLAY` moves to stack.
            // 2. Next call to `generate_legal_actions` sees card on stack.
            //    Generates `PAY_COST`.
            // 3. Agent chooses `PAY_COST`.
            //    `EffectResolver::resolve_pay_cost`:
            //       Taps mana.
            //       TRANSITIONS internal state?
            //       Wait, if we just tap mana, the card is still on stack.
            //       How do we know it's paid?
            //       Maybe we add `is_paid` flag to CardInstance? Or `flags`.

            // Let's use `CardInstance::summoning_sickness` or `is_tapped` on stack card to store state? Hacky.
            // `is_tapped` on stack is meaningless. Let's use `is_tapped = true` to mean "Cost Paid".

            if (stack_card.is_tapped) {
                 // Cost is paid. Generate RESOLVE_PLAY.
                 Action resolve;
                 resolve.type = ActionType::RESOLVE_PLAY;
                 resolve.card_id = stack_card.card_id;
                 resolve.source_instance_id = stack_card.instance_id;
                 actions.push_back(resolve);
            } else {
                 // Cost not paid. Generate PAY_COST.
                 Action pay;
                 pay.type = ActionType::PAY_COST;
                 pay.card_id = stack_card.card_id;
                 pay.source_instance_id = stack_card.instance_id;
                 actions.push_back(pay);

                 // Also allow Cancel? (Return to hand)
                 // Requirement doesn't explicitly say, but usually you can't cancel after declaring unless you can't pay.
                 // But if the agent realizes it can't pay, it might get stuck if we don't allow cancel or if PAY_COST doesn't handle failure gracefully.
                 // Let's assume PAY_COST handles failure by bouncing back to hand (as per plan).
            }

            return actions; // Stack blocks other main phase actions?
            // Yes, "Stack" implies priority.
        }

        const Player& active_player = game_state.players[game_state.active_player_id];
        const Player& opponent = game_state.players[1 - game_state.active_player_id];

        // If Block Phase, NAP acts
        if (game_state.current_phase == Phase::BLOCK) {
             const Player& defender = opponent; // NAP
             for (size_t i = 0; i < defender.battle_zone.size(); ++i) {
                 const auto& card = defender.battle_zone[i];
                 if (!card.is_tapped) {
                     if (card_db.count(card.card_id)) {
                         const auto& def = card_db.at(card.card_id);
                         if (def.keywords.blocker) {
                             Action block;
                             block.type = ActionType::BLOCK;
                             block.source_instance_id = card.instance_id;
                             block.slot_index = static_cast<int>(i);
                             actions.push_back(block);
                         }
                     }
                 }
             }
             Action pass;
             pass.type = ActionType::PASS;
             actions.push_back(pass);
             return actions;
        }

        // Active Player Actions
        switch (game_state.current_phase) {
            case Phase::START_OF_TURN:
            case Phase::DRAW:
                // No actions, auto-advance via next_phase usually, or we can emit PASS
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;
            case Phase::MANA:
                for (size_t i = 0; i < active_player.hand.size(); ++i) {
                    const auto& card = active_player.hand[i];
                    Action action;
                    // Use MOVE_CARD. In MANA Phase this implies destination=MANA_ZONE.
                    action.type = ActionType::MOVE_CARD;
                    action.card_id = card.card_id;
                    action.source_instance_id = card.instance_id;
                    action.slot_index = static_cast<int>(i);
                    actions.push_back(action);
                }
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::MAIN:
                for (size_t i = 0; i < active_player.hand.size(); ++i) {
                    const auto& card = active_player.hand[i];
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);

                        // Use DECLARE_PLAY instead of PLAY_CARD
                        // We check legality roughly here, but strict check happens at PAY_COST
                        // However, to avoid spamming invalid actions, we can do a preliminary check.

                        if (ManaSystem::can_pay_cost(game_state, active_player, def, card_db)) {
                            Action action;
                            action.type = ActionType::DECLARE_PLAY; // Changed from PLAY_CARD
                            action.card_id = card.card_id;
                            action.source_instance_id = card.instance_id;
                            action.slot_index = static_cast<int>(i);
                            actions.push_back(action);
                        }

                        if (def.keywords.hyper_energy) {
                            int untapped_creatures = 0;
                            for (const auto& c : active_player.battle_zone) {
                                if (!c.is_tapped && !c.summoning_sickness) untapped_creatures++;
                            }

                            for (int taps = 1; taps <= untapped_creatures; ++taps) {
                                int reduction = taps * 2;
                                int effective_cost = std::max(0, def.cost - reduction);

                                int available_mana = 0;
                                for(const auto& m : active_player.mana_zone) if(!m.is_tapped) available_mana++;

                                if (available_mana >= effective_cost) {
                                    Action action;
                                    action.type = ActionType::DECLARE_PLAY; // Changed
                                    action.card_id = card.card_id;
                                    action.source_instance_id = card.instance_id;
                                    action.slot_index = static_cast<int>(i);
                                    action.target_slot_index = taps;
                                    action.target_player = 254; // Hyper Energy Indicator
                                    actions.push_back(action);
                                }
                            }
                        }
                    }
                }
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::ATTACK:
                for (size_t i = 0; i < active_player.battle_zone.size(); ++i) {
                    const auto& card = active_player.battle_zone[i];

                    bool can_attack = !card.is_tapped;
                    if (can_attack) {
                        if (card.summoning_sickness) {
                            if (card_db.count(card.card_id)) {
                                const auto& def = card_db.at(card.card_id);
                                if (!def.keywords.speed_attacker && !def.keywords.evolution) {
                                    can_attack = false;
                                }
                            } else {
                                can_attack = false;
                            }
                        }
                    }

                    if (can_attack) {
                        Action attack_player;
                        attack_player.type = ActionType::ATTACK_PLAYER;
                        attack_player.source_instance_id = card.instance_id;
                        attack_player.slot_index = static_cast<int>(i);
                        attack_player.target_player = opponent.id;
                        actions.push_back(attack_player);

                        for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
                            const auto& opp_card = opponent.battle_zone[j];
                            if (opp_card.is_tapped) {

                                // Check Just Diver (Cannot be Attacked)
                                if (card_db.count(opp_card.card_id)) {
                                    const auto& opp_def = card_db.at(opp_card.card_id);
                                    if (TargetUtils::is_protected_by_just_diver(opp_card, opp_def, game_state, active_player.id)) {
                                        continue; // Cannot attack this creature
                                    }
                                }

                                Action attack_creature;
                                attack_creature.type = ActionType::ATTACK_CREATURE;
                                attack_creature.source_instance_id = card.instance_id;
                                attack_creature.slot_index = static_cast<int>(i);
                                attack_creature.target_instance_id = opp_card.instance_id;
                                attack_creature.target_slot_index = static_cast<int>(j);
                                actions.push_back(attack_creature);
                            }
                        }
                    }
                }
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            default:
                break;
        }

        return actions;
    }

}
