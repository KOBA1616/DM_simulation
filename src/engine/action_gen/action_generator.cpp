#include "action_generator.hpp"
#include "engine/card_system/target_utils.hpp"
#include "engine/mana/mana_system.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ActionGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        std::vector<Action> actions;

        // Determine who is acting
        // If phase is BLOCK, it's the NAP. Otherwise AP.
        // But pending effects also have controllers.

        PlayerID decision_maker = game_state.active_player_id;

        // 0. Pending Effects (The Stack)
        if (!game_state.pending_effects.empty()) {
            // Priority Check
            bool ap_has = false;
            for (const auto& eff : game_state.pending_effects) {
                if (eff.controller == game_state.active_player_id) { ap_has = true; break; }
            }
            decision_maker = ap_has ? game_state.active_player_id : (1 - game_state.active_player_id);
            
            // Only generate actions if the viewer/requester is the decision maker?
            // ActionGenerator usually generates ALL legal actions for the current state,
            // but effectively only one player can act.
            // We should filter for decision_maker.
            
            for (size_t i = 0; i < game_state.pending_effects.size(); ++i) {
                const auto& eff = game_state.pending_effects[i];
                if (eff.controller != decision_maker) continue;

                // Handle Generic Target Selection
                if (eff.resolve_type == ResolveType::TARGET_SELECT) {
                    // Check if we have enough targets already
                    if (eff.target_instance_ids.size() >= eff.num_targets_needed) {
                        Action resolve;
                        resolve.type = ActionType::RESOLVE_EFFECT;
                        resolve.slot_index = static_cast<int>(i);
                        actions.push_back(resolve);
                        continue; // Skip generation of SELECT_TARGET
                    }

                    const auto& filter = eff.filter;
                    bool found_target = false;

                    // Iterate Zones specified in filter
                    for (const auto& zone_str : filter.zones) {
                        // Determine target player(s)
                        std::vector<int> players_to_check;
                        if (filter.owner.has_value()) {
                            if (filter.owner.value() == "SELF") players_to_check.push_back(decision_maker);
                            else if (filter.owner.value() == "OPPONENT") players_to_check.push_back(1 - decision_maker);
                            else if (filter.owner.value() == "BOTH") {
                                players_to_check.push_back(decision_maker);
                                players_to_check.push_back(1 - decision_maker);
                            }
                        } else {
                            // Default: usually implies "Any valid target"?
                            // For safety, default to SELF unless zone suggests otherwise?
                            // Actually, if I cast a spell "Destroy creature", target is usually Opponent.
                            // But "Destroy YOUR creature" is Self.
                            // The filter MUST specify owner for safety, or we assume Current Player (Self).
                            // Let's assume SELF if unspecified.
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
                            else if (zone_str == "EFFECT_BUFFER" || zone_str == "BUFFER") {
                                // Effect buffer is global in GameState, but we iterate per player context for logic
                                // Check if this player is the "owner" of the search?
                                // Usually LOOK_TO_BUFFER comes from Deck which is player-specific.
                                // Let's check game_state.effect_buffer.
                                // We iterate players_to_check, but the buffer is global.
                                // We should only check it once per call, or just assign it.
                                // Since logic iterates players, we can just assign the buffer.
                                zone_ptr = &game_state.effect_buffer;
                            }

                            if (zone_ptr) {
                                for (const auto& card : *zone_ptr) {
                                    if (card_db.count(card.card_id) == 0) continue;
                                    const auto& def = card_db.at(card.card_id);

                                    if (TargetUtils::is_valid_target(card, def, filter, decision_maker, (PlayerID)pid)) {
                                        Action select;
                                        select.type = ActionType::SELECT_TARGET;
                                        select.target_instance_id = card.instance_id;
                                        select.slot_index = static_cast<int>(i); // Pending Effect Index
                                        actions.push_back(select);
                                        found_target = true;
                                    }
                                }
                            }
                        }
                    }

                    // Optional Pass or No Targets
                    if (eff.optional || !found_target) {
                        Action pass;
                        pass.type = ActionType::PASS; // Means "Done Selecting" or "Select None"
                        pass.slot_index = static_cast<int>(i);
                        actions.push_back(pass);
                    }
                }
                // Handle Shield Trigger
                else if (eff.type == EffectType::SHIELD_TRIGGER) {
                    Action use;
                    use.type = ActionType::USE_SHIELD_TRIGGER;
                    use.source_instance_id = eff.source_instance_id;
                    use.target_player = eff.controller;
                    use.slot_index = static_cast<int>(i);
                    actions.push_back(use);

                    Action pass;
                    pass.type = ActionType::RESOLVE_EFFECT; // Skip using trigger
                    pass.slot_index = static_cast<int>(i);
                    actions.push_back(pass);
                }
                // Handle Revolution Change (ON_ATTACK_FROM_HAND)
                else if (eff.type == EffectType::ON_ATTACK_FROM_HAND) {
                    const Player& player = game_state.players[eff.controller];

                    // Generate USE_ABILITY actions for all valid Revolution Change cards in Hand
                    // Revolution Change: Requires Civ/Race match with the ATTACKING creature (eff.source_instance_id)
                    // We need to look up the Attacker.
                    // The attacker should still be in Battle Zone.

                    bool can_use = false;
                    auto attacker_it = std::find_if(player.battle_zone.begin(), player.battle_zone.end(),
                        [&](const CardInstance& c) { return c.instance_id == eff.source_instance_id; });

                    if (attacker_it != player.battle_zone.end()) {
                        const CardDefinition& attacker_def = card_db.at(attacker_it->card_id);

                        for (size_t k = 0; k < player.hand.size(); ++k) {
                            const auto& card = player.hand[k];
                            if (card_db.count(card.card_id)) {
                                const auto& def = card_db.at(card.card_id);
                                if (def.keywords.revolution_change) {
                                    // Check Conditions (Simplified: assume valid if keyword set for now,
                                    // or check generic matching logic if we had it exposed easily).
                                    // TODO: Implement strict checks based on Revolution Change requirement (Fire, Dragon, etc.)
                                    // For now, if the card has 'revolution_change' keyword, we allow it.

                                    Action use;
                                    use.type = ActionType::USE_ABILITY;
                                    use.source_instance_id = card.instance_id; // Card in hand to use
                                    use.slot_index = static_cast<int>(k); // Index in hand
                                    // We don't link back to the pending effect index in generic Action,
                                    // but we assume EffectResolver finds it (Step 0 checks active pending effects).
                                    actions.push_back(use);
                                    can_use = true;
                                }
                            }
                        }
                    }

                    // Always allow PASS (Don't use Revolution Change)
                    Action pass;
                    pass.type = ActionType::RESOLVE_EFFECT; // Consumes the pending effect without doing anything
                    pass.slot_index = static_cast<int>(i);
                    actions.push_back(pass);
                }
                // Legacy / Fallback for non-generic selection (if any)
                else if (eff.num_targets_needed > (int)eff.target_instance_ids.size()) {
                     // Fallback to old hardcoded logic if resolve_type wasn't set to TARGET_SELECT
                     // (Though we should ensure it IS set)
                     // ...
                     // If no actions generated yet, add RESOLVE_EFFECT to avoid stuck state
                     if (actions.empty()) {
                         Action resolve;
                         resolve.type = ActionType::RESOLVE_EFFECT;
                         resolve.slot_index = static_cast<int>(i);
                         actions.push_back(resolve);
                     }
                }
                else {
                    // Ready to resolve
                    Action resolve;
                    resolve.type = ActionType::RESOLVE_EFFECT;
                    resolve.slot_index = static_cast<int>(i);
                    actions.push_back(resolve);
                }
            }
            return actions;
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
             // Pass (Don't Block)
             Action pass;
             pass.type = ActionType::PASS;
             actions.push_back(pass);
             return actions;
        }

        // Active Player Actions
        switch (game_state.current_phase) {
            case Phase::MANA:
                // 1. Charge Mana
                for (size_t i = 0; i < active_player.hand.size(); ++i) {
                    const auto& card = active_player.hand[i];
                    Action action;
                    action.type = ActionType::MANA_CHARGE;
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
                // 1. Play Card
                for (size_t i = 0; i < active_player.hand.size(); ++i) {
                    const auto& card = active_player.hand[i];
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);

                        // Normal Play
                        if (ManaSystem::can_pay_cost(game_state, active_player, def, card_db)) {
                            Action action;
                            action.type = ActionType::PLAY_CARD;
                            action.card_id = card.card_id;
                            action.source_instance_id = card.instance_id;
                            action.slot_index = static_cast<int>(i);
                            actions.push_back(action);
                        }

                        // Hyper Energy Play (Tap creatures to reduce cost)
                        if (def.keywords.hyper_energy) {
                            // Find untapped creatures
                            int untapped_creatures = 0;
                            for (const auto& c : active_player.battle_zone) {
                                if (!c.is_tapped && !c.summoning_sickness) untapped_creatures++;
                            }

                            // Generate actions for tapping 1, 2, ... N creatures
                            // Assuming Hyper Energy allows reducing cost by 2 per tap (standard?)
                            // Or just "Hyper Energy" implies a specific mode.
                            // Requirement: "Generate independent actions for 'Tap 1', 'Tap 2'"

                            // Let's assume we can tap up to 'untapped_creatures'.
                            // And for each count, we generate a PLAY_CARD action with 'target_slot_index'
                            // indicating the number of taps required?
                            // Or utilize 'extra_info' (not in struct).
                            // Let's use target_instance_id to store the "Tap Count" for now,
                            // or create a new ActionType if needed. But requirement says "independent actions".
                            // ActionType::PLAY_CARD usually has target_instance_id = -1.

                            // Using 'target_player' or 'target_instance_id' to store tap count mode.
                            // Let's use target_instance_id as the "Tap Count" (Mocking metadata).
                            // We need to verify if this is safe. target_instance_id is usually an ID.
                            // Maybe we should use a dedicated ActionType::PLAY_HYPER_ENERGY?
                            // But keeping ActionType compact is good for AI.
                            // The AI needs to know it is playing with Hyper Energy.

                            // Let's assume a convention: For PLAY_CARD, if target_instance_id > 0, it means Hyper Energy Tap Count.
                            // This is risky if target_instance_id collides with actual IDs.
                            // Better: Use `target_slot_index` (which is usually for battle zone index) for tap count.
                            // Play Card usually doesn't use target_slot_index.

                            for (int taps = 1; taps <= untapped_creatures; ++taps) {
                                // Check if affordable with this many taps?
                                // Assuming each tap reduces cost by 2 (standard assumption for now, need spec).
                                // Or does it reduce cost to X?
                                // "Hyper Energy: You may tap creatures you control... Each creature tapped reduces cost by 2."

                                int reduction = taps * 2;
                                int effective_cost = std::max(0, def.cost - reduction);

                                // Check if we can pay the REST.
                                // We simulate the cost reduction.
                                // Currently ManaSystem::can_pay_cost checks base cost.
                                // We need a way to check modified cost.

                                // Mocking the check:
                                // if (ManaSystem::can_pay_cost(..., effective_cost))
                                // But ManaSystem api takes CardDefinition.
                                // We might need to assume it's legal if we have enough mana for reduced cost.
                                // Simplification:
                                int available_mana = 0;
                                for(const auto& m : active_player.mana_zone) if(!m.is_tapped) available_mana++;

                                if (available_mana >= effective_cost) {
                                    Action action;
                                    action.type = ActionType::PLAY_CARD;
                                    action.card_id = card.card_id;
                                    action.source_instance_id = card.instance_id;
                                    action.slot_index = static_cast<int>(i);
                                    action.target_slot_index = taps; // Encodes "Hyper Mode: Tap N"
                                    // We need to differentiate this from normal play in EffectResolver.
                                    // Maybe add a flag? Action struct is fixed.
                                    // Use target_player = 254 to signify Hyper Energy
                                    action.target_player = 254;
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
                        // Check Summoning Sickness vs Speed Attacker / Evolution
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
                        // Attack Player
                        Action attack_player;
                        attack_player.type = ActionType::ATTACK_PLAYER;
                        attack_player.source_instance_id = card.instance_id;
                        attack_player.slot_index = static_cast<int>(i);
                        attack_player.target_player = opponent.id;
                        actions.push_back(attack_player);

                        // Attack Tapped Creatures
                        for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
                            const auto& opp_card = opponent.battle_zone[j];
                            if (opp_card.is_tapped) {
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
