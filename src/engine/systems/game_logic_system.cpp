#include "game_logic_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/cost_payment_system.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/flow/reaction_system.hpp"

#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    // Helper to extract int from json args with default
    static int get_arg_int(const nlohmann::json& args, const std::string& key, int def = -1) {
        if (args.contains(key) && args[key].is_number()) {
            return args[key];
        }
        return def;
    }

    // Helper to get power (Ported from EffectResolver)
    static int get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;
        power += creature.power_mod;
        power += PassiveEffectSystem::instance().get_power_buff(game_state, creature, card_db);
        return power;
    }

    static int get_breaker_count(const CardInstance& creature, const std::map<CardID, CardDefinition>& card_db) {
         if (!card_db.count(creature.card_id)) return 1;
         const auto& k = card_db.at(creature.card_id).keywords;
         if (k.triple_breaker) return 3;
         if (k.double_breaker) return 2;
         return 1;
    }

    void GameLogicSystem::handle_play_card(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        int source_id = get_arg_int(inst.args, "source_id");
        int target_id = get_arg_int(inst.args, "target_id", -1);
        int target_player = get_arg_int(inst.args, "target_player", state.active_player_id);
        int payment_units = get_arg_int(inst.args, "payment_units", 0);

        // 1. Move to Stack
        auto move_cmd = std::make_shared<TransitionCommand>(
             source_id,
             Zone::HAND,
             Zone::STACK,
             state.active_player_id
        );
        state.execute_command(move_cmd);

        // 2. Hyper Energy / Special Payment
        if (target_player == 254) {
             Player& player = state.players[state.active_player_id];

             // The frontend/AI logic for Hyper Energy passes units (tap count)
             // We need to verify and tap creatures.
             // For robustness, we tap valid untapped creatures starting from index 0 or similar simple heuristic if specific IDs aren't passed.
             // Wait, ActionGenerator usually handles generating specific Tap commands?
             // No, Hyper Energy in this codebase seems to rely on the Engine to do the tapping based on the count.

             int taps_needed = payment_units;
             int taps_done = 0;

             // Find untapped creatures without summoning sickness
             for (auto& c : player.battle_zone) {
                 if (taps_done >= taps_needed) break;
                 if (!c.is_tapped && !c.summoning_sickness) {
                     auto tap_cmd = std::make_shared<MutateCommand>(
                         c.instance_id,
                         MutateCommand::MutationType::TAP
                     );
                     state.execute_command(tap_cmd);
                     taps_done++;
                 }
             }

             // Calculate remaining cost (simplified)
             // In a real scenario, ManaSystem::auto_tap_mana handles the mana part.
             // We assume ManaSystem is invoked by the AI/Player via PAY_COST action usually,
             // but here we are inside PLAY_CARD which orchestrates it.
             // Since we are migrating, we replicate the "Auto Pay" behavior for now.

             // Determine cost
             int card_id = -1;
             if (!state.stack_zone.empty() && state.stack_zone.back().instance_id == source_id) {
                 card_id = state.stack_zone.back().card_id;
             }

             int reduction = taps_done * 2;
             if (card_id != -1 && card_db.count(card_id)) {
                 const auto& def = card_db.at(card_id);
                 int base_adjusted = ManaSystem::get_adjusted_cost(state, player, def);
                 int final_cost = std::max(0, base_adjusted - reduction);
                 ManaSystem::auto_tap_mana(state, player, def, final_cost, card_db);
             }

             // Push RESOLVE_PLAY
             nlohmann::json args;
             args["type"] = "RESOLVE_PLAY";
             args["source_id"] = source_id;
             args["reduction"] = reduction;
             args["spawn_source"] = (int)SpawnSource::HAND_SUMMON;
             pipeline.execute({Instruction(InstructionOp::GAME_ACTION, args)}, state, card_db);
             return;
        }

        // Standard Play Setup
        if (!state.stack_zone.empty() && state.stack_zone.back().instance_id == source_id) {
            auto& stack_card = state.stack_zone.back();
            stack_card.is_tapped = false;
            stack_card.summoning_sickness = true;
            stack_card.power_mod = target_id; // Metadata passing
        }

        // Push RESOLVE_PLAY instruction
        nlohmann::json resolve_args;
        resolve_args["type"] = "RESOLVE_PLAY";
        resolve_args["source_id"] = source_id;
        resolve_args["spawn_source"] = (int)SpawnSource::HAND_SUMMON;
        resolve_args["evo_source_id"] = target_id;

        pipeline.execute({Instruction(InstructionOp::GAME_ACTION, resolve_args)}, state, card_db);
    }

    void GameLogicSystem::handle_resolve_play(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        (void)pipeline;
        int stack_id = get_arg_int(inst.args, "source_id");
        int evo_source_id = get_arg_int(inst.args, "evo_source_id", -1);
        int dest_override = get_arg_int(inst.args, "dest_override", 0);
        SpawnSource spawn_source = (SpawnSource)get_arg_int(inst.args, "spawn_source", (int)SpawnSource::HAND_SUMMON);

        // Locate card in Stack
        auto& stack = state.stack_zone;
        auto it = std::find_if(stack.begin(), stack.end(), [&](const CardInstance& c){ return c.instance_id == stack_id; });

        if (it == stack.end()) return;

        CardInstance card = *it;
        if (evo_source_id == -1 && card.power_mod != 0) {
             evo_source_id = card.power_mod;
        }
        card.power_mod = 0;

        const CardDefinition* def = nullptr;
        if (card_db.count(card.card_id)) def = &card_db.at(card.card_id);

        if (def && def->type == CardType::SPELL) {
            Zone dest = Zone::GRAVEYARD;
            if (dest_override == 1) {
                // Logic to put on deck bottom is complex with current TransitionCommand enums.
                // Assuming Graveyard for robustness if DECK_BOTTOM not implemented in enum.
                // But wait, TransitionCommand handles zones. We can add a specialized command later.
                dest = Zone::GRAVEYARD;
                // If we really needed Deck Bottom, we'd need to manually manipulate or extend TransitionCommand.
                // For this migration, we stick to safe defaults or implement fully.
                // Let's use Graveyard to ensure no crash, consistent with existing stub behavior.
            }

            auto move_cmd = std::make_shared<TransitionCommand>(
                card.instance_id, Zone::STACK, dest, state.active_player_id
            );
            state.execute_command(move_cmd);

            GenericCardSystem::resolve_trigger(state, TriggerType::ON_PLAY, card.instance_id, card_db);
            GenericCardSystem::resolve_trigger(state, TriggerType::ON_CAST_SPELL, card.instance_id, card_db);
            state.turn_stats.spells_cast_this_turn++;

        } else {
            // Move to Battle Zone
            auto move_cmd = std::make_shared<TransitionCommand>(
                card.instance_id, Zone::STACK, Zone::BATTLE, state.active_player_id
            );
            state.execute_command(move_cmd);

            // Post-move setup
            bool speed_attacker = (def && def->keywords.speed_attacker) || (def && def->keywords.evolution);

            // Find in BZ to apply flags
            Player& p = state.players[state.active_player_id];
            for (auto& c : p.battle_zone) {
                if (c.instance_id == card.instance_id) {
                    c.summoning_sickness = !speed_attacker;
                    c.is_tapped = false;
                    c.turn_played = state.turn_number;

                    // Handle Evolution underlying cards
                    if (evo_source_id != -1) {
                        auto s_it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const CardInstance& sc){ return sc.instance_id == evo_source_id; });
                        if (s_it != p.battle_zone.end()) {
                            // Move source underneath evolved creature
                            // We do this by copying and removing source.
                            // Ideally this should be a "Stack" command.
                            c.underlying_cards.push_back(*s_it);
                            p.battle_zone.erase(s_it);
                        }
                    }
                    break;
                }
            }

            GenericCardSystem::resolve_trigger(state, TriggerType::ON_PLAY, card.instance_id, card_db);
            state.turn_stats.creatures_played_this_turn++;
        }

        state.on_card_play(card.card_id, state.turn_number, spawn_source != SpawnSource::HAND_SUMMON, 0, state.active_player_id);
    }

    void GameLogicSystem::handle_attack(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
         (void)pipeline;
         int source_id = get_arg_int(inst.args, "source_id");
         int target_id = get_arg_int(inst.args, "target_id", -1);
         int target_player = get_arg_int(inst.args, "target_player", 1 - state.active_player_id);

         auto tap_cmd = std::make_shared<MutateCommand>(
             source_id, MutateCommand::MutationType::TAP
         );
         state.execute_command(tap_cmd);

         state.current_attack.source_instance_id = source_id;
         state.current_attack.target_instance_id = target_id;
         state.current_attack.target_player = target_player;
         state.current_attack.is_blocked = false;
         state.current_attack.blocker_instance_id = -1;

         GenericCardSystem::resolve_trigger(state, TriggerType::ON_ATTACK, source_id, card_db);
    }

    void GameLogicSystem::handle_block(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        (void)pipeline;
        int blocker_id = get_arg_int(inst.args, "source_id");

        state.current_attack.is_blocked = true;
        state.current_attack.blocker_instance_id = blocker_id;

        auto tap_cmd = std::make_shared<MutateCommand>(
             blocker_id, MutateCommand::MutationType::TAP
        );
        state.execute_command(tap_cmd);

        GenericCardSystem::resolve_trigger(state, TriggerType::ON_BLOCK, blocker_id, card_db);

        state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, blocker_id, state.active_player_id);
    }

    void GameLogicSystem::handle_resolve_battle(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
         (void)pipeline;
         (void)inst;

         int attacker_id = state.current_attack.source_instance_id;
         int defender_id = -1;

         // Determine Defender
         if (state.current_attack.is_blocked) {
             defender_id = state.current_attack.blocker_instance_id;
         } else if (state.current_attack.target_instance_id != -1) {
             defender_id = state.current_attack.target_instance_id;
         } else {
             // Direct Attack on Player
             Player& defender = state.get_non_active_player();
             Player& attacker_player = state.get_active_player();

             int breaker_count = 1;
             auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == attacker_id; });
             if (it != attacker_player.battle_zone.end()) {
                 breaker_count = get_breaker_count(*it, card_db);
             }

             if (defender.shield_zone.empty()) {
                  state.pending_effects.emplace_back(EffectType::BREAK_SHIELD, attacker_id, state.active_player_id);
             } else {
                 int shields_to_break = std::min((int)defender.shield_zone.size(), breaker_count);
                 for (int i=0; i<shields_to_break; ++i) {
                      state.pending_effects.emplace_back(EffectType::BREAK_SHIELD, attacker_id, state.active_player_id);
                 }
             }
             return;
         }

         // Creature Battle Logic
         Player& p1 = state.get_active_player();
         Player& p2 = state.get_non_active_player();

         CardInstance* attacker = nullptr;
         CardInstance* defender = nullptr;

         // Find instances (simplified scan)
         for (auto& c : p1.battle_zone) if (c.instance_id == attacker_id) attacker = &c;
         if (!attacker) return; // Attacker gone?

         // Defender might be in P1 (if P1 attacked own creature?) - Unlikely in standard rules but engine supports targeting.
         // Usually defender is in P2.
         for (auto& c : p2.battle_zone) if (c.instance_id == defender_id) defender = &c;
         if (!defender) {
             // Check P1 too just in case
             for (auto& c : p1.battle_zone) if (c.instance_id == defender_id) defender = &c;
         }

         if (!defender) {
             // Defender gone, battle fizzles
             return;
         }

         int p_att = get_creature_power(*attacker, state, card_db);
         int p_def = get_creature_power(*defender, state, card_db);

         bool att_wins = p_att > p_def;
         bool def_wins = p_def > p_att;
         bool draw = p_att == p_def;

         // Slayer check could be added here (check keywords)
         bool att_slayer = false;
         bool def_slayer = false;
         if (card_db.count(attacker->card_id) && card_db.at(attacker->card_id).keywords.slayer) att_slayer = true;
         if (card_db.count(defender->card_id) && card_db.at(defender->card_id).keywords.slayer) def_slayer = true;

         if (att_slayer) def_wins = true; // Slayer destroys opponent
         if (def_slayer) att_wins = true;

         if (att_wins || draw) {
             // Defender Destroyed
             // We need owner of defender
             PlayerID def_owner = defender->owner;
             if (def_owner > 1) def_owner = state.card_owner_map[defender_id];

             auto destroy_cmd = std::make_shared<TransitionCommand>(
                 defender_id, Zone::BATTLE, Zone::GRAVEYARD, def_owner
             );
             state.execute_command(destroy_cmd);

             GenericCardSystem::resolve_trigger(state, TriggerType::ON_DESTROY, defender_id, card_db);
         }

         if (def_wins || draw) {
             // Attacker Destroyed
             PlayerID att_owner = attacker->owner;
             if (att_owner > 1) att_owner = state.card_owner_map[attacker_id];

             auto destroy_cmd = std::make_shared<TransitionCommand>(
                 attacker_id, Zone::BATTLE, Zone::GRAVEYARD, att_owner
             );
             state.execute_command(destroy_cmd);

             GenericCardSystem::resolve_trigger(state, TriggerType::ON_DESTROY, attacker_id, card_db);
         }
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        (void)pipeline;
        int target_shield_id = get_arg_int(inst.args, "target_id");
        int source_id = get_arg_int(inst.args, "source_id");
        int target_player_id = get_arg_int(inst.args, "target_player", 1 - state.active_player_id);

        Player& defender = state.players[target_player_id];
        if (defender.shield_zone.empty()) {
            // Direct attack success -> Win
            // Verify if source is attacker?
             if (target_player_id != state.active_player_id) {
                 state.winner = (state.active_player_id == 0) ? GameResult::P1_WIN : GameResult::P2_WIN;
             }
             return;
        }

        GenericCardSystem::resolve_trigger(state, TriggerType::AT_BREAK_SHIELD, source_id, card_db);

        // Select Shield
        int shield_index = -1;
        if (target_shield_id != -1) {
            for (size_t i = 0; i < defender.shield_zone.size(); ++i) {
                if (defender.shield_zone[i].instance_id == target_shield_id) {
                    shield_index = i;
                    break;
                }
            }
        }
        if (shield_index == -1) {
            shield_index = defender.shield_zone.size() - 1;
        }

        CardInstance shield = defender.shield_zone[shield_index];

        // Shield Burn Check
        bool shield_burn = false;
        Player& attacker_player = state.get_active_player();
        auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(),
             [&](const CardInstance& c){ return c.instance_id == source_id; });
        if (it != attacker_player.battle_zone.end()) {
             if (card_db.count(it->card_id) && card_db.at(it->card_id).keywords.shield_burn) {
                 shield_burn = true;
             }
        }

        if (shield_burn) {
             auto move_cmd = std::make_shared<TransitionCommand>(
                 shield.instance_id, Zone::SHIELD, Zone::GRAVEYARD, defender.id
             );
             state.execute_command(move_cmd);
             GenericCardSystem::resolve_trigger(state, TriggerType::ON_DESTROY, shield.instance_id, card_db);
        } else {
             auto move_cmd = std::make_shared<TransitionCommand>(
                 shield.instance_id, Zone::SHIELD, Zone::HAND, defender.id
             );
             state.execute_command(move_cmd);

             bool is_trigger = false;
             if (card_db.count(shield.card_id)) {
                 const auto& def = card_db.at(shield.card_id);
                 if (TargetUtils::has_keyword_simple(state, shield, def, "SHIELD_TRIGGER")) {
                     is_trigger = true;
                 }
             }

             if (is_trigger) {
                 state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield.instance_id, defender.id);
             }

             ReactionSystem::check_and_open_window(state, card_db, "ON_SHIELD_ADD", defender.id);
        }
    }
}
