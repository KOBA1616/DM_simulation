#include "play_system.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include "engine/systems/rules/restriction_system.hpp"
#include "engine/systems/effects/trigger_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/action_primitive_utils.hpp"
#include <iostream>
#include <fstream>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void PlaySystem::handle_play_card(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                      const std::map<CardID, CardDefinition>& card_db) {
        int card_id = exec.resolve_int(inst.args.value("card", 0));
        int instance_id = card_id;

        CardInstance* card = state.get_card_instance(instance_id);
        if (!card) return;

        // Gatekeeper
        int origin_int = exec.resolve_int(inst.args.value("origin_zone", -1));
        std::string origin_str = "";
        if (origin_int != -1) {
            Zone origin = static_cast<Zone>(origin_int);
            if (origin == Zone::DECK) origin_str = "DECK";
            else if (origin == Zone::HAND) origin_str = "HAND";
            else if (origin == Zone::MANA) origin_str = "MANA_ZONE";
            else if (origin == Zone::GRAVEYARD) origin_str = "GRAVEYARD";
            else if (origin == Zone::SHIELD) origin_str = "SHIELD_ZONE";
        }

        const auto& def = card_db.at(card->card_id);
        if (RestrictionSystem::instance().is_play_forbidden(state, *card, def, origin_str, card_db)) return;

        std::vector<Instruction> generated;
        Zone dest = Zone::BATTLE;
        bool to_bottom = false;

        bool play_for_free = false;
        bool put_in_play = false;
        if (inst.args.contains("play_flags")) {
            const auto& pf = inst.args["play_flags"];
            if (pf.is_string()) {
                std::string s = pf.get<std::string>();
                if (s == "PLAY_FOR_FREE") play_for_free = true;
                if (s == "PUT_IN_PLAY") put_in_play = true;
            } else if (pf.is_array()) {
                for (const auto& x : pf) {
                    if (!x.is_string()) continue;
                    std::string s = x.get<std::string>();
                    if (s == "PLAY_FOR_FREE") play_for_free = true;
                    if (s == "PUT_IN_PLAY") put_in_play = true;
                }
            }
        }

        if (play_for_free) {
            auto flow_cmd = std::make_unique<FlowCommand>(FlowCommand::FlowType::SET_PLAYED_WITHOUT_MANA, 1);
            state.execute_command(std::move(flow_cmd));
        }

        if (put_in_play) {
             nlohmann::json args = inst.args;
             Instruction resolve_inst(InstructionOp::GAME_ACTION, args);
             resolve_inst.args["type"] = "RESOLVE_PLAY";
             {
                 auto block = std::make_shared<std::vector<Instruction>>();
                 block->push_back(resolve_inst);
                 exec.call_stack.push_back({block, 0, LoopContext{}});
             }
             return;
        }

        if (inst.args.contains("dest_override")) {
             ZoneDestination override_val = static_cast<ZoneDestination>(exec.resolve_int(inst.args["dest_override"]));
             if (override_val == ZoneDestination::DECK_BOTTOM) {
                 dest = Zone::DECK;
                 to_bottom = true;
             } else if (override_val == ZoneDestination::DECK_TOP) {
                 dest = Zone::DECK;
             } else if (override_val == ZoneDestination::HAND) {
                 dest = Zone::HAND;
             } else if (override_val == ZoneDestination::MANA_ZONE) {
                 dest = Zone::MANA;
             } else if (override_val == ZoneDestination::SHIELD_ZONE) {
                 dest = Zone::SHIELD;
             }
        } else {
             dest = Zone::STACK;
        }

        Instruction move(InstructionOp::MOVE);
        move.args["target"] = instance_id;

        if (dest == Zone::STACK) move.args["to"] = "STACK";
        else if (dest == Zone::DECK) move.args["to"] = "DECK";
        else if (dest == Zone::HAND) move.args["to"] = "HAND";
        else if (dest == Zone::MANA) move.args["to"] = "MANA";
        else if (dest == Zone::SHIELD) move.args["to"] = "SHIELD";
        else move.args["to"] = "BATTLE";

        if (to_bottom) move.args["to_bottom"] = true;

        generated.push_back(move);

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    void PlaySystem::handle_mana_charge(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         int card_id = exec.resolve_int(inst.args.value("card", 0));
         if (card_id < 0) return;

         const CardInstance* card_ptr = state.get_card_instance(card_id);
         if (!card_ptr) return;

         PlayerID owner = card_ptr->owner;
         if (state.turn_stats.mana_charged_by_player[owner]) {
             return;
         }

         Instruction move = utils::ActionPrimitiveUtils::create_mana_charge_instruction(card_id);

         auto block = std::make_shared<std::vector<Instruction>>();
         block->push_back(move);
         exec.call_stack.push_back({block, 0, LoopContext{}});

         auto flow_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_MANA_CHARGED, 1);
         state.execute_command(std::move(flow_cmd));
    }

    void PlaySystem::handle_resolve_play(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                         const std::map<CardID, CardDefinition>& card_db) {
        int instance_id = exec.resolve_int(inst.args.value("card", 0));
        const CardInstance* card = state.get_card_instance(instance_id);
        if (!card) return;

        // Gatekeeper
        int origin_int = exec.resolve_int(inst.args.value("origin_zone", -1));
        std::string origin_str = "";
        if (origin_int != -1) {
             Zone origin = static_cast<Zone>(origin_int);
             if (origin == Zone::DECK) origin_str = "DECK";
             else if (origin == Zone::HAND) origin_str = "HAND";
             else if (origin == Zone::MANA) origin_str = "MANA_ZONE";
             else if (origin == Zone::GRAVEYARD) origin_str = "GRAVEYARD";
             else if (origin == Zone::SHIELD) origin_str = "SHIELD_ZONE";
        }

        if (!card_db.count(card->card_id)) return;

        bool is_spell_side = exec.resolve_int(inst.args.value("is_spell_side", 0)) != 0;
        const auto& base_def = card_db.at(card->card_id);
        const auto& def = (is_spell_side && base_def.spell_side) ? *base_def.spell_side : base_def;

        if (RestrictionSystem::instance().is_play_forbidden(state, *card, def, origin_str, card_db)) return;

        state.on_card_play(card->card_id, state.turn_number, false, 0, card->owner);

        std::vector<Instruction> compiled_effects;

        if (def.type == CardType::SPELL) {
            nlohmann::json trig_args;
            trig_args["type"] = "CHECK_SPELL_CAST_TRIGGERS";
            trig_args["card"] = instance_id;
            compiled_effects.emplace_back(InstructionOp::GAME_ACTION, trig_args);

            nlohmann::json move_args;
            move_args["target"] = instance_id;
            move_args["to"] = "GRAVEYARD";
            compiled_effects.emplace_back(InstructionOp::MOVE, move_args);
        } else if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
            bool is_evolution = def.keywords.evolution;

            if (is_evolution) {
                FilterDef evo_filter;
                evo_filter.zones = {"BATTLE_ZONE"};
                evo_filter.races = def.races;
                evo_filter.owner = "SELF";

                Instruction select(InstructionOp::SELECT);
                select.args["filter"] = evo_filter;
                select.args["count"] = 1;
                select.args["out"] = "$evo_target";
                compiled_effects.push_back(select);

                Instruction attach(InstructionOp::MOVE);
                attach.args["target"] = instance_id;
                attach.args["attach_to"] = "$evo_target";
                compiled_effects.push_back(attach);
            } else {
                nlohmann::json move_args;
                move_args["target"] = instance_id;
                move_args["to"] = "BATTLE";
                compiled_effects.emplace_back(InstructionOp::MOVE, move_args);

                nlohmann::json untap_args;
                untap_args["type"] = "UNTAP";
                untap_args["target"] = instance_id;
                compiled_effects.emplace_back(InstructionOp::MODIFY, untap_args);
            }

            nlohmann::json trig_args;
            trig_args["type"] = "CHECK_CREATURE_ENTER_TRIGGERS";
            trig_args["card"] = instance_id;
            compiled_effects.emplace_back(InstructionOp::GAME_ACTION, trig_args);
        }

        if (!compiled_effects.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(compiled_effects);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    void PlaySystem::handle_use_ability(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                        const std::map<CardID, CardDefinition>& card_db) {
        int source_id = exec.resolve_int(inst.args.value("source", -1));
        int target_id = exec.resolve_int(inst.args.value("target", -1));

        if (source_id == -1 || target_id == -1) return;

        auto return_cmd = std::make_unique<TransitionCommand>(target_id, Zone::BATTLE, Zone::HAND, state.active_player_id);
        state.execute_command(std::move(return_cmd));

        auto play_cmd = std::make_unique<TransitionCommand>(source_id, Zone::HAND, Zone::BATTLE, state.active_player_id);
        state.execute_command(std::move(play_cmd));

        CardInstance* new_creature = state.get_card_instance(source_id);
        if (new_creature) {
            new_creature->turn_played = state.turn_number;
            auto tap_cmd = std::make_shared<MutateCommand>(source_id, MutateCommand::MutationType::TAP);
            state.execute_command(std::move(tap_cmd));
        }

        auto flow_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_SOURCE, source_id);
        state.execute_command(std::move(flow_cmd));

        (void)card_db;
    }

}
