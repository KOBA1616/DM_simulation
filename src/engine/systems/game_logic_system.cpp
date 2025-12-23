#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/card/effect_system.hpp" // Added include for EffectSystem
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/command_system.hpp" // Added for CommandSystem
#include "engine/systems/flow/phase_manager.hpp" // Added for PhaseManager
#include "engine/systems/mana/mana_system.hpp" // Added for ManaSystem
#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    void GameLogicSystem::dispatch_action(PipelineExecutor& pipeline, core::GameState& state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db) {
        // Map ActionType to handler
        // Simplified mapping for now

        switch (action.type) {
            case ActionType::PLAY_CARD:
            {
                // Convert Action to Instruction
                nlohmann::json args;
                args["card"] = action.source_instance_id;
                Instruction inst(InstructionOp::PLAY, args);
                handle_play_card(pipeline, state, inst, card_db);
                break;
            }
            case ActionType::RESOLVE_PLAY:
            {
                nlohmann::json args;
                args["card"] = action.source_instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_PLAY";
                handle_resolve_play(pipeline, state, inst, card_db);
                break;
            }
            case ActionType::DECLARE_PLAY:
            {
                int iid = action.source_instance_id;
                int pid = state.active_player_id;
                // Move to Stack
                auto cmd = std::make_unique<TransitionCommand>(iid, Zone::HAND, Zone::STACK, pid);
                state.execute_command(std::move(cmd));

                // Set Neo flag on the stack instance
                if (auto* c = state.get_card_instance(iid)) {
                    c->is_neo_evolution_play = action.is_evolution_play;
                }
                break;
            }
            case ActionType::PAY_COST:
            {
                int iid = action.source_instance_id;
                // Auto tap mana
                if (auto* c = state.get_card_instance(iid)) {
                    if (card_db.count(c->card_id)) {
                        const auto& def = card_db.at(c->card_id);
                        ManaSystem::auto_tap_mana(state, state.players[state.active_player_id], def, card_db);
                    }
                    // Mark as paid (using is_tapped flag on stack card)
                    c->is_tapped = true;
                }
                break;
            }
            case ActionType::ATTACK_CREATURE:
            case ActionType::ATTACK_PLAYER:
            {
                nlohmann::json args;
                args["source"] = action.source_instance_id;
                args["target"] = action.target_instance_id;
                Instruction inst(InstructionOp::ATTACK, args);
                handle_attack(pipeline, state, inst, card_db);
                break;
            }
            case ActionType::BLOCK:
            {
                nlohmann::json args;
                args["blocker"] = action.source_instance_id;
                Instruction inst(InstructionOp::BLOCK, args);
                handle_block(pipeline, state, inst, card_db);
                break;
            }
            case ActionType::RESOLVE_BATTLE:
            {
                nlohmann::json args;
                args["attacker"] = action.source_instance_id;
                args["defender"] = action.target_instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_BATTLE";
                handle_resolve_battle(pipeline, state, inst, card_db);
                break;
            }
            case ActionType::PASS:
            {
                PhaseManager::next_phase(state, card_db);
                break;
            }
            case ActionType::MANA_CHARGE:
            {
                int iid = action.source_instance_id;
                int pid = state.active_player_id;
                if (const auto* c = state.get_card_instance(iid)) {
                    pid = c->owner;
                }

                if (iid >= 0) {
                     auto cmd = std::make_unique<TransitionCommand>(iid, Zone::HAND, Zone::MANA, pid);
                     state.execute_command(std::move(cmd));
                }
                break;
            }
            case ActionType::MOVE_CARD:
            {
                int iid = action.source_instance_id;
                int pid = state.active_player_id;
                if (const auto* c = state.get_card_instance(iid)) {
                    pid = c->owner;
                }
                if (iid >= 0) {
                     Zone dest = Zone::GRAVEYARD; // Default
                     if (action.destination_override == 3) dest = Zone::HAND;
                     else if (action.destination_override == 4) dest = Zone::MANA;
                     else if (action.destination_override == 5) dest = Zone::SHIELD;
                     else if (action.destination_override == 1) dest = Zone::DECK;
                     else if (state.current_phase == Phase::MANA) dest = Zone::MANA; // Fallback for Mana Phase
                     // TransitionCommand doesn't support deck bottom explicitly via Zone::DECK.
                     // But for Mana Charge (4), it works.

                     // Helper: deck bottom usually requires specific handling or TransitionCommand index.
                     // For now, support MANA/HAND/GRAVE/SHIELD.

                     auto cmd = std::make_unique<TransitionCommand>(iid, Zone::HAND, dest, pid);
                     // Note: Assuming FROM HAND. ActionGenerator for Mana sets it from Hand.
                     // If generic MOVE_CARD, we might need source zone?
                     // Action struct doesn't have source zone.
                     // We assume Hand for Mana Charge context.

                     state.execute_command(std::move(cmd));
                }
                break;
            }
            case ActionType::USE_ABILITY:
            {
                nlohmann::json args;
                args["source"] = action.source_instance_id;
                args["target"] = action.target_instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "USE_ABILITY";
                handle_use_ability(pipeline, state, inst, card_db);
                break;
            }
            // ...
            default: break;
        }
    }

    void GameLogicSystem::resolve_action_oneshot(core::GameState& state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db) {
        PipelineExecutor pipeline;
        dispatch_action(pipeline, state, action, card_db);
        pipeline.execute(nullptr, state, card_db); // Run the pipeline
    }

    void GameLogicSystem::resolve_play_from_stack(core::GameState& game_state, int stack_instance_id, int cost_reduction, core::SpawnSource spawn_source, core::PlayerID controller, const std::map<core::CardID, core::CardDefinition>& card_db, int evo_source_id, int dest_override) {
        // Implementation stub for linker
        (void)game_state; (void)stack_instance_id; (void)cost_reduction; (void)spawn_source; (void)controller; (void)card_db; (void)evo_source_id; (void)dest_override;
    }

    void GameLogicSystem::handle_play_card(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                           const std::map<core::CardID, core::CardDefinition>& card_db) {
        int card_id = exec.resolve_int(inst.args.value("card", 0));
        int instance_id = card_id; // "card" arg is assumed to be instance_id

        CardInstance* card = state.get_card_instance(instance_id);
        if (!card) return;

        bool is_evolution = false;
        FilterDef evo_filter;
        if (card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);

            // Auto-set G-Neo flag
            if (def.type == CardType::G_NEO_CREATURE) {
                card->is_g_neo = true;
            }

            if (def.keywords.evolution || card->is_neo_evolution_play) {
                is_evolution = true;
                // Task B: Refined Evolution Filters
                evo_filter.zones = {"BATTLE_ZONE"};
                evo_filter.races = def.races; // Evolution matches race
                evo_filter.owner = "SELF";
            }
        }

        std::vector<Instruction> generated;

        if (is_evolution) {
            // 1. Select Base
            Instruction select(InstructionOp::SELECT);
            select.args["filter"] = evo_filter;
            select.args["count"] = 1;
            select.args["out"] = "$evo_target";
            generated.push_back(select);

            // 2. Attach
            Instruction attach(InstructionOp::MOVE);
            attach.args["target"] = instance_id;
            attach.args["attach_to"] = "$evo_target";
            generated.push_back(attach);

        } else {
             const auto& def = card_db.at(card->card_id);
             Zone dest = Zone::BATTLE;
             if (def.type == CardType::SPELL) {
                 dest = Zone::STACK;
             }

             Instruction move(InstructionOp::MOVE);
             move.args["target"] = instance_id;
             if (dest == Zone::STACK) move.args["to"] = "STACK";
             else move.args["to"] = "BATTLE";
             generated.push_back(move);
        }

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    void GameLogicSystem::handle_resolve_play(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        int instance_id = exec.resolve_int(inst.args.value("card", 0));
        const CardInstance* card = state.get_card_instance(instance_id);
        if (!card) return;

        if (!card_db.count(card->card_id)) return;
        const auto& def = card_db.at(card->card_id);

        // 1. Compile Effects
        std::vector<Instruction> compiled_effects;
        std::map<std::string, int> ctx;

        if (def.type == CardType::SPELL) {
            for (const auto& eff : def.effects) {
                 EffectSystem::instance().compile_effect(state, eff, instance_id, ctx, card_db, compiled_effects);
            }

            // 2. Move to Graveyard (after effects)
            nlohmann::json move_args;
            move_args["target"] = instance_id;
            move_args["to"] = "GRAVEYARD";
            compiled_effects.emplace_back(InstructionOp::MOVE, move_args);
        }

        if (!compiled_effects.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(compiled_effects);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    void GameLogicSystem::handle_attack(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                        const std::map<core::CardID, core::CardDefinition>& card_db) {
         int instance_id = exec.resolve_int(inst.args.value("source", 0));
         auto cmd = std::make_unique<MutateCommand>(instance_id, MutateCommand::MutationType::TAP);
         state.execute_command(std::move(cmd));

         auto phase_cmd = std::make_unique<FlowCommand>(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(Phase::BLOCK));
         state.execute_command(std::move(phase_cmd));

         (void)card_db;
    }

    void GameLogicSystem::handle_block(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        // ...
        (void)exec; (void)state; (void)inst; (void)card_db;
    }

    void GameLogicSystem::handle_resolve_battle(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                const std::map<core::CardID, core::CardDefinition>& card_db) {
        int attacker_id = exec.resolve_int(inst.args.value("attacker", -1));
        int defender_id = exec.resolve_int(inst.args.value("defender", -1));

        const CardInstance* attacker = state.get_card_instance(attacker_id);
        const CardInstance* defender = state.get_card_instance(defender_id);

        if (!attacker || !defender) return;

        int power_attacker = get_creature_power(*attacker, state, card_db);
        int power_defender = get_creature_power(*defender, state, card_db);

        bool attacker_dies = false;
        bool defender_dies = false;

        bool attacker_slayer = false;
        if (card_db.count(attacker->card_id)) attacker_slayer = card_db.at(attacker->card_id).keywords.slayer;

        bool defender_slayer = false;
        if (card_db.count(defender->card_id)) defender_slayer = card_db.at(defender->card_id).keywords.slayer;

        if (power_attacker > power_defender) {
            defender_dies = true;
            if (defender_slayer) attacker_dies = true;
        } else if (power_attacker < power_defender) {
            attacker_dies = true;
            if (attacker_slayer) defender_dies = true;
        } else {
            attacker_dies = true;
            defender_dies = true;
        }

        if (attacker_dies) {
            auto cmd = std::make_unique<TransitionCommand>(attacker_id, Zone::BATTLE, Zone::GRAVEYARD, attacker->owner);
            state.execute_command(std::move(cmd));
        }

        if (defender_dies) {
            auto cmd = std::make_unique<TransitionCommand>(defender_id, Zone::BATTLE, Zone::GRAVEYARD, defender->owner);
            state.execute_command(std::move(cmd));
        }
    }

    int GameLogicSystem::get_creature_power(const core::CardInstance& creature, const core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;
        for (const auto& mod : game_state.active_modifiers) {
             // ...
        }
        return power;
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        int shield_id = exec.resolve_int(inst.args.value("shield", -1));
        if (shield_id == -1) return;

        std::vector<Instruction> generated;

        // 1. Move Shield to Hand
        Instruction move(InstructionOp::MOVE);
        move.args["to"] = "HAND";
        move.args["target"] = shield_id;
        generated.push_back(move);

        // 2. Check S-Trigger
        const auto* card = state.get_card_instance(shield_id);
        if (card && card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);
            if (def.keywords.shield_trigger) {
                Instruction check(InstructionOp::GAME_ACTION);
                check.args["type"] = "CHECK_S_TRIGGER";
                check.args["card"] = shield_id;
                generated.push_back(check);
            }
        }

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    // Helper for CHECK_S_TRIGGER
    void handle_check_s_trigger(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                const std::map<core::CardID, core::CardDefinition>& card_db) {
        int card_id = exec.resolve_int(inst.args.value("card", -1));

        std::string decision_key = "$strigger_" + std::to_string(card_id);
        ContextValue val = exec.get_context_var(decision_key);

        bool use = false;
        bool decided = false;

        if (std::holds_alternative<int>(val)) {
             use = std::get<int>(val) == 1;
             decided = true;
        }

        if (!decided) {
             exec.execution_paused = true;
             exec.waiting_for_key = decision_key;
             state.waiting_for_user_input = true;
             state.pending_query = GameState::QueryContext{
                 0, "SELECT_OPTION", {}, {}, {"No", "Yes"}
             };
             return;
        }

        if (use) {
             const auto& def = card_db.at(state.get_card_instance(card_id)->card_id);

             Instruction play_inst(InstructionOp::PLAY);
             play_inst.args["card"] = card_id;
             // We rely on standard play.
             // Note: S-Trigger Play is free.
             // handle_play_card does NOT check cost. It just moves card.
             // Payment (PAY_COST) is a separate action in the loop.
             // S-Trigger execution does not generate PAY_COST action.
             // So simply calling PLAY instruction works!

             auto block = std::make_shared<std::vector<Instruction>>();
             block->push_back(play_inst);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    void GameLogicSystem::handle_mana_charge(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         // ...
         (void)exec; (void)state; (void)inst;
    }

    void GameLogicSystem::handle_resolve_reaction(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                  const std::map<core::CardID, core::CardDefinition>& card_db) {
         // ...
         (void)exec; (void)state; (void)inst; (void)card_db;
    }

    void GameLogicSystem::handle_use_ability(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                             const std::map<core::CardID, core::CardDefinition>& card_db) {
        int source_id = exec.resolve_int(inst.args.value("source", -1)); // Card in Hand
        int target_id = exec.resolve_int(inst.args.value("target", -1)); // Attacker in Battle Zone

        if (source_id == -1 || target_id == -1) return;

        // 1. Return Attacker to Hand
        auto return_cmd = std::make_unique<TransitionCommand>(target_id, Zone::BATTLE, Zone::HAND, state.active_player_id);
        state.execute_command(std::move(return_cmd));

        // 2. Put Revolution Change Creature into Battle Zone
        auto play_cmd = std::make_unique<TransitionCommand>(source_id, Zone::HAND, Zone::BATTLE, state.active_player_id);
        state.execute_command(std::move(play_cmd));

        // 3. Update State
        CardInstance* new_creature = state.get_card_instance(source_id);
        if (new_creature) {
            new_creature->turn_played = state.turn_number;
            auto tap_cmd = std::make_unique<MutateCommand>(source_id, MutateCommand::MutationType::TAP);
            state.execute_command(std::move(tap_cmd));
        }

        auto flow_cmd = std::make_unique<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_SOURCE, source_id);
        state.execute_command(std::move(flow_cmd));

        (void)card_db;
    }

    void GameLogicSystem::handle_select_target(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        exec.execution_paused = true;
        // ... set query ...
        (void)state; (void)inst;
    }

    void GameLogicSystem::handle_execute_command(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         if (!inst.args.contains("cmd")) return;

         try {
             CommandDef cmd = inst.args["cmd"].get<CommandDef>();

             int source_id = -1;
             auto v_source = exec.get_context_var("$source");
             if (std::holds_alternative<int>(v_source)) source_id = std::get<int>(v_source);

             int controller_id = state.active_player_id;
             auto v_ctrl = exec.get_context_var("$controller");
             if (std::holds_alternative<int>(v_ctrl)) controller_id = std::get<int>(v_ctrl);

             std::map<std::string, int> temp_ctx;
             for (const auto& kv : exec.context) {
                 if (std::holds_alternative<int>(kv.second)) {
                     temp_ctx[kv.first] = std::get<int>(kv.second);
                 }
             }

             CommandSystem::execute_command(state, cmd, source_id, controller_id, temp_ctx);

             for (const auto& kv : temp_ctx) {
                 exec.set_context_var(kv.first, kv.second);
             }

         } catch (const std::exception& e) {
             std::cerr << "[Pipeline] Failed to execute command: " << e.what() << std::endl;
         }
    }

    void GameLogicSystem::handle_game_result(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        std::string type = exec.resolve_string(inst.args.value("type", ""));
        int player = exec.resolve_int(inst.args.value("player", 0));

        if (type == "WIN_GAME") {
            if (player == 0) state.winner = GameResult::P1_WIN;
            else state.winner = GameResult::P2_WIN;
        } else if (type == "LOSE_GAME") {
            if (player == 0) state.winner = GameResult::P2_WIN;
            else state.winner = GameResult::P1_WIN;
        }

        exec.call_stack.clear();
        exec.execution_paused = true;
    }

}
