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
                // Currently ActionType::RESOLVE_BATTLE does not carry target info in standard fields cleanly
                // But ActionGenerator::resolve_battle sets target_instance_id for creature-creature battles
                // If one of them is creature.
                // Assuming action.source = attacker, action.target = defender
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
                if (iid >= 0) {
                     auto cmd = std::make_unique<TransitionCommand>(iid, Zone::HAND, Zone::MANA, pid);
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
        // Execute remaining?
        // If execution paused, it handles it via state.active_pipeline
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
            if (def.keywords.evolution) {
                is_evolution = true;
                // Task B: Refined Evolution Filters
                evo_filter.zones = {"BATTLE_ZONE"};
                evo_filter.races = def.races; // Evolution matches race

                // Specific evolution conditions (NEO, etc.) would be handled here by inspecting CardDefinition
                // or a specific evolution field if added.
                // For strict filtering, TargetUtils::is_valid_target uses races/civs if set.
                // Since we set races here, it enforces race matching.

                // Also enforce owner
                evo_filter.owner = "SELF";
            }
        }

        if (is_evolution) {
            std::string selection_key = "$evo_target";

            ContextValue val = exec.get_context_var(selection_key);
            std::vector<int> targets;
            if (std::holds_alternative<std::vector<int>>(val)) {
                targets = std::get<std::vector<int>>(val);
            }

            if (targets.empty()) {
                exec.execution_paused = true;
                exec.waiting_for_key = selection_key;
                state.waiting_for_user_input = true;

                // Populate valid targets for UI using TargetUtils
                // We can reuse PipelineExecutor::handle_select logic by creating a dummy instruction?
                // Or just manual query.
                // Reusing handle_select logic is better but we are inside handle_play_card.

                // Manual query population:
                std::vector<int> valid_targets;
                // Iterate battle zone
                const auto& battle = state.players[state.active_player_id].battle_zone;
                for (const auto& c : battle) {
                    if (card_db.count(c.card_id)) {
                        const auto& def = card_db.at(c.card_id);
                         if (TargetUtils::is_valid_target(c, def, evo_filter, state, state.active_player_id, state.active_player_id)) {
                             valid_targets.push_back(c.instance_id);
                         }
                    }
                }

                state.pending_query = GameState::QueryContext{
                    0, "SELECT_TARGET", {{"min", 1}, {"max", 1}}, valid_targets, {}
                };
                return;
            }

            int base_id = targets[0];
            auto cmd = std::make_unique<AttachCommand>(instance_id, base_id, Zone::HAND);
            state.execute_command(std::move(cmd));

        } else {
             const auto& def = card_db.at(card->card_id);
             Zone dest = Zone::BATTLE;
             if (def.type == CardType::SPELL) {
                 dest = Zone::STACK;
             }

             auto cmd = std::make_unique<TransitionCommand>(instance_id, Zone::HAND, dest, state.active_player_id);
             state.execute_command(std::move(cmd));

        }

        // Push RESOLVE_PLAY for Spells immediately
        if (card_db.at(card->card_id).type == CardType::SPELL && !is_evolution) {
             // Create a new block for resolution
             nlohmann::json resolve_args;
             resolve_args["card"] = instance_id;
             resolve_args["type"] = "RESOLVE_PLAY";

             std::vector<Instruction> block;
             block.emplace_back(InstructionOp::GAME_ACTION, resolve_args);

             exec.call_stack.push_back({std::make_shared<std::vector<Instruction>>(block), 0, LoopContext{}});
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

        // For Spells, we execute all effects.
        // For Creatures, this is ON_PLAY (CIP).
        // Currently EffectSystem::compile_effect handles "trigger" checks?
        // If we just want to execute the "main" effect of a spell, we iterate def.effects.

        if (def.type == CardType::SPELL) {
            for (const auto& eff : def.effects) {
                 // Spells usually don't have triggers like ON_PLAY explicitly in JSON?
                 // Or they do, but with trigger="NONE" or implicit?
                 // Usually Spell effects are just the list.
                 // We should check if EffectSystem needs a trigger type.
                 EffectSystem::instance().compile_effect(state, eff, instance_id, ctx, card_db, compiled_effects);
            }

            // 2. Move to Graveyard (after effects)
            // We append this to the END of compiled effects.
            nlohmann::json move_args;
            move_args["target"] = instance_id;
            move_args["to"] = "GRAVEYARD";
            compiled_effects.emplace_back(InstructionOp::MOVE, move_args);
        } else {
            // Creature ON_PLAY
            // Handled via TriggerSystem usually?
            // If we are here, it means we are explicitly resolving play.
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

         // Transition to BLOCK phase
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

        // Calculate powers
        int power_attacker = get_creature_power(*attacker, state, card_db);
        int power_defender = get_creature_power(*defender, state, card_db);

        bool attacker_dies = false;
        bool defender_dies = false;

        // Slayer check
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
            // Equal power: Both die
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

        // Apply modifiers from game_state.active_modifiers or card internal modifiers
        // For now, simple implementation
        // Check game state active modifiers
        for (const auto& mod : game_state.active_modifiers) {
             // ... Check target ...
             // For simplicity, we assume no modifiers in this stub unless we implement full matching
        }

        return power;
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        int shield_id = exec.resolve_int(inst.args.value("shield", -1));
        if (shield_id == -1) return;

        auto cmd = std::make_unique<TransitionCommand>(shield_id, Zone::SHIELD, Zone::HAND, state.active_player_id);
        state.execute_command(std::move(cmd));

        const auto* card = state.get_card_instance(shield_id);
        if (card && card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);
            if (def.keywords.shield_trigger) {

                std::string decision_key = "$strigger_decision_" + std::to_string(shield_id);
                ContextValue val = exec.get_context_var(decision_key);

                bool use_trigger = false;
                bool decision_made = false;

                if (std::holds_alternative<int>(val)) {
                    use_trigger = std::get<int>(val) == 1;
                    decision_made = true;
                } else if (std::holds_alternative<std::vector<int>>(val)) {
                     const auto& vec = std::get<std::vector<int>>(val);
                     if(!vec.empty()) use_trigger = vec[0] == 1;
                     decision_made = true;
                }

                if (!decision_made) {
                    exec.waiting_for_key = decision_key;
                    exec.execution_paused = true;

                    state.waiting_for_user_input = true;
                    state.pending_query = GameState::QueryContext{
                        0, "SELECT_OPTION", {}, {}, {"No", "Yes"} // 0=No, 1=Yes
                    };
                    return;
                }

                if (use_trigger) {
                    // Task A: Complete Effect Resolution
                    // 1. Play (Free)
                    Zone dest = (def.type == CardType::SPELL) ? Zone::GRAVEYARD : Zone::BATTLE;
                    auto play_cmd = std::make_unique<TransitionCommand>(shield_id, Zone::HAND, dest, state.active_player_id);
                    state.execute_command(std::move(play_cmd));

                    // 2. Compile Effects
                    std::vector<Instruction> compiled_effects;
                    std::map<std::string, int> ctx; // Empty initial context

                    for (const auto& eff : def.effects) {
                         // Check trigger? S-Trigger execution usually executes ALL main effects (Spells)
                         // Or CIP effects (Creatures).
                         if (def.type == CardType::SPELL) {
                             EffectSystem::instance().compile_effect(state, eff, shield_id, ctx, card_db, compiled_effects);
                         } else {
                             // For creatures, playing it triggers ON_PLAY normally via `resolve_trigger`.
                             // However, usually S-Trigger creatures have CIP effects.
                             // The `TransitionCommand` moves it to Battle Zone.
                             // The GameLoop or TriggerSystem should pick up the ON_PLAY trigger.
                             // BUT, we are inside a pipeline execution (BREAK_SHIELD).
                             // We might need to ensure triggers are processed.
                             // Since we are handling this inline, we assume standard trigger system handles CIP.
                             // So we only compile effects for SPELLS here.
                         }
                    }

                    if (!compiled_effects.empty()) {
                         // Inject instructions into the pipeline
                         auto block = std::make_shared<std::vector<Instruction>>(compiled_effects);
                         exec.call_stack.push_back({block, 0, LoopContext{}});
                    }
                }
            }
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
            // Set Summoning Sickness (turn_played = current_turn)
            new_creature->turn_played = state.turn_number;

            // Set Tapped
            auto tap_cmd = std::make_unique<MutateCommand>(source_id, MutateCommand::MutationType::TAP);
            state.execute_command(std::move(tap_cmd));
        }

        // 4. Update Attack Source
        auto flow_cmd = std::make_unique<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_SOURCE, source_id);
        state.execute_command(std::move(flow_cmd));

        (void)card_db;
    }

    void GameLogicSystem::handle_select_target(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        exec.execution_paused = true;
        // ... set query ...
        (void)state; (void)inst;
    }

    // New: Handle Command Execution within Pipeline
    void GameLogicSystem::handle_execute_command(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         if (!inst.args.contains("cmd")) return;

         try {
             CommandDef cmd = inst.args["cmd"].get<CommandDef>();

             // Resolve necessary IDs
             int source_id = -1;
             auto v_source = exec.get_context_var("$source");
             if (std::holds_alternative<int>(v_source)) source_id = std::get<int>(v_source);

             int controller_id = state.active_player_id;
             auto v_ctrl = exec.get_context_var("$controller");
             if (std::holds_alternative<int>(v_ctrl)) controller_id = std::get<int>(v_ctrl);

             // Construct a temporary execution context map for CommandSystem compatibility
             // Sync FROM pipeline
             std::map<std::string, int> temp_ctx;
             for (const auto& kv : exec.context) {
                 if (std::holds_alternative<int>(kv.second)) {
                     temp_ctx[kv.first] = std::get<int>(kv.second);
                 }
             }

             CommandSystem::execute_command(state, cmd, source_id, controller_id, temp_ctx);

             // Sync TO pipeline (if CommandSystem modified context)
             for (const auto& kv : temp_ctx) {
                 exec.set_context_var(kv.first, kv.second);
             }

         } catch (const std::exception& e) {
             std::cerr << "[Pipeline] Failed to execute command: " << e.what() << std::endl;
         }
    }

}
