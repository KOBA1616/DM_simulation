#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/card/effect_system.hpp" // Added include for EffectSystem
#include "engine/systems/trigger_system/trigger_system.hpp" // Added for TriggerSystem
#include "engine/systems/card/passive_effect_system.hpp" // Added for PassiveEffectSystem
#include "engine/systems/restriction_system.hpp" // Added for RestrictionSystem
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/command_system.hpp" // Added for CommandSystem
#include "engine/systems/flow/phase_manager.hpp" // Added for PhaseManager
#include "engine/systems/mana/mana_system.hpp" // Added for ManaSystem
#include "engine/systems/breaker/breaker_system.hpp" // Added for BreakerSystem
#include "engine/utils/action_primitive_utils.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    // Forward declaration of helper
    void handle_check_s_trigger(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                const std::map<core::CardID, core::CardDefinition>& card_db);

    // Private helper implementation
    std::pair<core::Zone, core::PlayerID> GameLogicSystem::get_card_location(const core::GameState& state, int instance_id) {
        if (instance_id < 0 || (size_t)instance_id >= state.card_owner_map.size()) return {Zone::GRAVEYARD, 0};
        PlayerID owner = state.get_card_owner(instance_id);
        if (owner >= state.players.size()) return {Zone::GRAVEYARD, 0};
        const Player& p = state.players[owner];

        auto has = [&](const std::vector<CardInstance>& v) {
            for(const auto& c : v) if(c.instance_id == instance_id) return true;
            return false;
        };

        if(has(p.hand)) return {Zone::HAND, owner};
        if(has(p.battle_zone)) return {Zone::BATTLE, owner};
        if(has(p.mana_zone)) return {Zone::MANA, owner};
        if(has(p.shield_zone)) return {Zone::SHIELD, owner};
        if(has(p.graveyard)) return {Zone::GRAVEYARD, owner};
        if(has(p.deck)) return {Zone::DECK, owner};
        if(has(p.effect_buffer)) return {Zone::BUFFER, owner};
        if(has(p.stack)) return {Zone::STACK, owner};

        return {Zone::GRAVEYARD, owner};
    }

    void GameLogicSystem::dispatch_action(PipelineExecutor& pipeline, core::GameState& state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db) {
        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "dispatch_action entry type=" << static_cast<int>(action.type)
                     << " src=" << action.source_instance_id << " tgt=" << action.target_instance_id
                     << " slot=" << action.slot_index << "\n";
                diag.close();
            }
        } catch(...) {}
        // Map PlayerIntent to handler
        // Simplified mapping for now

        // NOTE: Architecture Transition (Action/Command Conversion)
        // The conversion from Action to Instruction here is a transitional implementation.
        // In the future, the architecture will migrate to one where AI and GUI directly generate Commands,
        // allowing this conversion layer to be removed.
        // Action/Command変換: src/engine/systems/game_logic_system.cpp の dispatch_action で行われている Action から Instruction への変換は過渡的な実装です。
        // 将来的にAIやGUIが直接 Command を生成するアーキテクチャへ完全移行することで、この層を削除できます。

        switch (action.type) {
            case PlayerIntent::PLAY_CARD:
            {
                // Convert Action to Instruction
                nlohmann::json args;
                args["card"] = action.source_instance_id;
                Instruction inst(InstructionOp::PLAY, args);
                handle_play_card(pipeline, state, inst, card_db);
                break;
            }
            case PlayerIntent::RESOLVE_PLAY:
            {
                nlohmann::json args;
                args["card"] = action.source_instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_PLAY";
                {
                    // Instead of invoking resolve handler inline (which may push child frames
                    // while the caller's pc is still at the same instruction), enqueue a
                    // RESOLVE_PLAY instruction block on the pipeline. The pipeline executor
                    // will run it in sequence and properly advance the parent frame pc.
                    auto block = std::make_shared<std::vector<Instruction>>();
                    block->push_back(inst);

                    // Duplicate suppression: if an identical RESOLVE_PLAY for the same card
                    // is already present in the call_stack, skip enqueue to avoid runaway
                    // repeated pushes. This is a minimal safe guard to help diagnose and
                    // prevent infinite enqueue loops.
                    bool duplicate = false;
                    int inst_card = -1;
                    if (inst.args.contains("card")) inst_card = inst.args["card"].get<int>();
                    for (const auto &frame : pipeline.call_stack) {
                        if (!frame.instructions || frame.instructions->empty()) continue;
                        const Instruction &f0 = (*frame.instructions)[0];
                        if (f0.args.contains("type") && f0.args["type"].is_string() && f0.args["type"] == "RESOLVE_PLAY") {
                            if (inst_card >= 0 && f0.args.contains("card") && f0.args["card"].get<int>() == inst_card) {
                                duplicate = true;
                                break;
                            }
                        }
                    }

                    size_t before_size = pipeline.call_stack.size();
                    int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                    // Compact telemetry: emit a single concise line rather than many verbose diagnostics.
                    std::fprintf(stderr, "[TRACE] RESOLVE_PLAY card=%d duplicate=%d parent_idx=%d before_size=%zu\n", inst_card, duplicate ? 1 : 0, parent_idx, before_size);

                    if (duplicate) {
                        // Advance parent pc to avoid re-executing the same instruction
                        if (parent_idx >= 0 && parent_idx < (int)pipeline.call_stack.size()) {
                            pipeline.call_stack[parent_idx].pc++;
                        }
                    } else {
                        auto const_block = std::static_pointer_cast<const std::vector<Instruction>>(block);
                        pipeline.call_stack.push_back({const_block, 0, LoopContext{}});
                        if (parent_idx >= 0 && parent_idx < (int)pipeline.call_stack.size()) {
                            pipeline.call_stack[parent_idx].pc++;
                        }
                    }
                }
                break;
            }
            case PlayerIntent::DECLARE_PLAY:
            {
                int iid = action.source_instance_id;

                // Identify source zone dynamically
                auto loc = get_card_location(state, iid);

                // Move to Stack from wherever it is
                auto cmd = std::make_unique<TransitionCommand>(iid, loc.first, Zone::STACK, loc.second);
                state.execute_command(std::move(cmd));
                break;
            }
            case PlayerIntent::PAY_COST:
            {
                int iid = action.source_instance_id;
                // Auto tap mana
                if (auto* c = state.get_card_instance(iid)) {
                    if (card_db.count(c->card_id)) {
                        const auto& def = card_db.at(c->card_id);
                        ManaSystem::auto_tap_mana(state, state.players[state.active_player_id], def, card_db);
                    }
                    // Mark as paid (using is_tapped flag on stack card)
                    // Use Command to allow undo
                    auto cmd = std::make_unique<MutateCommand>(iid, MutateCommand::MutationType::TAP);
                    state.execute_command(std::move(cmd));
                }
                break;
            }
            case PlayerIntent::ATTACK_CREATURE:
            case PlayerIntent::ATTACK_PLAYER:
            {
                nlohmann::json args;
                args["source"] = action.source_instance_id;
                args["target"] = action.target_instance_id;
                Instruction inst(InstructionOp::ATTACK, args);
                handle_attack(pipeline, state, inst, card_db);
                break;
            }
            case PlayerIntent::BLOCK:
            {
                nlohmann::json args;
                args["blocker"] = action.source_instance_id;
                Instruction inst(InstructionOp::BLOCK, args);
                handle_block(pipeline, state, inst, card_db);
                break;
            }
            case PlayerIntent::RESOLVE_BATTLE:
            {
                nlohmann::json args;
                args["attacker"] = action.source_instance_id;
                args["defender"] = action.target_instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_BATTLE";
                handle_resolve_battle(pipeline, state, inst, card_db);

                // Cleanup Pending Effect
                if (action.slot_index >= 0 && action.slot_index < (int)state.pending_effects.size()) {
                    // Safety check: Ensure type matches
                    if (state.pending_effects[action.slot_index].type == EffectType::RESOLVE_BATTLE) {
                        state.pending_effects.erase(state.pending_effects.begin() + action.slot_index);
                    }
                }
                break;
            }
            case PlayerIntent::BREAK_SHIELD:
            {
                nlohmann::json args;
                args["source_id"] = action.source_instance_id;

                if (action.slot_index >= 0 && action.slot_index < (int)state.pending_effects.size()) {
                     const auto& pe = state.pending_effects[action.slot_index];
                     if (!pe.target_instance_ids.empty()) {
                         args["shields"] = pe.target_instance_ids;
                     } else {
                         // Auto-select shields if not provided
                         const CardInstance* source = state.get_card_instance(action.source_instance_id);
                         if (source) {
                             int breaker_count = get_breaker_count(state, *source, card_db);
                             // Identify opponent
                             PlayerID opp = 1 - state.active_player_id; // Assume attack is active player
                             const auto& shields = state.players[opp].shield_zone;
                             std::vector<int> targets;
                             int count = std::min(breaker_count, (int)shields.size());
                             if (breaker_count >= 999) count = (int)shields.size(); // World Breaker

                             for (int i = 0; i < count; ++i) {
                                 targets.push_back(shields[i].instance_id);
                             }
                             args["shields"] = targets;
                         }
                     }
                }

                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "BREAK_SHIELD";
                handle_break_shield(pipeline, state, inst, card_db);

                // Cleanup Pending Effect
                if (action.slot_index >= 0 && action.slot_index < (int)state.pending_effects.size()) {
                    if (state.pending_effects[action.slot_index].type == EffectType::BREAK_SHIELD) {
                        state.pending_effects.erase(state.pending_effects.begin() + action.slot_index);
                    }
                }
                break;
            }
            case PlayerIntent::PASS:
            {
                // Use fast_forward to automatically progress through phases with no legal actions
                PhaseManager::fast_forward(state, card_db);
                break;
            }
            case PlayerIntent::MANA_CHARGE:
            {
            nlohmann::json args;
            args["card"] = action.source_instance_id;
            Instruction inst(InstructionOp::GAME_ACTION, args);
            inst.args["type"] = "MANA_CHARGE";
            handle_mana_charge(pipeline, state, inst);
            
            // DM Rule: After mana charge (max 1 per turn), automatically advance to next phase
            // Use fast_forward to skip phases with no legal actions
            PhaseManager::fast_forward(state, card_db);
                break;
            }
            case PlayerIntent::PLAY_CARD_INTERNAL:
            {
                nlohmann::json args;
                args["card"] = action.source_instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "PLAY_CARD_INTERNAL";

                ZoneDestination dest_override = ZoneDestination::NONE;

                // Lookup pending effect to determine destination if needed
                if (action.slot_index >= 0 && action.slot_index < (int)state.pending_effects.size()) {
                    const auto& eff = state.pending_effects[action.slot_index];

                    // Propagate Origin
                    if (eff.execution_context.count("$origin")) {
                        inst.args["origin_zone"] = eff.execution_context.at("$origin");
                    }

                    if (eff.effect_def.has_value()) {
                        for (const auto& act : eff.effect_def->actions) {
                            // Map all relevant destination zones to override flags
                            if (!act.destination_zone.empty()) {
                                if (act.destination_zone == "DECK_BOTTOM") {
                                    inst.args["dest_override"] = static_cast<int>(ZoneDestination::DECK_BOTTOM);
                                    dest_override = ZoneDestination::DECK_BOTTOM;
                                    break;
                                } else if (act.destination_zone == "DECK" || act.destination_zone == "DECK_TOP") {
                                    inst.args["dest_override"] = static_cast<int>(ZoneDestination::DECK_TOP);
                                    dest_override = ZoneDestination::DECK_TOP;
                                    break;
                                } else if (act.destination_zone == "HAND") {
                                    inst.args["dest_override"] = static_cast<int>(ZoneDestination::HAND);
                                    dest_override = ZoneDestination::HAND;
                                    break;
                                } else if (act.destination_zone == "MANA_ZONE") {
                                    inst.args["dest_override"] = static_cast<int>(ZoneDestination::MANA_ZONE);
                                    dest_override = ZoneDestination::MANA_ZONE;
                                    break;
                                } else if (act.destination_zone == "SHIELD_ZONE") {
                                    inst.args["dest_override"] = static_cast<int>(ZoneDestination::SHIELD_ZONE);
                                    dest_override = ZoneDestination::SHIELD_ZONE;
                                    break;
                                }
                            }
                        }
                    }
                }

                // New logic: Use Stack Lifecycle if no override (or standard play override)
                if (dest_override == ZoneDestination::NONE) {
                     // 1. Declare Play (Move to Stack)
                     int iid = action.source_instance_id;
                     auto loc = get_card_location(state, iid);
                     auto cmd = std::make_unique<TransitionCommand>(iid, loc.first, Zone::STACK, loc.second);
                     state.execute_command(std::move(cmd));

                     // 2. Pay Cost (Implicitly free or handled)
                     // Optimization: Skip explicit TAP on stack for internal plays as we resolve immediately.
                     // This prevents the card from entering Battle Zone tapped unless explicitly untapped.

                     // 3. Resolve Play
                     // We manually invoke handle_resolve_play via instruction queue to ensure it runs in pipeline
                     nlohmann::json resolve_args;
                     resolve_args["card"] = iid;
                     Instruction resolve_inst(InstructionOp::GAME_ACTION, resolve_args);
                     resolve_inst.args["type"] = "RESOLVE_PLAY";

                     // We push this directly to the CURRENT pipeline execution
                     // Since dispatch_action populates the pipeline, we can just call handle_resolve_play
                     // which pushes instructions to pipeline.call_stack
                     {
                         auto block = std::make_shared<std::vector<Instruction>>();
                         block->push_back(resolve_inst);

                         size_t before_size = pipeline.call_stack.size();
                         int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                         std::fprintf(stderr, "[TRACE] RESOLVE_PLAY_DECLARE card=%d parent_idx=%d before_size=%zu\n", iid, parent_idx, before_size);

                         auto const_block = std::static_pointer_cast<const std::vector<Instruction>>(block);
                         pipeline.call_stack.push_back({const_block, 0, LoopContext{}});
                         if (parent_idx >= 0 && parent_idx < (int)pipeline.call_stack.size()) {
                             pipeline.call_stack[parent_idx].pc++;
                         }
                     }
                } else {
                     // Legacy/Override path (direct move)
                     handle_play_card(pipeline, state, inst, card_db);
                }
                break;
            }
            case PlayerIntent::USE_ABILITY:
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

    void GameLogicSystem::resolve_play_from_stack(core::GameState& game_state, int stack_instance_id, int cost_reduction, core::SpawnSource spawn_source, core::PlayerID controller, const std::map<core::CardID, core::CardDefinition>& card_db, int evo_source_id, core::ZoneDestination dest_override) {
        // Resolve a play that is currently on the stack: invoke resolution logic
        // by constructing a PipelineExecutor, delegating to handle_resolve_play and
        // running the pipeline so compiled effects (and final MOVE to GRAVE) execute.
        (void)cost_reduction; (void)spawn_source; (void)evo_source_id; (void)dest_override;

        // Create pipeline and push resolve-play block
        dm::engine::systems::PipelineExecutor pipeline;
        // Construct an instruction that represents resolving the play
        nlohmann::json args;
        args["card"] = stack_instance_id;
        Instruction inst(InstructionOp::GAME_ACTION, args);
        inst.args["type"] = "RESOLVE_PLAY";

        // Let handle_resolve_play populate pipeline.call_stack
        handle_resolve_play(pipeline, game_state, inst, card_db);

        // Execute the pipeline to process effects and final move to graveyard
        pipeline.execute(nullptr, game_state, card_db);
    }

    void GameLogicSystem::handle_play_card(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                           const std::map<core::CardID, core::CardDefinition>& card_db) {
        // This function is now primarily for "Move to Playing Area" with overrides.
        // Standard "Play" goes through DECLARE_PLAY -> STACK -> RESOLVE_PLAY.

        int card_id = exec.resolve_int(inst.args.value("card", 0));
        int instance_id = card_id;
        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "handle_play_card entry inst_card_arg=" << card_id << "\n";
                diag.close();
            }
        } catch(...) {}

        CardInstance* card = state.get_card_instance(instance_id);
        if (!card) {
            try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){d<<"handle_play_card: card not found id="<<instance_id<<"\n";}} catch(...) {}
            return;
        }

        // --- Gatekeeper: Check for Prohibitions (CANNOT_SUMMON, etc.) ---
        int origin_int = exec.resolve_int(inst.args.value("origin_zone", -1));
        std::string origin_str = "";

        // If origin not provided, try to infer or default
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

        // Note: Evolution logic moved to handle_resolve_play

        std::vector<Instruction> generated;
        Zone dest = Zone::BATTLE;
        bool to_bottom = false;

        // Check for play_flags provided by the instruction (migration: Python converter may add these)
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

        // If this play is flagged as played-for-free, mark it in the flow so systems can observe it.
        if (play_for_free) {
            auto flow_cmd = std::make_unique<game_command::FlowCommand>(game_command::FlowCommand::FlowType::SET_PLAYED_WITHOUT_MANA, 1);
            state.execute_command(std::move(flow_cmd));
        }

        // If PUT_IN_PLAY is specified, force destination to Battle zone (even for spells)
        if (put_in_play) {
             // Delegate to resolve play which handles Evolution and Move to Battle
             nlohmann::json args = inst.args;
             Instruction resolve_inst(InstructionOp::GAME_ACTION, args);
             resolve_inst.args["type"] = "RESOLVE_PLAY";
             {
                 auto block = std::make_shared<std::vector<Instruction>>();
                 block->push_back(resolve_inst);

                 size_t before_size = exec.call_stack.size();
                 int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                 int parent_pc = -1;
                 if (parent_idx >= 0) parent_pc = exec.call_stack[parent_idx].pc;
                 std::cerr << "[DIAG RESOLVE_ENQUEUE PUT_IN_PLAY] before_size=" << before_size
                           << " parent_idx=" << parent_idx << " parent_pc=" << parent_pc
                           << " inst=" << resolve_inst.args.dump() << std::endl;

                 {
                     size_t before_size = exec.call_stack.size();
                     int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                     int parent_pc = -1;
                     if (parent_idx >= 0) parent_pc = exec.call_stack[parent_idx].pc;
                     std::string inst_dump = "{}";
                     if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
                     std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
                     exec.call_stack.push_back({block, 0, LoopContext{}});
                     size_t after_size = exec.call_stack.size();
                     std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
                     if (parent_idx >= 0 && parent_idx < (int)exec.call_stack.size()) {
                         exec.call_stack[parent_idx].pc++;
                         std::fprintf(stderr, "[DIAG ADVANCE] %s:%d advanced_parent_idx=%d new_pc=%d\n", __FILE__, __LINE__, parent_idx, exec.call_stack[parent_idx].pc);
                     }
                 }

                 size_t after_size = exec.call_stack.size();
                 std::cerr << "[DIAG RESOLVE_ENQUEUE PUT_IN_PLAY] after_size=" << after_size << std::endl;
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
             // Default destination for Play is STACK (for proper resolution)
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

        // If we moved to STACK here (legacy path), we should probably trigger RESOLVE_PLAY?
        // But handle_play_card is used by PLAY_CARD_INTERNAL (legacy path) which might expect atomic move.
        // The new PLAY_CARD_INTERNAL logic above handles the STACK->RESOLVE flow explicitly.
        // So this function is just a dumb mover now.

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             {
                 size_t before_size = exec.call_stack.size();
                 int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                 int parent_pc = -1;
                 if (parent_idx >= 0) parent_pc = exec.call_stack[parent_idx].pc;
                 std::string inst_dump = "{}";
                 if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
                 exec.call_stack.push_back({block, 0, LoopContext{}});
                 size_t after_size = exec.call_stack.size();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
                 if (parent_idx >= 0 && parent_idx < (int)exec.call_stack.size()) {
                     exec.call_stack[parent_idx].pc++;
                     std::fprintf(stderr, "[DIAG ADVANCE] %s:%d advanced_parent_idx=%d new_pc=%d\n", __FILE__, __LINE__, parent_idx, exec.call_stack[parent_idx].pc);
                 }
             }
        }
    }

    void GameLogicSystem::handle_apply_buffer_move(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                  const std::map<core::CardID, core::CardDefinition>& card_db) {
        // Expect inst.args["shields"] = [id...]
        std::vector<int> shield_ids;
        if (inst.args.find("shields") != inst.args.end()) {
            try {
                for (auto &v : inst.args["shields"]) shield_ids.push_back((int)v);
            } catch (...) {}
        }
        if (shield_ids.empty()) return;

        std::vector<Instruction> generated;

        // For each shield id, find it in players' effect_buffer, move to hand.
        for (int shield_id : shield_ids) {
            bool moved = false;
            for (auto pid = 0u; pid < state.players.size(); ++pid) {
                auto &buf = state.players[pid].effect_buffer;
                auto it = std::find_if(buf.begin(), buf.end(), [shield_id](const core::CardInstance& c){ return c.instance_id == shield_id; });
                if (it != buf.end()) {
                    core::CardInstance card = *it;
                    buf.erase(it);
                    // Move to hand (use add_card_to_zone if available)
                    state.add_card_to_zone(card, core::Zone::HAND, (core::PlayerID)pid);
                    moved = true;
                    // Check if player chose to trigger (context variable $strigger_<shield_id>)
                    std::string key = "$strigger_" + std::to_string(shield_id);
                    auto cv = exec.get_context_var(key);
                    // If shield trigger declared, auto-play. Other declarations
                    // (guard/strikeback) set context flags for later handling.
                    if (std::holds_alternative<int>(cv) && std::get<int>(cv) == 1) {
                        Instruction play_inst(InstructionOp::PLAY);
                        play_inst.args["card"] = shield_id;
                        generated.push_back(play_inst);
                    } else {
                        // Check guard/strikeback flags and preserve them in context
                        std::string gk = "$guard_" + std::to_string(shield_id);
                        std::string sk = "$sback_" + std::to_string(shield_id);
                        auto gv = exec.get_context_var(gk);
                        auto sv = exec.get_context_var(sk);
                        if (std::holds_alternative<int>(gv) && std::get<int>(gv) == 1) {
                            // Guard chosen — other systems can inspect $guard_<id>
                        }
                        if (std::holds_alternative<int>(sv) && std::get<int>(sv) == 1) {
                            // StrikeBack chosen — other systems can inspect $sback_<id>
                        }
                    }
                    break;
                }
            }
        }

        if (!generated.empty()) {
            auto block = std::make_shared<std::vector<Instruction>>(generated);
            {
                size_t before_size = exec.call_stack.size();
                int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                int parent_pc = -1;
                if (parent_idx >= 0) parent_pc = exec.call_stack[parent_idx].pc;
                std::string inst_dump = "{}";
                if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
                std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
                exec.call_stack.push_back({block, 0, LoopContext{}});
                size_t after_size = exec.call_stack.size();
                std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
                if (parent_idx >= 0 && parent_idx < (int)exec.call_stack.size()) {
                    exec.call_stack[parent_idx].pc++;
                    std::fprintf(stderr, "[DIAG ADVANCE] %s:%d advanced_parent_idx=%d new_pc=%d\n", __FILE__, __LINE__, parent_idx, exec.call_stack[parent_idx].pc);
                }
            }
        }
    }

    void GameLogicSystem::handle_resolve_play(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_resolve_play entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
        int instance_id = exec.resolve_int(inst.args.value("card", 0));
        const CardInstance* card = state.get_card_instance(instance_id);
        if (!card) return;

        // --- Gatekeeper: Prohibition Check (Point of No Return) ---
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

        if (card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);
            if (RestrictionSystem::instance().is_play_forbidden(state, *card, def, origin_str, card_db)) return;
        }

        if (!card_db.count(card->card_id)) return;
        const auto& def = card_db.at(card->card_id);

        // Record Stats
        state.on_card_play(card->card_id, state.turn_number, false, 0, card->owner);

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
        } else if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {

            bool is_evolution = def.keywords.evolution;

            if (is_evolution) {
                // Task B: Refined Evolution Filters
                FilterDef evo_filter;
                evo_filter.zones = {"BATTLE_ZONE"};
                evo_filter.races = def.races; // Evolution matches race
                evo_filter.owner = "SELF";

                // 1. Select Base
                Instruction select(InstructionOp::SELECT);
                select.args["filter"] = evo_filter;
                select.args["count"] = 1;
                select.args["out"] = "$evo_target";
                compiled_effects.push_back(select);

                // 2. Attach
                // Note: Attachment command logic needs to support Stack->Battle transition (under card).
                Instruction attach(InstructionOp::MOVE);
                attach.args["target"] = instance_id;
                attach.args["attach_to"] = "$evo_target";
                compiled_effects.push_back(attach);
            } else {
                // Move to Battle Zone
                nlohmann::json move_args;
                move_args["target"] = instance_id;
                move_args["to"] = "BATTLE";
                compiled_effects.emplace_back(InstructionOp::MOVE, move_args);

                // Ensure it enters Untapped (fix for Stack Tap-to-Pay mechanic)
                nlohmann::json untap_args;
                untap_args["type"] = "UNTAP";
                untap_args["target"] = instance_id;
                compiled_effects.emplace_back(InstructionOp::MODIFY, untap_args);
            }

            // Check Triggers (ON_PLAY)
            nlohmann::json trig_args;
            trig_args["type"] = "CHECK_CREATURE_ENTER_TRIGGERS";
            trig_args["card"] = instance_id;
            compiled_effects.emplace_back(InstructionOp::GAME_ACTION, trig_args);
        }

        if (!compiled_effects.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(compiled_effects);
             {
                 size_t before_size = exec.call_stack.size();
                 int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                 int parent_pc = -1;
                 if (parent_idx >= 0) parent_pc = exec.call_stack[parent_idx].pc;
                 std::string inst_dump = "{}";
                 if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
                 exec.call_stack.push_back({block, 0, LoopContext{}});
                 size_t after_size = exec.call_stack.size();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
                 if (parent_idx >= 0 && parent_idx < (int)exec.call_stack.size()) {
                     exec.call_stack[parent_idx].pc++;
                     std::fprintf(stderr, "[DIAG ADVANCE] %s:%d advanced_parent_idx=%d new_pc=%d\n", __FILE__, __LINE__, parent_idx, exec.call_stack[parent_idx].pc);
                 }
             }
        }
    }

    void GameLogicSystem::handle_attack(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                        const std::map<core::CardID, core::CardDefinition>& card_db) {
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_attack entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
            int instance_id = exec.resolve_int(inst.args.value("source", 0));
            int target_id = exec.resolve_int(inst.args.value("target", -1));
            try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_attack entry src="<<instance_id<<" tgt="<<target_id<<"\n";} } catch(...) {}

         const CardInstance* card = state.get_card_instance(instance_id);
         if (!card || !card_db.count(card->card_id)) return;
         const auto& def = card_db.at(card->card_id);

         // --- Gatekeeper: Strict Attack Validation ---
         if (RestrictionSystem::instance().is_attack_forbidden(state, *card, def, target_id, card_db)) return;

         // 1. Update Attack State using Commands
         auto cmd_src = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_SOURCE, instance_id);
         state.execute_command(std::move(cmd_src));

         auto cmd_tgt = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_TARGET, target_id);
         state.execute_command(std::move(cmd_tgt));

         // Reset blocking state implicitly by clearing blocker (if any from previous? should be new attack)
         auto cmd_blk = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_BLOCKING_CREATURE, -1);
         state.execute_command(std::move(cmd_blk));

         // Infer target player if target_id is -1 (Player Attack)
         int target_player = -1;
         if (target_id == -1) {
             target_player = 1 - state.active_player_id;
         }
         auto cmd_plyr = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_PLAYER, target_player);
         state.execute_command(std::move(cmd_plyr));

         auto cmd_tap = std::make_shared<MutateCommand>(instance_id, MutateCommand::MutationType::TAP);
         state.execute_command(std::move(cmd_tap));

         auto phase_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(Phase::BLOCK));
         state.execute_command(std::move(phase_cmd));

         (void)card_db;
    }

    void GameLogicSystem::handle_block(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_block entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
        int blocker_id = exec.resolve_int(inst.args.value("blocker", -1));
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_block entry blocker="<<blocker_id<<"\n";} } catch(...) {}
        if (blocker_id == -1) return;

        const CardInstance* blocker = state.get_card_instance(blocker_id);
        if (!blocker || !card_db.count(blocker->card_id)) return;
        const auto& def = card_db.at(blocker->card_id);

        // --- Gatekeeper: Strict Block Validation ---
        if (RestrictionSystem::instance().is_block_forbidden(state, *blocker, def, card_db)) return;

        // 1. Tap Blocker
        auto cmd_tap = std::make_shared<MutateCommand>(blocker_id, MutateCommand::MutationType::TAP);
        state.execute_command(std::move(cmd_tap));

        // 2. Update Attack State
        auto cmd_blk = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_BLOCKING_CREATURE, blocker_id);
        state.execute_command(std::move(cmd_blk));

        // 3. Trigger ON_BLOCK
        // Need access to Card Definition
        if (blocker && card_db.count(blocker->card_id)) {
             // Logic to queue triggers (if any)
             // Using TriggerSystem directly
             TriggerSystem::instance().resolve_trigger(state, TriggerType::ON_BLOCK, blocker_id, card_db);
        }

        // 4. Queue RESOLVE_BATTLE?
        // Attack/Block sequence usually:
        // Attack -> Block Phase -> (Player actions) -> Block Declared -> Battle
        // If Block is declared, we move to Battle.
        // Or if Block is optional step?
        // Usually after Block, we proceed to battle immediately unless there are triggers.
        // We can queue RESOLVE_BATTLE pending effect.

        PendingEffect pe(EffectType::RESOLVE_BATTLE, state.current_attack.source_instance_id, state.active_player_id);
        // Target is blocker
        pe.target_instance_ids = {blocker_id};

        TriggerSystem::instance().add_pending_effect(state, pe);

        (void)exec;
    }

    void GameLogicSystem::handle_resolve_battle(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                const std::map<core::CardID, core::CardDefinition>& card_db) {
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_resolve_battle entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
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

        std::vector<Instruction> generated;

        if (attacker_dies) {
            Instruction move(InstructionOp::MOVE);
            move.args["target"] = attacker_id;
            move.args["to"] = "GRAVEYARD";
            generated.push_back(move);
        }

        if (defender_dies) {
            Instruction move(InstructionOp::MOVE);
            move.args["target"] = defender_id;
            move.args["to"] = "GRAVEYARD";
            generated.push_back(move);
        }

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             {
                 size_t before_size = exec.call_stack.size();
                 int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                 int parent_pc = -1;
                 if (parent_idx >= 0) parent_pc = exec.call_stack[parent_idx].pc;
                 std::string inst_dump = "{}";
                 if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
                 exec.call_stack.push_back({block, 0, LoopContext{}});
                 size_t after_size = exec.call_stack.size();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
                 if (parent_idx >= 0 && parent_idx < (int)exec.call_stack.size()) {
                     exec.call_stack[parent_idx].pc++;
                     std::fprintf(stderr, "[DIAG ADVANCE] %s:%d advanced_parent_idx=%d new_pc=%d\n", __FILE__, __LINE__, parent_idx, exec.call_stack[parent_idx].pc);
                 }
             }
        }
    }

    int GameLogicSystem::get_creature_power(const core::CardInstance& creature, const core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;
        // Use mod to silence unused variable warning if loop body is empty or doesn't use it
        for (const auto& mod : game_state.active_modifiers) {
             (void)mod;
             // Logic for checking power modifier would go here
        }
        return power;
    }

    int GameLogicSystem::get_breaker_count(const core::GameState& state, const core::CardInstance& creature, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 1;
        const auto& def = card_db.at(creature.card_id);
        return BreakerSystem::get_breaker_count(state, creature, def);
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_break_shield entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
        // Support either single shield ("shield") or batch ("shields")
        std::vector<int> shield_ids;
        if (inst.args.find("shields") != inst.args.end()) {
            try {
                for (auto &v : inst.args["shields"]) shield_ids.push_back((int)v);
            } catch (...) { }
        } else {
            int single = exec.resolve_int(inst.args.value("shield", -1));
            if (single != -1) shield_ids.push_back(single);
        }

        if (shield_ids.empty()) return;

        // (debug prints removed)

        std::vector<Instruction> generated;

        // 0. Check Before Break Triggers (once per action)
        int source_id = exec.resolve_int(inst.args.value("source_id", -1));
        if (source_id != -1) {
             auto effects = TriggerSystem::instance().get_trigger_effects(state, TriggerType::BEFORE_BREAK_SHIELD, source_id, card_db);
             if (!effects.empty()) {
                 std::map<std::string, int> ctx;
                 for (const auto& eff : effects) {
                     EffectSystem::instance().compile_effect(state, eff, source_id, ctx, card_db, generated);
                 }
             }
        }
        // 1. For each shield id: remove from shield_zone into player's effect_buffer, and generate S-trigger checks as needed
        std::vector<int> to_apply_list;
        for (int shield_id : shield_ids) {
            bool found = false;
            // Find which player owns this shield and remove it into effect_buffer
            for (auto pid = 0u; pid < state.players.size(); ++pid) {
                auto &vec = state.players[pid].shield_zone;
                auto it = std::find_if(vec.begin(), vec.end(), [shield_id](const CardInstance& c){ return c.instance_id == shield_id; });
                if (it != vec.end()) {
                    CardInstance ci = *it;
                    vec.erase(it);
                    state.players[pid].effect_buffer.push_back(ci);
                    found = true;
                    break;
                }
            }
            if (!found) continue;

            // Always record for later apply
            to_apply_list.push_back(shield_id);

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
        }

        // After all checks, add APPLY_BUFFER_MOVE instruction which will move buffered shields to hand and auto-play as needed
        if (!to_apply_list.empty()) {
            nlohmann::json apply_args;
            apply_args["type"] = "APPLY_BUFFER_MOVE";
            apply_args["shields"] = to_apply_list;
            generated.emplace_back(InstructionOp::GAME_ACTION, apply_args);
        }

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

        // Helper for CHECK_S_TRIGGER (static method implementation)
        void GameLogicSystem::handle_check_s_trigger(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                      const std::map<core::CardID, core::CardDefinition>& card_db) {
           try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_check_s_trigger entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
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

               // Offer multiple declaration options: No / Shield Trigger / Guard Strike / Strike Back
               std::vector<std::string> opts = {"No", "Shield Trigger", "Guard Strike", "Strike Back"};
               auto cmd = std::make_unique<QueryCommand>("SELECT_OPTION", std::vector<int>{}, std::map<std::string, int>{}, opts);
               state.execute_command(std::move(cmd));
               return;
           }

           // Map selected option index (expected to be int) to specific flags
           if (std::holds_alternative<int>(val)) {
               int sel = std::get<int>(val);
               // 0 = No, 1 = Shield Trigger, 2 = Guard Strike, 3 = Strike Back
               if (sel == 1) {
                   exec.set_context_var(decision_key, 1);
               } else if (sel == 2) {
                   std::string k = "$guard_" + std::to_string(card_id);
                   exec.set_context_var(k, 1);
               } else if (sel == 3) {
                   std::string k = "$sback_" + std::to_string(card_id);
                   exec.set_context_var(k, 1);
               } else {
                   exec.set_context_var(decision_key, 0);
               }
           }

           // If Shield Trigger selected, auto-play as before
           ContextValue post = exec.get_context_var(decision_key);
           if (std::holds_alternative<int>(post) && std::get<int>(post) == 1) {
               (void)card_db;
               Instruction play_inst(InstructionOp::PLAY);
               play_inst.args["card"] = card_id;

               auto block = std::make_shared<std::vector<Instruction>>();
               block->push_back(play_inst);
               exec.call_stack.push_back({block, 0, LoopContext{}});
           }
        }

    void GameLogicSystem::handle_mana_charge(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         int card_id = exec.resolve_int(inst.args.value("card", 0));
         if (card_id < 0) return;

         Instruction move = utils::ActionPrimitiveUtils::create_mana_charge_instruction(card_id);

         auto block = std::make_shared<std::vector<Instruction>>();
         block->push_back(move);
         exec.call_stack.push_back({block, 0, LoopContext{}});
         
         // Mark that mana has been charged this turn (max 1 per turn rule)
         state.turn_stats.mana_charged_this_turn = true;
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
            auto tap_cmd = std::make_shared<MutateCommand>(source_id, MutateCommand::MutationType::TAP);
            state.execute_command(std::move(tap_cmd));
        }

        auto flow_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_SOURCE, source_id);
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

    void GameLogicSystem::handle_check_creature_enter_triggers(PipelineExecutor& exec, GameState& state, const Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db) {
        int card_id = exec.resolve_int(inst.args.value("card", 0));
        TriggerSystem::instance().resolve_trigger(state, TriggerType::ON_PLAY, card_id, card_db);
        TriggerSystem::instance().resolve_trigger(state, TriggerType::ON_OTHER_ENTER, card_id, card_db);
    }

    void GameLogicSystem::handle_game_result(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        std::string type = exec.resolve_string(inst.args.value("type", ""));
        int player = exec.resolve_int(inst.args.value("player", 0));

        GameResult res = GameResult::NONE;

        if (type == "WIN_GAME") {
            if (player == 0) res = GameResult::P1_WIN;
            else res = GameResult::P2_WIN;
        } else if (type == "LOSE_GAME") {
            if (player == 0) res = GameResult::P2_WIN;
            else res = GameResult::P1_WIN;
        }

        if (res != GameResult::NONE) {
            auto cmd = std::make_unique<GameResultCommand>(res);
            state.execute_command(std::move(cmd));
        }

        exec.call_stack.clear();
        exec.execution_paused = true;
    }

}
