#include "game_logic_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/systems/rules/condition_system.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"
#include "engine/systems/effects/effect_system.hpp"
#include "engine/systems/effects/trigger_system.hpp"
#include "engine/systems/effects/passive_effect_system.hpp"
#include "engine/systems/rules/restriction_system.hpp"
#include "core/game_state.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/infrastructure/commands/command_system.hpp"
#include "engine/systems/flow/phase_system.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include "engine/systems/mechanics/payment_plan.hpp"
#include "engine/systems/breaker/breaker_system.hpp"
#include "engine/systems/mechanics/battle_system.hpp"
#include "engine/systems/mechanics/shield_system.hpp"
#include "engine/systems/mechanics/play_system.hpp"
#include "engine/systems/mechanics/cost_payment_system.hpp"
#include "engine/utils/action_primitive_utils.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <optional>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    // Forward declaration of helper
    void handle_check_s_trigger(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                const std::map<core::CardID, core::CardDefinition>& card_db);

    // Private helper implementation
    std::pair<core::Zone, core::PlayerID> GameLogicSystem::get_card_location(const core::GameState& state, int instance_id) {
        if (instance_id < 0 || (size_t)instance_id >= state.card_owner_map.size()) return {Zone::GRAVEYARD, PlayerID{0}};
        PlayerID owner = state.get_card_owner(instance_id);
        if (owner >= state.players.size()) return {Zone::GRAVEYARD, PlayerID{0}};
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


    void GameLogicSystem::dispatch_command(PipelineExecutor& pipeline, core::GameState& state, const core::CommandDef& cmd, const std::map<core::CardID, core::CardDefinition>& card_db) {
        std::cerr << "\n=== dispatch_command called ===" << std::endl;
        std::cerr << "Command type: " << static_cast<int>(cmd.type) << std::endl;
        std::cerr << "PLAY_FROM_ZONE value: " << static_cast<int>(core::CommandType::PLAY_FROM_ZONE) << std::endl;

        // NOTE: dispatch_command は外部からのユーザー/AI 入力をパイプラインに反映する
        // 唯一の責務ポイントとして扱われるべきです。
        // - waiting_for_user_input / pipeline.execution_paused の両方のケースで
        //   入力を受け取り、context 変数へ保存し、パイプライン実行を再開します。
        // - 他箇所でコンテキストを直接書き換えないでください。

        switch (cmd.type) {
            case core::CommandType::PLAY_FROM_ZONE:
            {
                // Stack Lifecycle: DECLARE_PLAY -> PAY_COST -> RESOLVE_PLAY
                int iid = cmd.instance_id;
                bool is_spell_side = (cmd.amount == 1); // Convention: amount=1 means spell side

                // Step 1: DECLARE_PLAY - Move to Stack
                auto loc = get_card_location(state, iid);
                std::cerr << "[PLAY TRACE] instance_id=" << iid
                          << " from_zone=" << static_cast<int>(loc.first)
                          << " owner=" << static_cast<int>(loc.second)
                          << std::endl;
                auto declare_cmd = std::make_unique<game_command::TransitionCommand>(iid, loc.first, Zone::STACK, loc.second);
                state.execute_command(std::move(declare_cmd));
                
                // Step 2: PAY_COST - Auto tap mana
                if (auto* c = state.get_card_instance(iid)) {
                    if (card_db.count(c->card_id)) {
                        const auto& base_def = card_db.at(c->card_id);
                        const auto& def = (is_spell_side && base_def.spell_side) ? *base_def.spell_side : base_def;

                        // 再発防止: テスト失敗時にカード定義へ cost_reductions が載っているかを
                        // 実行時に必ず可視化して、JSONロード/レジストリ経路の切り分けを容易にする。
                        std::cerr << "[PAYMENT TRACE] card_id=" << def.id
                                  << " base_cost=" << def.cost
                                  << " cost_reductions.size=" << def.cost_reductions.size()
                                  << std::endl;

                        bool payment_success = false;
                            std::cerr << "[PAYMENT TRACE] cmd.payment_mode='" << cmd.payment_mode
                                      << "' cmd.str_param='" << cmd.str_param << "' cmd.str_val='" << cmd.str_val
                                      << "' reduction_id='" << cmd.reduction_id << "' payment_units=" << cmd.payment_units << std::endl;
                        if (cmd.payment_mode == "ACTIVE_PAYMENT" || cmd.str_param == "ACTIVE_PAYMENT") {
                            // Determine requested units: prefer explicit `payment_units` when present
                            int requested_units = 0;
                            if (cmd.payment_units > 0) requested_units = cmd.payment_units;
                            else if (cmd.target_instance > 0) requested_units = cmd.target_instance;
                            else if (cmd.target_slot_index > 0) requested_units = cmd.target_slot_index;
                            // If no explicit units were provided but an ACTIVE_PAYMENT exists,
                            // default to 1 unit to avoid ambiguous zero-selection cases.
                            if (requested_units == 0) requested_units = 1;

                            const CostReductionDef* selected_reduction = nullptr;
                            // Prefer explicit reduction_id (stable identifier) when available
                            if (!cmd.reduction_id.empty()) {
                                for (const auto& reduction : def.cost_reductions) {
                                    if (reduction.type != ReductionType::ACTIVE_PAYMENT) continue;
                                    if (reduction.id == cmd.reduction_id) {
                                        selected_reduction = &reduction;
                                        break;
                                    }
                                }
                            }
                            // Fallback: legacy name-based selection (str_val)
                            if (!selected_reduction && !cmd.str_val.empty()) {
                                for (const auto& reduction : def.cost_reductions) {
                                    if (reduction.type != ReductionType::ACTIVE_PAYMENT) continue;
                                    if (reduction.name == cmd.str_val) {
                                        selected_reduction = &reduction;
                                        break;
                                    }
                                }
                            }
                            // Final fallback: pick first ACTIVE_PAYMENT entry
                            if (!selected_reduction) {
                                for (const auto& reduction : def.cost_reductions) {
                                    if (reduction.type == ReductionType::ACTIVE_PAYMENT) {
                                        selected_reduction = &reduction;
                                        break;
                                    }
                                }
                            }

                            if (selected_reduction && requested_units > 0) {
                                const int max_units = CostPaymentSystem::calculate_max_units(
                                    state, state.active_player_id, *selected_reduction,
                                    card_db);
                                // Trace: report selection and capacity
                                std::cerr << "[PAYMENT TRACE] requested_units=" << requested_units
                                          << " max_units=" << max_units << std::endl;
                                if (requested_units <= max_units) {
                                    const int actual_reduction =
                                        CostPaymentSystem::execute_payment(
                                            state, state.active_player_id,
                                            *selected_reduction, requested_units,
                                            card_db);

                                    std::optional<std::string> active_name;
                                    if (!selected_reduction->id.empty()) active_name = selected_reduction->id;
                                    else if (!selected_reduction->name.empty()) active_name = selected_reduction->name;

                                    auto plan = dm::engine::evaluate_cost(def, requested_units, active_name, requested_units);
                                    int effective_cost = plan.final_cost;

                                    std::cerr << "[PAYMENT TRACE] selected_reduction.id='" << (selected_reduction->id.empty() ? "(empty)" : selected_reduction->id)
                                              << "' name='" << (selected_reduction->name.empty() ? "(empty)" : selected_reduction->name)
                                              << "' reduction_amount=" << selected_reduction->reduction_amount
                                              << " actual_reduction=" << actual_reduction
                                              << " plan.final_cost=" << effective_cost << std::endl;

                                    // Ensure the payment calculation affects final mana tapping.
                                    payment_success = ManaSystem::auto_tap_mana(
                                        state, state.players[state.active_player_id],
                                        def, effective_cost, card_db);

                                    std::cerr << "[PAYMENT TRACE] mana_auto_tap_result=" << (payment_success ? "OK" : "FAIL") << std::endl;
                                } else {
                                    std::cerr << "[PAYMENT TRACE] requested_units > max_units, cannot execute payment" << std::endl;
                                }
                            }
                        } else {
                            // Fallback: standard auto-tap when no ACTIVE_PAYMENT requested
                            payment_success = ManaSystem::auto_tap_mana(
                                state, state.players[state.active_player_id], def,
                                card_db);
                        }

                        if (payment_success) {
                            // Mark as paid (tap the stack card)
                            auto tap_cmd = std::make_unique<game_command::MutateCommand>(iid, game_command::MutateCommand::MutationType::TAP);
                            state.execute_command(std::move(tap_cmd));
                            
                            // Step 3: RESOLVE_PLAY - Execute resolution
                            nlohmann::json resolve_args;
                            resolve_args["card"] = iid;
                            resolve_args["is_spell_side"] = is_spell_side;
                            Instruction resolve_inst(InstructionOp::GAME_ACTION, resolve_args);
                            resolve_inst.args["type"] = "RESOLVE_PLAY";
                            handle_resolve_play(pipeline, state, resolve_inst, card_db);
                        } else {
                            // Payment failed - return card from stack to hand
                            auto return_cmd = std::make_unique<game_command::TransitionCommand>(iid, Zone::STACK, Zone::HAND, state.active_player_id);
                            state.execute_command(std::move(return_cmd));
                        }
                    }
                } else {
                    std::cerr << "[PLAY TRACE] get_card_instance failed for instance_id=" << iid << std::endl;
                }
                break;
            }
            case core::CommandType::ATTACK_CREATURE:
            case core::CommandType::ATTACK_PLAYER:
            {
                nlohmann::json args;
                args["source"] = cmd.instance_id;
                args["target"] = cmd.target_instance;
                Instruction inst(InstructionOp::ATTACK, args);
                BattleSystem::instance().handle_attack(pipeline, state, inst, card_db);
                break;
            }
            case core::CommandType::BLOCK:
            {
                nlohmann::json args;
                args["blocker"] = cmd.instance_id;
                Instruction inst(InstructionOp::BLOCK, args);
                BattleSystem::instance().handle_block(pipeline, state, inst, card_db);
                break;
            }
            case core::CommandType::RESOLVE_BATTLE:
            {
                nlohmann::json args;
                args["attacker"] = cmd.instance_id;
                args["defender"] = cmd.target_instance;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_BATTLE";
                BattleSystem::instance().handle_resolve_battle(pipeline, state, inst, card_db);

                // Cleanup Pending Effect if amount implies slot_index
                // (Assuming RESOLVE_BATTLE command is generated from a pending effect context)
                // Note: CommandDef doesn't track slot_index explicitly, but we can pass it in amount if needed.
                // Logic in dispatch_action used slot_index to clear pending effect.
                // We should assume if amount is provided and valid, it refers to pending effect index.
                if (cmd.amount >= 0 && cmd.amount < (int)state.pending_effects.size()) {
                    if (state.pending_effects[cmd.amount].type == EffectType::RESOLVE_BATTLE) {
                        state.pending_effects.erase(state.pending_effects.begin() + cmd.amount);
                    }
                }
                break;
            }
            case core::CommandType::BREAK_SHIELD:
            {
                nlohmann::json args;
                args["source_id"] = cmd.instance_id;

                // Handle Pending Effect Logic (if amount is index)
                bool from_pending = false;
                if (cmd.amount >= 0 && cmd.amount < (int)state.pending_effects.size()) {
                     const auto& pe = state.pending_effects[cmd.amount];
                     if (pe.type == EffectType::BREAK_SHIELD) {
                         from_pending = true;
                         if (!pe.target_instance_ids.empty()) {
                             args["shields"] = pe.target_instance_ids;
                         } else {
                             // Auto-select shields if not provided
                             const CardInstance* source = state.get_card_instance(cmd.instance_id);
                             if (source) {
                                 int breaker_count = get_breaker_count(state, *source, card_db);
                                 PlayerID opp = 1 - state.active_player_id;
                                 const auto& shields = state.players[opp].shield_zone;
                                 std::vector<int> targets;
                                 int count = std::min(breaker_count, (int)shields.size());
                                 if (breaker_count >= 999) count = (int)shields.size();

                                 for (int i = 0; i < count; ++i) {
                                     targets.push_back(shields[i].instance_id);
                                 }
                                 args["shields"] = targets;
                             }
                         }
                     }
                }

                // If not from pending, maybe cmd.target_instance specifies single shield?
                if (!from_pending && cmd.target_instance > 0) {
                    args["shields"] = std::vector<int>{cmd.target_instance};
                }

                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "BREAK_SHIELD";
                ShieldSystem::instance().handle_break_shield(pipeline, state, inst, card_db);

                // Cleanup Pending Effect
                if (from_pending) {
                    state.pending_effects.erase(state.pending_effects.begin() + cmd.amount);
                }
                break;
            }
            case core::CommandType::PASS:
            {
                dm::engine::flow::PhaseSystem::instance().next_phase(state, card_db);
                break;
            }
            case core::CommandType::MANA_CHARGE:
            {
                nlohmann::json args;
                args["card"] = cmd.instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "MANA_CHARGE";
                handle_mana_charge(pipeline, state, inst);
                dm::engine::flow::PhaseSystem::instance().fast_forward(state, card_db);
                break;
            }
            case core::CommandType::USE_ABILITY:
            {
                nlohmann::json args;
                args["source"] = cmd.instance_id;
                args["target"] = cmd.target_instance;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "USE_ABILITY";
                handle_use_ability(pipeline, state, inst, card_db);
                break;
            }
            case core::CommandType::RESOLVE_EFFECT:
            {
                nlohmann::json args;
                args["slot_index"] = cmd.amount; // Use amount as slot_index
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_EFFECT";
                handle_resolve_effect(pipeline, state, inst, card_db);
                break;
            }
            case core::CommandType::RESOLVE_PLAY:
            {
                nlohmann::json args;
                args["card"] = cmd.instance_id;
                args["is_spell_side"] = (cmd.amount == 1);
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "RESOLVE_PLAY";

                // Logic mirroring dispatch_action's RESOLVE_PLAY handling:
                // Enqueue RESOLVE_PLAY instruction block
                auto block = std::make_shared<std::vector<Instruction>>();
                block->push_back(inst);

                // NOTE: 再発防止 — duplicate チェックは現経路では不要（pipeline での再帰防止は pipeline 自体が担う）
                // bool duplicate = false; は削除済み。ここで local 変数を残すと C4189 でビルドが停止する。

                pipeline.call_stack.push_back({std::static_pointer_cast<const std::vector<Instruction>>(block), 0, LoopContext{}});
                break;
            }
            case core::CommandType::SELECT_TARGET:
            {
                // 再発防止: SELECT_TARGET はユーザー入力コマンドなので、
                //   paused フラグ状態に依存せず常に context へ反映する。
                std::vector<int> selection = {cmd.instance_id};
                std::string ctx_key = pipeline.waiting_for_key;
                if (ctx_key.empty()) ctx_key = "$selection";
                if (ctx_key.rfind("$", 0) != 0) ctx_key = std::string("$") + ctx_key;
                try {
                    std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                    if (lout) {
                        lout << "DISPATCH_SET_CONTEXT key=" << ctx_key
                             << " value=[";
                        for (size_t i = 0; i < selection.size(); ++i) {
                            if (i) lout << ",";
                            lout << selection[i];
                        }
                        lout << "]\n";
                        lout.close();
                    }
                } catch (...) {}
                std::cerr << "DISPATCH_SET_CONTEXT key=" << ctx_key
                          << " value=[";
                for (size_t i = 0; i < selection.size(); ++i) {
                    if (i) std::cerr << ",";
                    std::cerr << selection[i];
                }
                std::cerr << "]\n";
                pipeline.set_context_var(ctx_key, selection);
                pipeline.execution_paused = false;
                state.waiting_for_user_input = false;
                break;
            }
            case core::CommandType::SELECT_NUMBER:
            {
                std::string ctx_key = pipeline.waiting_for_key;
                if (ctx_key.empty()) ctx_key = "$input";
                if (ctx_key.rfind("$", 0) != 0) ctx_key = std::string("$") + ctx_key;
                try {
                    std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                    if (lout) {
                        lout << "DISPATCH_SET_CONTEXT key=" << ctx_key
                             << " value=" << cmd.target_instance << "\n";
                        lout.close();
                    }
                } catch (...) {}
                std::cerr << "DISPATCH_SET_CONTEXT key=" << ctx_key
                          << " value=" << cmd.target_instance << "\n";
                pipeline.set_context_var(ctx_key, cmd.target_instance);
                pipeline.execution_paused = false;
                state.waiting_for_user_input = false;
                break;
            }
            case core::CommandType::CHOICE:
            {
                if (pipeline.execution_paused) {
                    pipeline.set_context_var(pipeline.waiting_for_key, cmd.target_instance);
                    pipeline.execution_paused = false;
                }
                break;
            }
            case core::CommandType::SELECT_FROM_BUFFER:
            {
                // 再発防止: SELECT_FROM_BUFFER の dispatch を実装しないとパイプラインが
                //   再開されず、ゲームがフリーズする。選択されたカード ID をコンテキスト
                //   変数（vector<int>）に追記してパイプライン実行を再開する。
                std::string ctx_key = pipeline.waiting_for_key;
                if (ctx_key.empty()) ctx_key = "$buffer_select";
                if (ctx_key.rfind("$", 0) != 0) ctx_key = std::string("$") + ctx_key;
                auto v = pipeline.get_context_var(ctx_key);
                std::vector<int> selection;
                if (std::holds_alternative<std::vector<int>>(v)) {
                    selection = std::get<std::vector<int>>(v);
                }
                if (cmd.instance_id > 0) {
                    selection.push_back(cmd.instance_id);
                }
                try {
                    std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                    if (lout) {
                        lout << "DISPATCH_SET_CONTEXT key=" << ctx_key
                             << " value=[";
                        for (size_t i = 0; i < selection.size(); ++i) {
                            if (i) lout << ",";
                            lout << selection[i];
                        }
                        lout << "]\n";
                        lout.close();
                    }
                } catch (...) {}
                std::cerr << "DISPATCH_SET_CONTEXT key=" << ctx_key
                          << " value=[";
                for (size_t i = 0; i < selection.size(); ++i) {
                    if (i) std::cerr << ",";
                    std::cerr << selection[i];
                }
                std::cerr << "]\n";
                pipeline.set_context_var(ctx_key, selection);
                pipeline.execution_paused = false;
                state.waiting_for_user_input = false;
                break;
            }
            case core::CommandType::SHIELD_TRIGGER:
            {
                // Execute Shield Trigger (Play Card Free)
                Instruction play_inst(InstructionOp::PLAY);
                play_inst.args["card"] = cmd.instance_id;

                auto block = std::make_shared<std::vector<Instruction>>();
                block->push_back(play_inst);
                pipeline.call_stack.push_back({block, 0, LoopContext{}});

                // Cleanup pending effect if needed
                if (cmd.amount >= 0 && cmd.amount < (int)state.pending_effects.size()) {
                    if (state.pending_effects[cmd.amount].type == EffectType::SHIELD_TRIGGER) {
                        state.pending_effects.erase(state.pending_effects.begin() + cmd.amount);
                    }
                }
                break;
            }
            default: break;
        }
    }


    void GameLogicSystem::resolve_command_oneshot(core::GameState& state, const core::CommandDef& cmd, const std::map<core::CardID, core::CardDefinition>& card_db) {
        std::cerr << "\n\n### GAME_LOGIC_SYSTEM::resolve_command_oneshot CALLED ###" << std::endl;
        std::cerr << "### Command type: " << static_cast<int>(cmd.type) << std::endl;
        
        PipelineExecutor pipeline;
        dispatch_command(pipeline, state, cmd, card_db);
        pipeline.execute(nullptr, state, card_db);
    }

    void GameLogicSystem::resolve_play_from_stack(core::GameState& game_state, int stack_instance_id, int cost_reduction, core::SpawnSource spawn_source, core::PlayerID /*controller*/, const std::map<core::CardID, core::CardDefinition>& card_db, int evo_source_id, core::ZoneDestination dest_override) {
        // Resolve a play that is currently on the stack: invoke resolution logic
        // by constructing a PipelineExecutor, delegating to handle_resolve_play and
        // running the pipeline so compiled effects (and final MOVE to GRAVE) execute.
        (void)cost_reduction; (void)spawn_source; (void)evo_source_id; (void)dest_override;
        // NOTE: 再発防止 — controller 引数は将来の使用に備えて残す。C4100 を抑制するため /*controller*/ とする。

        // Create pipeline and push resolve-play block
        dm::engine::systems::PipelineExecutor pipeline;
        // Construct an instruction that represents resolving the play
        nlohmann::json args;
        args["card"] = stack_instance_id;
        Instruction inst(InstructionOp::GAME_ACTION, args);
        inst.args["type"] = "RESOLVE_PLAY";

        // Let handle_resolve_play populate pipeline.call_stack
        PlaySystem::instance().handle_resolve_play(pipeline, game_state, inst, card_db);

        // Execute the pipeline to process effects and final move to graveyard
        pipeline.execute(nullptr, game_state, card_db);
    }

    void GameLogicSystem::handle_play_card(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                           const std::map<core::CardID, core::CardDefinition>& card_db) {
        PlaySystem::instance().handle_play_card(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_apply_buffer_move(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                  const std::map<core::CardID, core::CardDefinition>& card_db) {
        // NOTE: 再発防止 — card_db は将来のトリガー効果参照用に保持するが現時点では未使用。
        // C4100 を抑制するために (void) キャストを追加。
        (void)card_db;
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
                // Removed manual pc++ - pipeline_executor handles this automatically
            }
        }
    }

    void GameLogicSystem::handle_resolve_play(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        PlaySystem::instance().handle_resolve_play(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_resolve_effect(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                const std::map<core::CardID, core::CardDefinition>& card_db) {
        try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"handle_resolve_effect entry args="<<inst.args.dump()<<"\n"; d.flush(); d.close(); } } catch(...) {}
        
        // Get effect index from instruction arguments (slot_index)
        int effect_idx = exec.resolve_int(inst.args.value("slot_index", 0));
        
        std::cerr << "\n=== handle_resolve_effect called ===" << std::endl;
        std::cerr << "Effect index: " << effect_idx << std::endl;
        std::cerr << "Pending effects count: " << state.pending_effects.size() << std::endl;
        
        // Validate effect index
        if (effect_idx < 0 || effect_idx >= (int)state.pending_effects.size()) {
            std::cerr << "ERROR: Invalid effect index: " << effect_idx << std::endl;
            return;
        }
        
        // Get the pending effect (copy before erasing, so we can erase safely)
        auto pending_effect = state.pending_effects[effect_idx];
        
        std::cerr << "Source instance: " << pending_effect.source_instance_id << std::endl;
        std::cerr << "Has effect_def: " << (pending_effect.effect_def.has_value() ? "YES" : "NO") << std::endl;
        
        // 再発防止: エフェクトを pending_effects から先に削除してから命令をコンパイル・推送する。
        //   これにより WAIT_INPUT でパイプラインが一時停止しても二重解決が起きない。
        state.pending_effects.erase(state.pending_effects.begin() + effect_idx);
        std::cerr << "Removed effect from pending list, new count: " << state.pending_effects.size() << std::endl;
        
        // 再発防止: 各コマンドを GAME_ACTION(EXECUTE_COMMAND) としてフレームに積む
        //   (Lazy Compile パターン)。
        //   全コマンドを事前コンパイルすると、QUERY の出力変数 (var_DRAW_CARD_0 等) が
        //   まだ未確定の状態で DRAW_CARD の count が計算され count=0 になる。
        //   代わりに、各コマンドを実行時にコンパイルすることで、直前の命令の
        //   exec.context 値 (GET_STAT で設定された var_DRAW_CARD_0 等) を参照できる。
        if (pending_effect.effect_def.has_value()) {
            const auto& effect = pending_effect.effect_def.value();
            
            // 条件チェック
            if (!dm::engine::effects::EffectSystem::instance().check_condition(
                    state, effect.condition,
                    pending_effect.source_instance_id, card_db,
                    pending_effect.execution_context)) {
                std::cerr << "=== condition check failed, skipping ===" << std::endl;
                return;
            }
            
            core::PlayerID controller =
                dm::engine::effects::EffectSystem::get_controller(state, pending_effect.source_instance_id);
            
            // 実行コンテキストを共有パイプラインに注入
            for (const auto& kv : pending_effect.execution_context) {
                exec.set_context_var(kv.first, kv.second);
            }
            exec.set_context_var("$controller", (int)controller);
            exec.set_context_var("$source", pending_effect.source_instance_id);
            
            // 各コマンドを EXECUTE_COMMAND ディスパッチャとしてフレームに積む
            // handle_execute_command が実行時コンテキストでコンパイルしてサブフレームをプッシュする
            std::vector<core::Instruction> dispatch_frame;
            for (const auto& cmd : effect.commands) {
                core::Instruction exec_cmd(core::InstructionOp::GAME_ACTION);
                exec_cmd.args["type"] = "EXECUTE_COMMAND";
                // CommandDef を JSON としてシリアライズして命令に埋め込む
                exec_cmd.args["cmd"] = cmd;  // nlohmann ADL to_json による自動シリアライズ
                exec_cmd.args["source"] = pending_effect.source_instance_id;
                exec_cmd.args["controller"] = (int)controller;
                dispatch_frame.push_back(exec_cmd);
            }
            
            if (!dispatch_frame.empty()) {
                auto frame_ptr = std::make_shared<const std::vector<core::Instruction>>(std::move(dispatch_frame));
                exec.call_stack.push_back({frame_ptr, 0, LoopContext{}});
                std::cerr << "[handle_resolve_effect] Pushed " << frame_ptr->size()
                          << " EXECUTE_COMMAND dispatchers onto shared pipeline (lazy compile)" << std::endl;
            }
        }
        
        std::cerr << "=== handle_resolve_effect complete ===" << std::endl << std::endl;
    }

    void GameLogicSystem::handle_attack(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                        const std::map<core::CardID, core::CardDefinition>& card_db) {
        BattleSystem::instance().handle_attack(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_block(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        BattleSystem::instance().handle_block(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_resolve_battle(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                const std::map<core::CardID, core::CardDefinition>& card_db) {
        BattleSystem::instance().handle_resolve_battle(exec, state, inst, card_db);
    }

    int GameLogicSystem::get_creature_power(const core::CardInstance& creature, const core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        return BattleSystem::instance().get_creature_power(creature, game_state, card_db);
    }

    int GameLogicSystem::get_breaker_count(const core::GameState& state, const core::CardInstance& creature, const std::map<core::CardID, core::CardDefinition>& card_db) {
        return ShieldSystem::instance().get_breaker_count(state, creature, card_db);
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        ShieldSystem::instance().handle_break_shield(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_check_s_trigger(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                      const std::map<core::CardID, core::CardDefinition>& card_db) {
        ShieldSystem::instance().check_s_trigger(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_mana_charge(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        PlaySystem::instance().handle_mana_charge(exec, state, inst);
    }

    void GameLogicSystem::handle_resolve_reaction(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                  const std::map<core::CardID, core::CardDefinition>& card_db) {
         // ...
         (void)exec; (void)state; (void)inst; (void)card_db;
    }

    void GameLogicSystem::handle_use_ability(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                             const std::map<core::CardID, core::CardDefinition>& card_db) {
        PlaySystem::instance().handle_use_ability(exec, state, inst, card_db);
    }

    void GameLogicSystem::handle_select_target(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        exec.execution_paused = true;

        if (inst.args.is_null()) return;

        // Allow caller to supply explicit valid_targets, otherwise compute from filter
        std::vector<int> valid_targets;
        if (inst.args.contains("valid_targets") && inst.args["valid_targets"].is_array()) {
            for (const auto& v : inst.args["valid_targets"]) {
                try {
                    valid_targets.push_back(v.get<int>());
                } catch(...) {}
            }
        } else if (inst.args.contains("filter")) {
            core::FilterDef filter = inst.args.value("filter", core::FilterDef{});

            std::vector<core::Zone> zones;
            if (filter.zones.empty()) {
                zones = {core::Zone::BATTLE, core::Zone::HAND, core::Zone::MANA, core::Zone::SHIELD};
            } else {
                for (const auto& z_str : filter.zones) {
                    if (z_str == "BATTLE_ZONE") zones.push_back(core::Zone::BATTLE);
                    else if (z_str == "HAND") zones.push_back(core::Zone::HAND);
                    else if (z_str == "MANA_ZONE") zones.push_back(core::Zone::MANA);
                    else if (z_str == "SHIELD_ZONE") zones.push_back(core::Zone::SHIELD);
                    else if (z_str == "GRAVEYARD") zones.push_back(core::Zone::GRAVEYARD);
                    else if (z_str == "DECK") zones.push_back(core::Zone::DECK);
                    else if (z_str == "EFFECT_BUFFER") zones.push_back(core::Zone::BUFFER);
                }
            }

            PlayerID player_id = state.active_player_id;
            const auto& card_db = dm::engine::infrastructure::CardRegistry::get_all_definitions();

            for (PlayerID pid : {player_id, static_cast<PlayerID>(1 - player_id)}) {
                for (core::Zone z : zones) {
                    std::vector<int> zone_indices;
                    if (z == core::Zone::BUFFER) {
                        for (const auto& c : state.players[pid].effect_buffer) zone_indices.push_back(c.instance_id);
                    } else {
                        zone_indices = state.get_zone(pid, z);
                    }

                    for (int instance_id : zone_indices) {
                        if (instance_id < 0) continue;
                        const auto* card_ptr = state.get_card_instance(instance_id);
                        if (!card_ptr && z == core::Zone::BUFFER) {
                            const auto& buf = state.players[pid].effect_buffer;
                            auto it = std::find_if(buf.begin(), buf.end(), [instance_id](const core::CardInstance& c){ return c.instance_id == instance_id; });
                            if (it != buf.end()) card_ptr = &(*it);
                        }
                        if (!card_ptr) continue;
                        const auto& card = *card_ptr;

                        if (card_db.count(card.card_id)) {
                            const auto& def = card_db.at(card.card_id);
                            if (dm::engine::utils::TargetUtils::is_valid_target(card, def, filter, state, player_id, pid)) {
                                valid_targets.push_back(instance_id);
                            }
                        } else if (card.card_id == 0) {
                            if (dm::engine::utils::TargetUtils::is_valid_target(card, core::CardDefinition(), filter, state, player_id, pid)) {
                                valid_targets.push_back(instance_id);
                            }
                        }
                    }
                }
            }
        }

        int count = exec.resolve_int(inst.args.value("count", 1));

        // If no valid targets, behave like pipeline SELECT: set context var to empty and continue
        if (valid_targets.empty()) {
            std::string out_key = inst.args.value("out", "$selection");
            exec.set_context_var(out_key, std::vector<int>{});
            exec.execution_paused = false;
            return;
        }

        // If selection should auto-resolve (count >= available), set context and continue
        if (count <= 0 || count >= (int)valid_targets.size()) {
            std::string out_key = inst.args.value("out", "$selection");
            exec.set_context_var(out_key, valid_targets);
            exec.execution_paused = false;
            return;
        }

        // Otherwise pause execution and create pending query for user input
        exec.waiting_for_key = inst.args.value("out", std::string("$selection"));
        if (exec.waiting_for_key.rfind("$", 0) != 0)
            exec.waiting_for_key = std::string("$") + exec.waiting_for_key;
        state.waiting_for_user_input = true;
        state.pending_query = GameState::QueryContext{
            state.pending_query.query_id + 1,
            "SELECT_TARGET",
            {{"min", count}, {"max", count}},
            valid_targets,
            {}
        };
    }

    void GameLogicSystem::handle_execute_command(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         if (!inst.args.contains("cmd")) return;

         try {
             CommandDef cmd = inst.args["cmd"].get<CommandDef>();

             int source_id = inst.args.value("source", -1);
             int controller_id = inst.args.value("controller", (int)state.active_player_id);

             // 再発防止: ランタイムコンパイル時は int コンテキストを渡して
             //   QUERY結果 (例: var_DRAW_CARD_0) を次命令の count 解決に使えるようにする。
             //   vector<int> は execution_context に載せられないため、
             //   TARGETS 系は generate_instructions 側で target="$key" として
             //   実行時解決させる（count へ潰さない）。
             std::map<std::string, int> temp_ctx;
             for (const auto &kv : exec.context) {
                 if (std::holds_alternative<int>(kv.second)) {
                     std::string key = kv.first;
                     if (!key.empty() && key[0] == '$') key = key.substr(1);
                     temp_ctx[key] = std::get<int>(kv.second);
                 }
             }
             auto compiled = dm::engine::systems::CommandSystem::generate_instructions(
                 state, cmd, source_id, (core::PlayerID)controller_id, temp_ctx);

             // コンパイル結果を新フレームとして共有パイプラインの call_stack に積む
             // (クリエートローカルパイプラインを作らない)
             if (!compiled.empty()) {
                 auto frame_ptr = std::make_shared<const std::vector<core::Instruction>>(std::move(compiled));
                 exec.call_stack.push_back({frame_ptr, 0, LoopContext{}});
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

    void GameLogicSystem::handle_check_spell_cast_triggers(PipelineExecutor& exec, GameState& state, const Instruction& inst, const std::map<core::CardID, core::CardDefinition>& card_db) {
        int card_id = exec.resolve_int(inst.args.value("card", 0));
        
        // Debug logging  
        {
            std::ofstream log("c:\\temp\\spell_trigger_debug.txt", std::ios::app);
            if (log) {
                log << "CHECK_SPELL_CAST_TRIGGERS called, card_id=" << card_id << std::endl;
                const CardInstance* card = state.get_card_instance(card_id);
                if (card) {
                    log << "  Card found: " << card->card_id << std::endl;
                    if (card_db.count(card->card_id)) {
                        const auto& def = card_db.at(card->card_id);
                        log << "  Card def found, type=" << (int)def.type << ", effects=" << def.effects.size() << std::endl;
                        for (size_t i = 0; i < def.effects.size(); ++i) {
                            log << "    Effect " << i << ": trigger=" << (int)def.effects[i].trigger 
                                << ", trigger_scope=" << (int)def.effects[i].trigger_scope << std::endl;
                        }
                    } else {
                        log << "  Card def NOT found" << std::endl;
                    }
                } else {
                    log << "  Card NOT found (null)" << std::endl;
                }
                log << "  Calling TriggerSystem::resolve_trigger..." << std::endl;
                log.flush();
            }
        }
        
        TriggerSystem::instance().resolve_trigger(state, TriggerType::ON_CAST_SPELL, card_id, card_db);
        
        // Debug logging after
        {
            std::ofstream log("c:\\temp\\spell_trigger_debug.txt", std::ios::app);
            if (log) {
                log << "  After resolve_trigger, pending_effects.size()=" << state.pending_effects.size() << std::endl;
                log.flush();
            }
        }
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
