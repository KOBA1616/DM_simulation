#include "pipeline_executor.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/systems/rules/condition_system.hpp"
#include "engine/systems/director/game_logic_system.hpp"
#include <iostream>
#include "engine/diag_win32.h"
#include <cstdio>
#include <algorithm>
#include <random>
#include <set>
#include <filesystem>
#include <fstream>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    void PipelineExecutor::execute(const std::vector<Instruction>& instructions, GameState& state,
                                   const std::map<core::CardID, core::CardDefinition>& card_db) {
        auto shared_inst = std::make_shared<std::vector<Instruction>>(instructions);
        execute(shared_inst, state, card_db);
    }

    void PipelineExecutor::execute(std::shared_ptr<const std::vector<Instruction>> instructions, GameState& state,
                                   const std::map<core::CardID, core::CardDefinition>& card_db) {
        if ((!instructions || instructions->empty()) && call_stack.empty()) return;

        if (instructions && !instructions->empty()) {
            size_t before_size = call_stack.size();
            int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
            int parent_pc = -1;
            if (parent_idx >= 0) parent_pc = call_stack[parent_idx].pc;
            std::string inst_dump = "{}";
            if (instructions && !instructions->empty()) inst_dump = (*instructions)[0].args.dump();
            // Removed noisy stderr diagnostic
            // std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
            try { diag_write_win32(std::string("CALLSTACK PUSH before_size=") + std::to_string(before_size) + " parent_idx=" + std::to_string(parent_idx) + " inst=" + inst_dump); } catch(...) {}
            call_stack.push_back({instructions, 0, LoopContext{}});
            try { diag_write_win32(std::string("CALLSTACK PUSH after_size=") + std::to_string(call_stack.size()) + " inst=" + inst_dump); } catch(...) {}
            size_t after_size = call_stack.size();
            // std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
        }

        execution_paused = false;

        // Append a short trace entry for pipeline execute start
        try {
            std::filesystem::create_directories("logs");
            std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
            if (lout) {
                lout << "PIPELINE_EXECUTE start: turn=" << state.turn_number << " call_stack=" << call_stack.size() << "\n";
                lout.close();
            }
        } catch (...) {}

        // Safety: prevent runaway call stack (re-entrancy / infinite nesting).
        const size_t MAX_CALL_STACK = 200;

        // Track repeated top-instruction signatures to detect runaway loops
        int resolve_repeat_count = 0;
        std::string last_inst_sig;

        while (!call_stack.empty() && !execution_paused) {
            // Use index-based access to avoid invalidated references when call_stack is resized
            int frame_idx = (int)call_stack.size() - 1;
            auto& frame = call_stack[frame_idx];

            // Detect runaway nesting
            if (call_stack.size() > MAX_CALL_STACK) {
                try {
                    std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                    if (lout) {
                        lout << "PIPELINE_EXECUTE DUMP EXCEEDED: turn=" << state.turn_number
                             << " call_stack=" << call_stack.size()
                             << " pc=" << frame.pc << "\n";
                        // Dump top frame instruction if possible
                        if (frame.pc < (int)frame.instructions->size()) {
                            auto inst = (*frame.instructions)[frame.pc];
                            lout << "TOP_INSTRUCTION op=" << static_cast<int>(inst.op) << " args=" << inst.args.dump() << "\n";
                        }
                        lout.close();
                    }
                } catch (...) {}

                // Pause execution to allow external inspection/resume rather than looping forever
                execution_paused = true;
                break;
            }

            if (frame.pc >= (int)frame.instructions->size()) {
                try { diag_write_win32(std::string("CALLSTACK POP before_pc=") + std::to_string(frame.pc) + " stack_sz=" + std::to_string(call_stack.size())); } catch(...) {}
                call_stack.pop_back();
                try { diag_write_win32(std::string("CALLSTACK POP after_stack_sz=") + std::to_string(call_stack.size())); } catch(...) {}
                continue;
            }

            const auto& inst = (*frame.instructions)[frame.pc];

            int current_stack_size = (int)call_stack.size();
            int parent_pc_before = -1;
            if (current_stack_size > 0) parent_pc_before = call_stack[current_stack_size - 1].pc;

            // Detect repeating top-instruction signatures (consecutive identical instructions)
            try {
                std::string sig = std::to_string((int)inst.op) + ":" + inst.args.dump();
                if (!last_inst_sig.empty() && last_inst_sig == sig) {
                    resolve_repeat_count++;
                } else {
                    last_inst_sig = sig;
                    resolve_repeat_count = 1;
                }

                const int RESOLVE_REPEAT_THRESHOLD = 80;
                if (resolve_repeat_count > RESOLVE_REPEAT_THRESHOLD) {
                    try {
                        std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                        if (lout) {
                            lout << "PIPELINE_EXECUTE REPEAT_DETECTED: turn=" << state.turn_number
                                 << " repeats=" << resolve_repeat_count << " call_stack=" << call_stack.size() << "\n";
                            lout << "TOP_INST_SIG=" << sig << "\n";
                            // Dump a few top frames
                            for (int i = 0; i < 6 && i < (int)call_stack.size(); ++i) {
                                int idx = (int)call_stack.size() - 1 - i;
                                auto &f = call_stack[idx];
                                if (f.pc < (int)f.instructions->size()) {
                                    auto ii = (*f.instructions)[f.pc];
                                    lout << "FRAME_" << i << " op=" << (int)ii.op << " args=" << ii.args.dump() << " pc=" << f.pc << "\n";
                                }
                            }
                            lout.close();
                        }
                    } catch(...) {}
                    execution_paused = true;
                    break;
                }
            } catch(...) { resolve_repeat_count = 0; last_inst_sig.clear(); }

            // Record Execution History (Debug Hook)
            nlohmann::json log_entry;
            log_entry["op"] = static_cast<int>(inst.op);
            log_entry["args"] = inst.args;
            log_entry["pc"] = frame.pc;
            log_entry["stack_depth"] = call_stack.size();
            execution_history.push_back(log_entry);

            // Mark pre-execution in persistent diag (flush to file immediately)
            try {
                std::filesystem::create_directories("logs");
                std::ofstream diag("logs/crash_diag.txt", std::ios::app);
                if (diag) {
                    std::string args_dump = "";
                    try { args_dump = inst.args.dump(); } catch(...) { args_dump = "<dump_error>"; }
                    int parent_pc = -1;
                    if (call_stack.size() >= 2) parent_pc = call_stack[call_stack.size()-2].pc;
                    diag << "PRE_EXEC pc=" << frame.pc << " parent_pc=" << parent_pc
                         << " op=" << static_cast<int>(inst.op)
                         << " stack=" << call_stack.size()
                         << " args=" << args_dump << "\n";
                    diag.flush();
                    diag.close();
                }
            } catch(...) {}
            // Also emit low-level write for robustness during teardown
            try { diag_write_win32(std::string("PRE_EXEC pc=") + std::to_string(frame.pc) + " op=" + std::to_string((int)inst.op) + " stack=" + std::to_string(call_stack.size())); } catch(...) {}

            execute_instruction(inst, state, card_db);

            // low-level post-exec marker
            try { diag_write_win32(std::string("POST_EXEC pc=") + std::to_string(frame.pc) + " op=" + std::to_string((int)inst.op) + " stack=" + std::to_string(call_stack.size())); } catch(...) {}

            // Mark post-execution and include recent command_history snapshot
            try {
                std::filesystem::create_directories("logs");
                std::ofstream diag("logs/crash_diag.txt", std::ios::app);
                if (diag) {
                    diag << "POST_EXEC pc=" << frame.pc << " op=" << static_cast<int>(inst.op)
                         << " stack=" << call_stack.size();
                    try {
                        size_t chsz = state.command_history.size();
                        diag << " history_sz=" << chsz;
                        if (chsz > 0) {
                            diag << " recent_cmds=";
                            size_t start = (chsz > 5) ? chsz - 5 : 0;
                            for (size_t i = start; i < chsz; ++i) {
                                try {
                                    diag << static_cast<int>(state.command_history[i]->get_type());
                                } catch(...) { diag << "?"; }
                                if (i + 1 < chsz) diag << ",";
                            }
                        }
                    } catch(...) { diag << " history_err"; }
                    diag << "\n";
                    diag.flush();
                    diag.close();
                }
            } catch(...) {}

            // If new frames were pushed, log parent pc change for diagnostics
            if ((int)call_stack.size() > current_stack_size) {
                try {
                    std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                    if (lout) {
                        int parent_idx = current_stack_size - 1;
                        int parent_pc_after = -1;
                        if (parent_idx >= 0 && parent_idx < (int)call_stack.size()) parent_pc_after = call_stack[parent_idx].pc;
                        lout << "PIPELINE_EXECUTE PUSHED: turn=" << state.turn_number << " parent_idx=" << parent_idx
                             << " before_pc=" << parent_pc_before << " after_pc=" << parent_pc_after
                             << " new_call_stack=" << call_stack.size() << "\n";
                        lout.close();
                    }
                } catch(...) {}
            }

            if (execution_paused) break;

            if ((int)call_stack.size() > current_stack_size) {
                 if (inst.op != InstructionOp::IF &&
                     inst.op != InstructionOp::LOOP &&
                     inst.op != InstructionOp::REPEAT) {
                     if (current_stack_size > 0 && current_stack_size <= (int)call_stack.size()) {
                         // parent frame index is current_stack_size - 1
                         call_stack[current_stack_size - 1].pc++;
                     }
                 }
            } else if ((int)call_stack.size() < current_stack_size) {
                 // Returned.
            } else {
                 if (inst.op != InstructionOp::IF &&
                     inst.op != InstructionOp::LOOP &&
                     inst.op != InstructionOp::REPEAT) {
                     // frame_idx still valid unless call_stack changed size, but we are in the branch
                     // where size didn't change, so it's safe to use frame_idx
                     if (frame_idx < (int)call_stack.size()) call_stack[frame_idx].pc++;
                 }
            }
        }
    }

    void PipelineExecutor::resume(GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db,
                                  const ContextValue& input_value) {
        if (!waiting_for_key.empty()) {
            set_context_var(waiting_for_key, input_value);
            waiting_for_key.clear();
        }
        state.waiting_for_user_input = false;

        execute(std::shared_ptr<const std::vector<Instruction>>(), state, card_db);
    }

    void PipelineExecutor::set_context_var(const std::string& key, ContextValue value) {
        context[key] = value;
    }

    ContextValue PipelineExecutor::get_context_var(const std::string& key) const {
        auto it = context.find(key);
        if (it != context.end()) return it->second;
        return std::monostate{}; // Default
    }

    void PipelineExecutor::clear_context() {
        context.clear();
        execution_history.clear();
    }

    nlohmann::json PipelineExecutor::get_execution_history() const {
        return execution_history;
    }

    nlohmann::json PipelineExecutor::dump_context() const {
        nlohmann::json j_ctx = nlohmann::json::object();
        for (const auto& kv : context) {
            if (std::holds_alternative<int>(kv.second)) {
                j_ctx[kv.first] = std::get<int>(kv.second);
            } else if (std::holds_alternative<std::string>(kv.second)) {
                j_ctx[kv.first] = std::get<std::string>(kv.second);
            } else if (std::holds_alternative<std::vector<int>>(kv.second)) {
                j_ctx[kv.first] = std::get<std::vector<int>>(kv.second);
            } else {
                j_ctx[kv.first] = nullptr;
            }
        }
        return j_ctx;
    }

    nlohmann::json PipelineExecutor::dump_call_stack() const {
        nlohmann::json j_stack = nlohmann::json::array();
        for (const auto& frame : call_stack) {
            nlohmann::json f;
            f["pc"] = frame.pc;
            f["total_instructions"] = frame.instructions ? frame.instructions->size() : 0;
            // Maybe dump current instruction if within bounds
            if (frame.instructions && frame.pc < (int)frame.instructions->size()) {
                f["current_op"] = static_cast<int>((*frame.instructions)[frame.pc].op);
            }
            j_stack.push_back(f);
        }
        return j_stack;
    }

    void PipelineExecutor::execute_instruction(const Instruction& inst, GameState& state,
                                               const std::map<core::CardID, core::CardDefinition>& card_db) {
        // Lightweight diagnostic: record entry to instruction executor
        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                int top_pc = -1;
                if (!call_stack.empty()) top_pc = call_stack.back().pc;
                diag << "EXEC_INSTR op=" << static_cast<int>(inst.op)
                     << " top_pc=" << top_pc
                     << " call_stack_size=" << call_stack.size() << "\n";
                diag.close();
            }
        } catch(...) {}
        switch (inst.op) {
            // NOTE: Temporarily disable direct PLAY/ATTACK/BLOCK handlers
            // to bisect a native heap-corruption observed during pytest runs.
            // These were added to delegate legacy ops directly to GameLogicSystem,
            // but disabling them allows us to test whether that change caused
            // the crash. Re-enable after investigation if desired.
#if 0
            case InstructionOp::PLAY: {
                // Legacy/shortcut PLAY instruction should delegate to play handler
                GameLogicSystem::handle_play_card(*this, state, inst, card_db);
                break;
            }
            case InstructionOp::ATTACK: {
                GameLogicSystem::handle_attack(*this, state, inst, card_db);
                break;
            }
            case InstructionOp::BLOCK: {
                GameLogicSystem::handle_block(*this, state, inst, card_db);
                break;
            }
#endif
            case InstructionOp::GAME_ACTION: {
                if (inst.args.is_null()) return;
                std::string type = resolve_string(inst.args.value("type", ""));
                if (type == "PLAY_CARD") {
                    GameLogicSystem::handle_play_card(*this, state, inst, card_db);
                } else if (type == "RESOLVE_PLAY") {
                    GameLogicSystem::handle_resolve_play(*this, state, inst, card_db);
                } else if (type == "ATTACK") {
                    GameLogicSystem::handle_attack(*this, state, inst, card_db);
                } else if (type == "BLOCK") {
                    GameLogicSystem::handle_block(*this, state, inst, card_db);
                } else if (type == "RESOLVE_BATTLE") {
                    GameLogicSystem::handle_resolve_battle(*this, state, inst, card_db);
                } else if (type == "BREAK_SHIELD") {
                    GameLogicSystem::handle_break_shield(*this, state, inst, card_db);
                } else if (type == "CHECK_S_TRIGGER") {
                    GameLogicSystem::handle_check_s_trigger(*this, state, inst, card_db);
                } else if (type == "APPLY_BUFFER_MOVE") {
                    GameLogicSystem::handle_apply_buffer_move(*this, state, inst, card_db);
                } else if (type == "MANA_CHARGE") {
                    GameLogicSystem::handle_mana_charge(*this, state, inst);
                } else if (type == "RESOLVE_REACTION") {
                    GameLogicSystem::handle_resolve_reaction(*this, state, inst, card_db);
                } else if (type == "USE_ABILITY") {
                    GameLogicSystem::handle_use_ability(*this, state, inst, card_db);
                } else if (type == "SELECT_TARGET") {
                    GameLogicSystem::handle_select_target(*this, state, inst);
                } else if (type == "EXECUTE_COMMAND") {
                    GameLogicSystem::handle_execute_command(*this, state, inst);
                } else if (type == "PLAY_CARD_INTERNAL") {
                    GameLogicSystem::handle_play_card(*this, state, inst, card_db);
                } else if (type == "CHECK_CREATURE_ENTER_TRIGGERS") {
                    GameLogicSystem::handle_check_creature_enter_triggers(*this, state, inst, card_db);
                } else if (type == "CHECK_SPELL_CAST_TRIGGERS") {
                    GameLogicSystem::handle_check_spell_cast_triggers(*this, state, inst, card_db);
                }
                break;
            }
            case InstructionOp::SELECT: handle_select(inst, state, card_db); break;
            case InstructionOp::GET_STAT: handle_get_stat(inst, state, card_db); break;
            case InstructionOp::MOVE:   handle_move(inst, state); break;
            case InstructionOp::MODIFY: handle_modify(inst, state); break;
            case InstructionOp::IF:     handle_if(inst, state, card_db); break;
            case InstructionOp::LOOP:   handle_loop(inst, state, card_db); break;
            case InstructionOp::REPEAT: handle_loop(inst, state, card_db); break;
            case InstructionOp::COUNT:
            case InstructionOp::MATH:   handle_calc(inst, state); break;
            case InstructionOp::PRINT:  handle_print(inst, state); break;
            case InstructionOp::WAIT_INPUT: handle_wait_input(inst, state); break;
            default: break;
        }
    }

    int PipelineExecutor::resolve_int(const nlohmann::json& val) const {
        if (val.is_number()) return val.get<int>();
        if (val.is_string()) {
            std::string s = val.get<std::string>();
            if (s.rfind("$", 0) == 0) {
                auto v = get_context_var(s);
                if (std::holds_alternative<int>(v)) return std::get<int>(v);
            }
        }
        return 0;
    }

    std::string PipelineExecutor::resolve_string(const nlohmann::json& val) const {
        if (val.is_string()) {
            std::string s = val.get<std::string>();
            if (s.rfind("$", 0) == 0) {
                auto v = get_context_var(s);
                if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
            }
            return s;
        }
        return "";
    }

    void PipelineExecutor::execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd, core::GameState& state) {
        state.execute_command(std::move(cmd));
    }

    void PipelineExecutor::handle_select(const Instruction& inst, GameState& state,
                                         const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$selection");

        if (context.count(out_key)) return;

        FilterDef filter = inst.args.value("filter", FilterDef{});
        std::vector<int> valid_targets;

        std::vector<Zone> zones;
        if (filter.zones.empty()) {
            zones = {Zone::BATTLE, Zone::HAND, Zone::MANA, Zone::SHIELD};
        } else {
            for (const auto& z_str : filter.zones) {
                 if (z_str == "BATTLE_ZONE") zones.push_back(Zone::BATTLE);
                 else if (z_str == "HAND") zones.push_back(Zone::HAND);
                 else if (z_str == "MANA_ZONE") zones.push_back(Zone::MANA);
                 else if (z_str == "SHIELD_ZONE") zones.push_back(Zone::SHIELD);
                 else if (z_str == "GRAVEYARD") zones.push_back(Zone::GRAVEYARD);
                 else if (z_str == "DECK") zones.push_back(Zone::DECK);
                 else if (z_str == "EFFECT_BUFFER") zones.push_back(Zone::BUFFER);
            }
        }

        PlayerID player_id = state.active_player_id;

        for (PlayerID pid : {player_id, static_cast<PlayerID>(1 - player_id)}) {
            for (Zone z : zones) {
                std::vector<int> zone_indices;
                if (z == Zone::BUFFER) {
                     for(const auto& c : state.players[pid].effect_buffer) zone_indices.push_back(c.instance_id);
                } else {
                     zone_indices = state.get_zone(pid, z);
                }

                for (int instance_id : zone_indices) {
                    if (instance_id < 0) continue;
                    const auto* card_ptr = state.get_card_instance(instance_id);
                    if (!card_ptr && z == Zone::BUFFER) {
                        const auto& buf = state.players[pid].effect_buffer;
                        auto it = std::find_if(buf.begin(), buf.end(), [instance_id](const CardInstance& c){ return c.instance_id == instance_id; });
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
                         if (dm::engine::utils::TargetUtils::is_valid_target(card, CardDefinition(), filter, state, player_id, pid)) {
                            valid_targets.push_back(instance_id);
                        }
                    }
                }
            }
        }

        auto count_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
        int count = resolve_int(count_val);
        if (valid_targets.empty()) {
             set_context_var(out_key, std::vector<int>{});
             return;
        }

        if (count <= 0 || count >= (int)valid_targets.size()) {
             set_context_var(out_key, valid_targets);
             return;
        }

        execution_paused = true;
        waiting_for_key = out_key;
        state.waiting_for_user_input = true;
        state.pending_query = GameState::QueryContext{
            0, "SELECT_TARGET", {{"min", count}, {"max", count}}, valid_targets, {}
        };
    }

    void PipelineExecutor::handle_get_stat(const Instruction& inst, GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;

        std::string stat_name = resolve_string(inst.args.value("stat", ""));
        std::string out_key = inst.args.value("out", "$stat_result");

        PlayerID controller_id = state.active_player_id;
        auto v = get_context_var("$controller");
        if (std::holds_alternative<int>(v)) {
            controller_id = std::get<int>(v);
        }

        const Player& controller = state.players[controller_id];
        int result = 0;

        if (stat_name == "MANA_CIVILIZATION_COUNT") {
            std::set<std::string> civs;
            for (const auto& c : controller.mana_zone) {
                if (card_db.count(c.card_id)) {
                        const auto& cd = card_db.at(c.card_id);
                        for (const auto& civ : cd.civilizations) {
                            if (civ == Civilization::LIGHT) civs.insert("LIGHT");
                            if (civ == Civilization::WATER) civs.insert("WATER");
                            if (civ == Civilization::DARKNESS) civs.insert("DARKNESS");
                            if (civ == Civilization::FIRE) civs.insert("FIRE");
                            if (civ == Civilization::NATURE) civs.insert("NATURE");
                            if (civ == Civilization::ZERO) civs.insert("ZERO");
                        }
                }
            }
            result = (int)civs.size();
        } else if (stat_name == "SHIELD_COUNT") {
            result = (int)controller.shield_zone.size();
        } else if (stat_name == "HAND_COUNT") {
            result = (int)controller.hand.size();
        } else if (stat_name == "CARDS_DRAWN_THIS_TURN") {
            result = state.turn_stats.cards_drawn_this_turn;
        } else if (stat_name == "MANA_COUNT") {
            result = (int)controller.mana_zone.size();
        } else if (stat_name == "BATTLE_ZONE_COUNT") {
            result = (int)controller.battle_zone.size();
        } else if (stat_name == "GRAVEYARD_COUNT") {
            result = (int)controller.graveyard.size();
        }

        set_context_var(out_key, result);
    }

    void PipelineExecutor::handle_move(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) return;

        std::vector<int> targets;
        bool is_virtual_target = false;
        std::string virtual_target_type = "";
        int virtual_count = 0;

        if (inst.args.contains("target")) {
            auto target_val = inst.args["target"];
            if (target_val.is_string()) {
                std::string s = target_val.get<std::string>();
                if (s.rfind("$", 0) == 0) {
                    auto v = get_context_var(s);
                    if (std::holds_alternative<std::vector<int>>(v)) {
                        targets = std::get<std::vector<int>>(v);
                    } else if (std::holds_alternative<int>(v)) {
                        targets.push_back(std::get<int>(v));
                    }
                } else if (s == "DECK_TOP") {
                    is_virtual_target = true;
                    virtual_target_type = "DECK_TOP";
                    auto c_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
                    virtual_count = resolve_int(c_val);
                } else if (s == "DECK_BOTTOM") {
                     is_virtual_target = true;
                     virtual_target_type = "DECK_BOTTOM";
                     auto c_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
                     virtual_count = resolve_int(c_val);
                }
            } else if (target_val.is_number()) {
                targets.push_back(target_val.get<int>());
            }
        }

        // Support input_value_key as an alternative way to pass targets from previous Query/Select
        if (targets.empty() && inst.args.contains("input_value_key")) {
            auto iv = inst.args["input_value_key"];
            if (iv.is_string()) {
                std::string key = iv.get<std::string>();
                if (key.empty() == false) {
                    // Context keys are stored with a leading '$' in the pipeline
                    std::string ctx_key = key.rfind("$", 0) == 0 ? key : ("$" + key);
                    auto v = get_context_var(ctx_key);
                    if (std::holds_alternative<std::vector<int>>(v)) targets = std::get<std::vector<int>>(v);
                    else if (std::holds_alternative<int>(v)) targets.push_back(std::get<int>(v));
                }
            }
        }

        std::string to_zone_str = resolve_string(inst.args.value("to", ""));
        Zone to_zone = Zone::GRAVEYARD;
        if (to_zone_str == "HAND") to_zone = Zone::HAND;
        else if (to_zone_str == "MANA") to_zone = Zone::MANA;
        else if (to_zone_str == "BATTLE") to_zone = Zone::BATTLE;
        else if (to_zone_str == "SHIELD") to_zone = Zone::SHIELD;
        else if (to_zone_str == "DECK") to_zone = Zone::DECK;
        else if (to_zone_str == "BUFFER") to_zone = Zone::BUFFER;
        else if (to_zone_str == "STACK") to_zone = Zone::STACK;

        bool to_bottom = false;
        if (inst.args.contains("to_bottom") && inst.args["to_bottom"].is_boolean()) {
            to_bottom = inst.args["to_bottom"];
        }

        if (is_virtual_target) {
            PlayerID pid = state.active_player_id;
            const auto& deck = state.players[pid].deck;

            if (virtual_target_type == "DECK_TOP") {
                int available = (int)deck.size();
                int count = std::min(virtual_count, available);
                for (int i = 0; i < count; ++i) {
                     if (available - 1 - i >= 0) {
                         targets.push_back(deck[available - 1 - i].instance_id);
                     }
                }
            } else if (virtual_target_type == "DECK_BOTTOM") {
                 int available = (int)deck.size();
                 int count = std::min(virtual_count, available);
                 for (int i = 0; i < count; ++i) {
                     targets.push_back(deck[i].instance_id);
                 }
            }
        }

        for (int id : targets) {
             const CardInstance* card_ptr = state.get_card_instance(id);
             if (!card_ptr) {
                 for (const auto& p : state.players) {
                     for (const auto& c : p.effect_buffer) {
                         if (c.instance_id == id) {
                             card_ptr = &c;
                             break;
                         }
                     }
                     if (card_ptr) break;
                 }
             }

             if (!card_ptr) continue;

             PlayerID owner = card_ptr->owner;
             if (owner > 1) {
                 owner = state.get_card_owner(id);
             } else {
                 owner = state.active_player_id;
             }

             Zone from_zone = Zone::GRAVEYARD;
             bool found = false;
             const Player& p = state.players[owner];

             for(const auto& c : p.hand) if(c.instance_id == id) { from_zone = Zone::HAND; found = true; break; }
             if(!found) for(const auto& c : p.battle_zone) if(c.instance_id == id) { from_zone = Zone::BATTLE; found = true; break; }
             if(!found) for(const auto& c : p.mana_zone) if(c.instance_id == id) { from_zone = Zone::MANA; found = true; break; }
             if(!found) for(const auto& c : p.shield_zone) if(c.instance_id == id) { from_zone = Zone::SHIELD; found = true; break; }
             if(!found) for(const auto& c : p.deck) if(c.instance_id == id) { from_zone = Zone::DECK; found = true; break; }
             if(!found) for(const auto& c : p.graveyard) if(c.instance_id == id) { from_zone = Zone::GRAVEYARD; found = true; break; }
             if(!found) for(const auto& c : p.effect_buffer) if(c.instance_id == id) { from_zone = Zone::BUFFER; found = true; break; }
             if(!found) for(const auto& c : p.stack) if(c.instance_id == id) { from_zone = Zone::STACK; found = true; break; }

             if (!found) continue;

             int dest_idx = to_bottom ? 0 : -1;
             auto cmd = std::make_unique<TransitionCommand>(id, from_zone, to_zone, owner, dest_idx);
             // Temp debug: record attempted transition details
             try {
                 std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
                 if (lout) {
                     lout << "PIPELINE_MOVE id=" << id
                          << " from=" << static_cast<int>(from_zone)
                          << " to=" << static_cast<int>(to_zone)
                          << " owner=" << owner << "\n";
                     lout.close();
                 }
             } catch(...) {}

             execute_command(std::move(cmd), state);
        }
    }

    void PipelineExecutor::handle_modify(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) return;

        std::string mod_type_str = resolve_string(inst.args.value("type", ""));

        if (mod_type_str == "SHUFFLE") {
             std::string target_zone = resolve_string(inst.args.value("target", ""));
             if (target_zone == "DECK") {
                 PlayerID pid = state.active_player_id;
                 auto cmd = std::make_unique<ShuffleCommand>(pid);
                 execute_command(std::move(cmd), state);
             }
             return;
        }

        std::vector<int> targets;
        if (inst.args.contains("target")) {
            auto target_val = inst.args["target"];
            if (target_val.is_string() && target_val.get<std::string>().rfind("$", 0) == 0) {
                auto v = get_context_var(target_val.get<std::string>());
                if (std::holds_alternative<std::vector<int>>(v)) targets = std::get<std::vector<int>>(v);
                else if (std::holds_alternative<int>(v)) targets.push_back(std::get<int>(v));
            } else if (target_val.is_number()) {
                targets.push_back(target_val.get<int>());
            }
        }

        MutateCommand::MutationType type;
        auto val_json = inst.args.contains("value") ? inst.args["value"] : nlohmann::json(0);
        int val = resolve_int(val_json);
        std::string str_val = resolve_string(inst.args.value("str_value", ""));

        if (mod_type_str == "TAP") type = MutateCommand::MutationType::TAP;
        else if (mod_type_str == "UNTAP") type = MutateCommand::MutationType::UNTAP;
        else if (mod_type_str == "POWER_ADD") type = MutateCommand::MutationType::POWER_MOD;
        else if (mod_type_str == "ADD_KEYWORD") type = MutateCommand::MutationType::ADD_KEYWORD;
        else if (mod_type_str == "REMOVE_KEYWORD") type = MutateCommand::MutationType::REMOVE_KEYWORD;
        else if (mod_type_str == "ADD_PASSIVE") type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
        else if (mod_type_str == "ADD_COST_MODIFIER") type = MutateCommand::MutationType::ADD_COST_MODIFIER;
        else if (mod_type_str == "STAT") {
            StatCommand::StatType s_type;
            std::string stat_name = resolve_string(inst.args.value("stat", ""));
            if (stat_name == "CARDS_DRAWN") s_type = StatCommand::StatType::CARDS_DRAWN;
            else if (stat_name == "CARDS_DISCARDED") s_type = StatCommand::StatType::CARDS_DISCARDED;
            else if (stat_name == "CREATURES_PLAYED") s_type = StatCommand::StatType::CREATURES_PLAYED;
            else if (stat_name == "SPELLS_CAST") s_type = StatCommand::StatType::SPELLS_CAST;
            else return;

            auto cmd = std::make_unique<StatCommand>(s_type, val);
            execute_command(std::move(cmd), state);
            return;
        }
        else return;

        // --- NEW: Intercept specific keywords to create PASSIVE EFFECTS instead ---
        // Also support passing them directly via ADD_PASSIVE to streamline ModifierHandler
        bool is_passive_keyword = (type == MutateCommand::MutationType::ADD_KEYWORD &&
                                  (str_val == "CANNOT_ATTACK" || str_val == "CANNOT_BLOCK" || str_val == "CANNOT_ATTACK_OR_BLOCK"));

        if (is_passive_keyword) {
             // Treat as ADD_PASSIVE flow below but with specific target iteration
        }

        // Handling ADD_PASSIVE for both Filter-based (global/broad) and Target-based (specific)
        if (type == MutateCommand::MutationType::ADD_PASSIVE_EFFECT || is_passive_keyword) {
             std::vector<PassiveType> p_types;
             if (str_val == "LOCK_SPELL") p_types.push_back(PassiveType::CANNOT_USE_SPELLS);
             else if (str_val == "POWER") p_types.push_back(PassiveType::POWER_MODIFIER);
             else if (str_val == "CANNOT_ATTACK") p_types.push_back(PassiveType::CANNOT_ATTACK);
             else if (str_val == "CANNOT_BLOCK") p_types.push_back(PassiveType::CANNOT_BLOCK);
             else if (str_val == "CANNOT_ATTACK_OR_BLOCK") {
                 p_types.push_back(PassiveType::CANNOT_ATTACK);
                 p_types.push_back(PassiveType::CANNOT_BLOCK);
             } else {
                 // Fallback or unknown
             }

             if (!p_types.empty()) {
                 int duration = inst.args.value("duration", 1);
                 int source_id = -1;
                 auto v = get_context_var("$source");
                 if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);

                 // Resolve filter references from execution context
                 FilterDef filter = inst.args.value("filter", FilterDef{});
                 if (filter.cost_ref.has_value()) {
                     const auto& key = filter.cost_ref.value();
                     auto ctx_val = get_context_var(key);
                     if (std::holds_alternative<int>(ctx_val)) {
                         filter.exact_cost = std::get<int>(ctx_val);
                     }
                 }

                 // If targets are provided (e.g. from Select or specific target logic), we apply specifically
                 if (!targets.empty()) {
                     for (int id : targets) {
                         for (auto pt : p_types) {
                             PassiveEffect eff;
                             eff.type = pt;
                             eff.value = val;
                             eff.turns_remaining = duration;
                             eff.controller = state.active_player_id;
                             eff.source_instance_id = source_id;
                             eff.specific_targets = std::vector<int>{id};
                             // Pass condition if present
                             if (inst.args.contains("condition")) {
                                 dm::core::from_json(inst.args["condition"], eff.condition);
                             }

                             auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                             cmd->passive_effect = eff;
                             execute_command(std::move(cmd), state);
                         }
                     }
                 } else {
                     // Filter based (Global)
                     for (auto pt : p_types) {
                         PassiveEffect eff;
                         eff.type = pt;
                         eff.target_filter = filter;
                         eff.value = val;
                         eff.turns_remaining = duration;
                         eff.controller = state.active_player_id;
                         eff.source_instance_id = source_id;
                         if (inst.args.contains("condition")) {
                             dm::core::from_json(inst.args["condition"], eff.condition);
                         }

                         auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                         cmd->passive_effect = eff;
                         execute_command(std::move(cmd), state);
                     }
                 }
                 return;
             }
             // If type matches but str_val not handled, fall through?
             // Or if it was supposed to be simple ADD_PASSIVE without str_val mapping logic (unlikely in current design).
        }

        if (type == MutateCommand::MutationType::ADD_PASSIVE_EFFECT) {
             // Fallback for cases not handled above (e.g. direct integer type set?)
             // Currently we rely on str_val. If str_val is empty or unknown, we might have an issue.
             // But existing code only handled LOCK_SPELL and POWER.
             // Let's keep the old block logic just in case but integrated.
             // The above block covers LOCK_SPELL and POWER.
             // So we effectively replaced the logic.
             return;
        }

        if (type == MutateCommand::MutationType::ADD_COST_MODIFIER) {
             CostModifier mod;
             mod.reduction_amount = val;
             mod.condition_filter = inst.args.value("filter", FilterDef{});
             mod.turns_remaining = inst.args.value("duration", 1);
             int source_id = -1;
             auto v = get_context_var("$source");
             if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
             mod.source_instance_id = source_id;
             mod.controller = state.active_player_id;

             auto cmd = std::make_unique<MutateCommand>(-1, type);
             cmd->cost_modifier = mod;
             execute_command(std::move(cmd), state);
             return;
        }

        if (type == MutateCommand::MutationType::ADD_COST_MODIFIER) {
             CostModifier mod;
             mod.reduction_amount = val;
             mod.condition_filter = inst.args.value("filter", FilterDef{});
             mod.turns_remaining = inst.args.value("duration", 1);
             int source_id = -1;
             auto v = get_context_var("$source");
             if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
             mod.source_instance_id = source_id;
             mod.controller = state.active_player_id;

             auto cmd = std::make_unique<MutateCommand>(-1, type);
             cmd->cost_modifier = mod;
             execute_command(std::move(cmd), state);
             return;
        }

        for (int id : targets) {
            auto cmd = std::make_unique<MutateCommand>(id, type, val, str_val);
            execute_command(std::move(cmd), state);
        }
    }

    void PipelineExecutor::handle_if(const Instruction& inst, GameState& state,
                                     const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null() || !inst.args.contains("cond")) {
            call_stack.back().pc++;
            return;
        }

        bool res = check_condition(inst.args["cond"], state, card_db);

        auto block = res ? std::make_shared<std::vector<Instruction>>(inst.then_block)
                         : std::make_shared<std::vector<Instruction>>(inst.else_block);

        call_stack.back().pc++;

        {
            size_t before_size = call_stack.size();
            int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
            int parent_pc = -1;
            if (parent_idx >= 0) parent_pc = call_stack[parent_idx].pc;
            std::string inst_dump = "{}";
            if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
            std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
            call_stack.push_back({block, 0, LoopContext{}});
            size_t after_size = call_stack.size();
            std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
        }
    }

    void PipelineExecutor::handle_loop(const Instruction& inst, GameState& state,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        auto& frame = call_stack.back();
        auto& ctx = frame.loop_ctx;

        if (!ctx.active) {
            ctx.active = true;
            ctx.index = 0;

            if (inst.op == InstructionOp::REPEAT || (inst.args.contains("count") && !inst.args.contains("in"))) {
                auto c_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
                ctx.max = resolve_int(c_val);
                ctx.var_name = inst.args.value("var", "$i");
                ctx.collection.clear();
            } else {
                 ctx.var_name = inst.args.value("as", "$it");
                 if (inst.args.contains("in")) {
                    auto val = inst.args["in"];
                    if (val.is_string() && val.get<std::string>().rfind("$", 0) == 0) {
                        auto v = get_context_var(val.get<std::string>());
                        if (std::holds_alternative<std::vector<int>>(v)) {
                            ctx.collection = std::get<std::vector<int>>(v);
                        }
                    }
                }
                ctx.max = (int)ctx.collection.size();
            }
        }

        if (ctx.index < ctx.max) {
             if (ctx.collection.empty()) {
                 set_context_var(ctx.var_name, ctx.index);
             } else {
                 set_context_var(ctx.var_name, ctx.collection[ctx.index]);
             }

             auto block = std::make_shared<std::vector<Instruction>>(inst.then_block);

             ctx.index++;

             {
                 size_t before_size = call_stack.size();
                 int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
                 int parent_pc = -1;
                 if (parent_idx >= 0) parent_pc = call_stack[parent_idx].pc;
                 std::string inst_dump = "{}";
                 if (block && !block->empty()) inst_dump = (*block)[0].args.dump();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx, parent_pc, inst_dump.c_str());
                 call_stack.push_back({block, 0, LoopContext{}});
                 size_t after_size = call_stack.size();
                 std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__, __LINE__, after_size);
             }
        } else {
             ctx.active = false;
             frame.pc++;
        }
    }

    void PipelineExecutor::handle_calc(const Instruction& inst, GameState& /*state*/) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$result");
        if (inst.op == InstructionOp::MATH) {
            auto l_val = inst.args.contains("lhs") ? inst.args["lhs"] : nlohmann::json(0);
            auto r_val = inst.args.contains("rhs") ? inst.args["rhs"] : nlohmann::json(0);
            int lhs = resolve_int(l_val);
            int rhs = resolve_int(r_val);
            std::string op = inst.args.value("op", "+");
            int res = 0;
            if (op == "+") res = lhs + rhs;
            else if (op == "-") res = lhs - rhs;
            else if (op == "*") res = lhs * rhs;
            else if (op == "/") res = (rhs != 0) ? lhs / rhs : 0;

            set_context_var(out_key, res);
        }
        else if (inst.op == InstructionOp::COUNT) {
             if (inst.args.contains("in")) {
                auto val = inst.args["in"];
                if (val.is_string() && val.get<std::string>().rfind("$", 0) == 0) {
                    auto v = get_context_var(val.get<std::string>());
                    if (std::holds_alternative<std::vector<int>>(v)) {
                        set_context_var(out_key, (int)std::get<std::vector<int>>(v).size());
                    } else {
                        set_context_var(out_key, 0);
                    }
                }
             }
        }
    }

    void PipelineExecutor::handle_print(const Instruction& inst, GameState& /*state*/) {
        if (inst.args.is_null()) return;
        std::cout << "[Pipeline] " << resolve_string(inst.args.value("msg", "")) << std::endl;
    }

    void PipelineExecutor::handle_wait_input(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) {
            return;
        }

        std::string out_key = inst.args.value("out", "$input");

        // If we already have the value (via resume), don't pause again
        if (context.count(out_key)) {
            return;
        }

        std::string query_type = inst.args.value("query_type", "NONE");
        std::vector<std::string> options;
        if (inst.args.contains("options")) {
            for (const auto& opt : inst.args["options"]) {
                if (opt.is_string()) options.push_back(opt.get<std::string>());
            }
        }

        execution_paused = true;
        waiting_for_key = out_key;
        state.waiting_for_user_input = true;

        // Setup pending query with min/max for SELECT_NUMBER
        nlohmann::json params;
        if (inst.args.contains("min")) {
            int min_val = resolve_int(inst.args["min"]);
            params["min"] = min_val;
        }
        if (inst.args.contains("max")) {
            int max_val = resolve_int(inst.args["max"]);
            params["max"] = max_val;
        }

        state.pending_query = GameState::QueryContext{
            0, query_type, {}, params, options
        };
    }

    bool PipelineExecutor::check_condition(const nlohmann::json& cond, GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (cond.is_null()) return false;

        if (cond.contains("type")) {
             std::string type = cond.value("type", "NONE");
             if (type != "NONE") {
                 core::ConditionDef def;
                 // Use centralized JSON parsing to ensure all fields (including filter) are handled
                 dm::core::from_json(cond, def);

                 int source_id = -1;
                 auto v = get_context_var("$source");
                 if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
                 else if (std::holds_alternative<std::vector<int>>(v)) {
                     const auto& vec = std::get<std::vector<int>>(v);
                     if (!vec.empty()) source_id = vec[0];
                 }

                 std::map<std::string, int> exec_ctx;
                 for (const auto& kv : context) {
                     if (std::holds_alternative<int>(kv.second)) {
                         exec_ctx[kv.first] = std::get<int>(kv.second);
                     }
                 }

                 return dm::engine::rules::ConditionSystem::instance().evaluate_def(state, def, source_id, card_db, exec_ctx);
             }
        }

        if (cond.contains("exists")) {
            std::string key = cond["exists"];
            auto v = get_context_var(key);
            if (std::holds_alternative<std::vector<int>>(v)) {
                return !std::get<std::vector<int>>(v).empty();
            }
            if (std::holds_alternative<int>(v)) return true;
        }

        if (cond.contains("op")) {
            int lhs = 0;
            if (cond.contains("lhs")) lhs = resolve_int(cond["lhs"]);

            int rhs = 0;
            if (cond.contains("rhs")) rhs = resolve_int(cond["rhs"]);

            std::string op = cond.value("op", "==");
            if (op == "==") return lhs == rhs;
            if (op == ">") return lhs > rhs;
            if (op == "<") return lhs < rhs;
            if (op == ">=") return lhs >= rhs;
            if (op == "<=") return lhs <= rhs;
            if (op == "!=") return lhs != rhs;
        }
        return false;
    }

}
