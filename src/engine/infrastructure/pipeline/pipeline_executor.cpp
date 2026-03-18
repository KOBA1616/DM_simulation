#include "pipeline_executor.hpp"
#include "engine/diag_win32.h"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/infrastructure/data/card_registry.hpp"
#include "engine/systems/director/game_logic_system.hpp"
#include "engine/systems/effects/passive_effect_system.hpp"
#include "engine/systems/rules/condition_system.hpp"
#include "engine/utils/target_utils.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

namespace dm::engine::systems {

using namespace core;
using namespace game_command;

void PipelineExecutor::execute(
    const std::vector<Instruction> &instructions, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  auto shared_inst = std::make_shared<std::vector<Instruction>>(instructions);
  execute(shared_inst, state, card_db);
}

void PipelineExecutor::execute(
    std::shared_ptr<const std::vector<Instruction>> instructions,
    GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  if ((!instructions || instructions->empty()) && call_stack.empty())
    return;

  if (instructions && !instructions->empty()) {
    size_t before_size = call_stack.size();
    int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
    int parent_pc = -1;
    if (parent_idx >= 0)
      parent_pc = call_stack[parent_idx].pc;
    std::string inst_dump = "{}";
    if (instructions && !instructions->empty())
      inst_dump = (*instructions)[0].args.dump();
    // Removed noisy stderr diagnostic
    // std::fprintf(stderr, "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d
    // parent_pc=%d inst=%s\n", __FILE__, __LINE__, before_size, parent_idx,
    // parent_pc, inst_dump.c_str());
    try {
      diag_write_win32(std::string("CALLSTACK PUSH before_size=") +
                       std::to_string(before_size) + " parent_idx=" +
                       std::to_string(parent_idx) + " inst=" + inst_dump);
    } catch (...) {
    }
    call_stack.push_back({instructions, 0, LoopContext{}});
    try {
      diag_write_win32(std::string("CALLSTACK PUSH after_size=") +
                       std::to_string(call_stack.size()) +
                       " inst=" + inst_dump);
    } catch (...) {
    }
    size_t after_size = call_stack.size();
    // std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__,
    // __LINE__, after_size);
  }

  execution_paused = false;

  // Append a short trace entry for pipeline execute start
  try {
    std::filesystem::create_directories("logs");
    std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
    if (lout) {
      lout << "PIPELINE_EXECUTE start: turn=" << state.turn_number
           << " call_stack=" << call_stack.size() << "\n";
      lout.close();
    }
  } catch (...) {
  }

  // Safety: prevent runaway call stack (re-entrancy / infinite nesting).
  const size_t MAX_CALL_STACK = 200;

  // Track repeated top-instruction signatures to detect runaway loops
  int resolve_repeat_count = 0;
  std::string last_inst_sig;

  while (!call_stack.empty() && !execution_paused) {
    // Use index-based access to avoid invalidated references when call_stack is
    // resized
    int frame_idx = (int)call_stack.size() - 1;
    auto &frame = call_stack[frame_idx];

    // Detect runaway nesting
    if (call_stack.size() > MAX_CALL_STACK) {
      try {
        std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
        if (lout) {
          lout << "PIPELINE_EXECUTE DUMP EXCEEDED: turn=" << state.turn_number
               << " call_stack=" << call_stack.size() << " pc=" << frame.pc
               << "\n";
          // Dump top frame instruction if possible
          if (frame.pc < (int)frame.instructions->size()) {
            auto inst = (*frame.instructions)[frame.pc];
            lout << "TOP_INSTRUCTION op=" << static_cast<int>(inst.op)
                 << " args=" << inst.args.dump() << "\n";
          }
          lout.close();
        }
      } catch (...) {
      }

      // Pause execution to allow external inspection/resume rather than looping
      // forever
      execution_paused = true;
      break;
    }

    if (frame.pc >= (int)frame.instructions->size()) {
      try {
        diag_write_win32(std::string("CALLSTACK POP before_pc=") +
                         std::to_string(frame.pc) +
                         " stack_sz=" + std::to_string(call_stack.size()));
      } catch (...) {
      }
      call_stack.pop_back();
      try {
        diag_write_win32(std::string("CALLSTACK POP after_stack_sz=") +
                         std::to_string(call_stack.size()));
      } catch (...) {
      }
      continue;
    }

    const auto &inst = (*frame.instructions)[frame.pc];

    int current_stack_size = (int)call_stack.size();
    int parent_pc_before = -1;
    if (current_stack_size > 0)
      parent_pc_before = call_stack[current_stack_size - 1].pc;

    // Detect repeating top-instruction signatures (consecutive identical
    // instructions)
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
            lout << "PIPELINE_EXECUTE REPEAT_DETECTED: turn="
                 << state.turn_number << " repeats=" << resolve_repeat_count
                 << " call_stack=" << call_stack.size() << "\n";
            lout << "TOP_INST_SIG=" << sig << "\n";
            // Dump a few top frames
            for (int i = 0; i < 6 && i < (int)call_stack.size(); ++i) {
              int idx = (int)call_stack.size() - 1 - i;
              auto &f = call_stack[idx];
              if (f.pc < (int)f.instructions->size()) {
                auto ii = (*f.instructions)[f.pc];
                lout << "FRAME_" << i << " op=" << (int)ii.op
                     << " args=" << ii.args.dump() << " pc=" << f.pc << "\n";
              }
            }
            lout.close();
          }
        } catch (...) {
        }
        execution_paused = true;
        break;
      }
    } catch (...) {
      resolve_repeat_count = 0;
      last_inst_sig.clear();
    }

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
        try {
          args_dump = inst.args.dump();
        } catch (...) {
          args_dump = "<dump_error>";
        }
        int parent_pc = -1;
        if (call_stack.size() >= 2)
          parent_pc = call_stack[call_stack.size() - 2].pc;
        diag << "PRE_EXEC pc=" << frame.pc << " parent_pc=" << parent_pc
             << " op=" << static_cast<int>(inst.op)
             << " stack=" << call_stack.size() << " args=" << args_dump << "\n";
        diag.flush();
        diag.close();
      }
    } catch (...) {
    }
    // Also emit low-level write for robustness during teardown
    try {
      diag_write_win32(std::string("PRE_EXEC pc=") + std::to_string(frame.pc) +
                       " op=" + std::to_string((int)inst.op) +
                       " stack=" + std::to_string(call_stack.size()));
    } catch (...) {
    }

    try {
      std::filesystem::create_directories("logs");
      std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
      if (lout) {
        std::string args_dump = "";
        try { args_dump = inst.args.dump(); } catch (...) { args_dump = "<dump_err>"; }
        try {
          lout << "INST_BEFORE pc=" << frame.pc << " op=" << static_cast<int>(inst.op)
               << " args=" << args_dump << " context=" << dump_context().dump() << "\n";
        } catch (...) {
          lout << "INST_BEFORE write_error\n";
        }
        lout.close();
      }
    } catch (...) {}

    execute_instruction(inst, state, card_db);

    try {
      std::filesystem::create_directories("logs");
      std::ofstream lout2("logs/pipeline_trace.txt", std::ios::app);
      if (lout2) {
        std::string args_dump = "";
        try { args_dump = inst.args.dump(); } catch (...) { args_dump = "<dump_err>"; }
        try {
          lout2 << "INST_AFTER pc=" << frame.pc << " op=" << static_cast<int>(inst.op)
                << " args=" << args_dump << " context=" << dump_context().dump() << "\n";
        } catch (...) {
          lout2 << "INST_AFTER write_error\n";
        }
        lout2.close();
      }
    } catch (...) {}

    // low-level post-exec marker
    try {
      diag_write_win32(std::string("POST_EXEC pc=") + std::to_string(frame.pc) +
                       " op=" + std::to_string((int)inst.op) +
                       " stack=" + std::to_string(call_stack.size()));
    } catch (...) {
    }

    // Mark post-execution and include recent command_history snapshot
    try {
      std::filesystem::create_directories("logs");
      std::ofstream diag("logs/crash_diag.txt", std::ios::app);
      if (diag) {
        diag << "POST_EXEC pc=" << frame.pc
             << " op=" << static_cast<int>(inst.op)
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
              } catch (...) {
                diag << "?";
              }
              if (i + 1 < chsz)
                diag << ",";
            }
          }
        } catch (...) {
          diag << " history_err";
        }
        diag << "\n";
        diag.flush();
        diag.close();
      }
    } catch (...) {
    }

    // If new frames were pushed, log parent pc change for diagnostics
    if ((int)call_stack.size() > current_stack_size) {
      try {
        std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
        if (lout) {
          int parent_idx = current_stack_size - 1;
          int parent_pc_after = -1;
          if (parent_idx >= 0 && parent_idx < (int)call_stack.size())
            parent_pc_after = call_stack[parent_idx].pc;
          lout << "PIPELINE_EXECUTE PUSHED: turn=" << state.turn_number
               << " parent_idx=" << parent_idx
               << " before_pc=" << parent_pc_before
               << " after_pc=" << parent_pc_after
               << " new_call_stack=" << call_stack.size() << "\n";
          lout.close();
        }
      } catch (...) {
      }
    }

    if (execution_paused)
      break;

    if ((int)call_stack.size() > current_stack_size) {
      if (inst.op != InstructionOp::IF && inst.op != InstructionOp::LOOP &&
          inst.op != InstructionOp::REPEAT) {
        if (current_stack_size > 0 &&
            current_stack_size <= (int)call_stack.size()) {
          // parent frame index is current_stack_size - 1
          call_stack[current_stack_size - 1].pc++;
        }
      }
    } else if ((int)call_stack.size() < current_stack_size) {
      // Returned.
    } else {
      if (inst.op != InstructionOp::IF && inst.op != InstructionOp::LOOP &&
          inst.op != InstructionOp::REPEAT) {
        // frame_idx still valid unless call_stack changed size, but we are in
        // the branch where size didn't change, so it's safe to use frame_idx
        if (frame_idx < (int)call_stack.size())
          call_stack[frame_idx].pc++;
      }
    }
  }
}

void PipelineExecutor::resume(
    GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db,
    const ContextValue &input_value) {
  if (!waiting_for_key.empty()) {
    set_context_var(waiting_for_key, input_value);
    waiting_for_key.clear();
  }
  state.waiting_for_user_input = false;

  execute(std::shared_ptr<const std::vector<Instruction>>(), state, card_db);
}

void PipelineExecutor::set_context_var(const std::string &key,
                                       ContextValue value) {
  // Store both $-prefixed and plain forms to avoid lookup mismatches.
  context[key] = value;
  if (!key.empty() && key[0] == '$') {
    std::string plain = key.substr(1);
    context[plain] = value;
  } else {
    std::string with_dollar = "$" + key;
    context[with_dollar] = value;
  }
}

ContextValue PipelineExecutor::get_context_var(const std::string &key) const {
  auto it = context.find(key);
  if (it != context.end())
    return it->second;
  // 再発防止: コンテキスト変数の格納キーと参照キーの $ プレフィックス不一致による取得失敗防止。
  //   set_context_var は "var_X" で格納するが、参照側は "$var_X" を使う場合がある。
  //   どちらの形式でも取得できるよう、$ あり/なしの両パターンをフォールバック検索する。
  if (!key.empty() && key[0] == '$') {
    // "$var_X" → "var_X" を試す
    auto it2 = context.find(key.substr(1));
    if (it2 != context.end())
      return it2->second;
  } else {
    // "var_X" → "$var_X" を試す
    auto it2 = context.find("$" + key);
    if (it2 != context.end())
      return it2->second;
  }
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
  for (const auto &kv : context) {
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
  for (const auto &frame : call_stack) {
    nlohmann::json f;
    f["pc"] = frame.pc;
    f["total_instructions"] =
        frame.instructions ? frame.instructions->size() : 0;
    // Maybe dump current instruction if within bounds
    if (frame.instructions && frame.pc < (int)frame.instructions->size()) {
      f["current_op"] = static_cast<int>((*frame.instructions)[frame.pc].op);
    }
    j_stack.push_back(f);
  }
  return j_stack;
}

void PipelineExecutor::execute_instruction(
    const Instruction &inst, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  // Lightweight diagnostic: record entry to instruction executor
  try {
    std::ofstream diag("logs/crash_diag.txt", std::ios::app);
    if (diag) {
      int top_pc = -1;
      if (!call_stack.empty())
        top_pc = call_stack.back().pc;
      diag << "EXEC_INSTR op=" << static_cast<int>(inst.op)
           << " top_pc=" << top_pc << " call_stack_size=" << call_stack.size()
           << "\n";
      diag.close();
    }
  } catch (...) {
  }
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
    if (inst.args.is_null())
      return;
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
      GameLogicSystem::handle_check_creature_enter_triggers(*this, state, inst,
                                                            card_db);
    } else if (type == "CHECK_SPELL_CAST_TRIGGERS") {
      GameLogicSystem::handle_check_spell_cast_triggers(*this, state, inst,
                                                        card_db);
    } else if (type == "AUTO_SELECT_BUFFER") {
      // 再発防止: 暗黙的バッファ選択 — フィルターに一致するバッファカードを
      //   $buffer_select コンテキスト変数に設定する。
      //   MOVE_BUFFER_TO_ZONE(with filter) が事前にこの命令を生成する。
      //   SELECT_FROM_BUFFER によるユーザー入力なしで自動選択できる。
      PlayerID pid = state.active_player_id;
      std::string out_key = inst.args.contains("out")
                                ? inst.args["out"].get<std::string>()
                                : "$buffer_select";

      // フィルター復元: 文明/タイプ/種族を JSON から解析
      std::vector<core::Civilization> filter_civs;
      std::vector<std::string> filter_types;
      std::vector<std::string> filter_races;
      std::optional<int> filter_min_cost, filter_max_cost, filter_exact_cost;

      if (inst.args.contains("filter") && inst.args["filter"].is_object()) {
        const auto &fj = inst.args["filter"];
        if (fj.contains("civilizations") && fj["civilizations"].is_array()) {
          for (const auto &cv : fj["civilizations"]) {
            try { filter_civs.push_back(cv.get<core::Civilization>()); } catch (...) {}
          }
        }
        if (fj.contains("types") && fj["types"].is_array()) {
          for (const auto &tv : fj["types"]) {
            try { filter_types.push_back(tv.get<std::string>()); } catch (...) {}
          }
        }
        if (fj.contains("races") && fj["races"].is_array()) {
          for (const auto &rv : fj["races"]) {
            try { filter_races.push_back(rv.get<std::string>()); } catch (...) {}
          }
        }
        if (fj.contains("min_cost") && fj["min_cost"].is_number())
          filter_min_cost = fj["min_cost"].get<int>();
        if (fj.contains("max_cost") && fj["max_cost"].is_number())
          filter_max_cost = fj["max_cost"].get<int>();
        if (fj.contains("exact_cost") && fj["exact_cost"].is_number())
          filter_exact_cost = fj["exact_cost"].get<int>();
      }

      std::vector<int> selected;
      for (const auto &card : state.players[pid].effect_buffer) {
        auto it = card_db.find(card.card_id);
        const core::CardDefinition *def_ptr = nullptr;
        core::CardDefinition empty_def;
        if (it != card_db.end()) def_ptr = &it->second;
        else def_ptr = &empty_def;

        bool matches = true;

        // 文明フィルター: 1つ以上一致すればOK
        if (matches && !filter_civs.empty()) {
          bool civ_ok = false;
          for (auto fc : filter_civs) {
            for (auto cc : def_ptr->civilizations) {
              if (static_cast<uint8_t>(fc) & static_cast<uint8_t>(cc)) {
                civ_ok = true; break;
              }
            }
            if (civ_ok) break;
          }
          if (!civ_ok) matches = false;
        }

        // タイプフィルター
        if (matches && !filter_types.empty()) {
          std::string card_type_str =
              (def_ptr->type == core::CardType::CREATURE)   ? "CREATURE"
            : (def_ptr->type == core::CardType::SPELL)      ? "SPELL"
            : (def_ptr->type == core::CardType::EVOLUTION_CREATURE) ? "EVOLUTION_CREATURE"
            : "";
          bool type_ok = false;
          for (const auto &t : filter_types) {
            if (card_type_str == t) { type_ok = true; break; }
          }
          if (!type_ok) matches = false;
        }

        // 種族フィルター
        if (matches && !filter_races.empty()) {
          bool race_ok = false;
          for (const auto &fr : filter_races) {
            for (const auto &cr : def_ptr->races) {
              if (cr == fr) { race_ok = true; break; }
            }
            if (race_ok) break;
          }
          if (!race_ok) matches = false;
        }

        // コストフィルター
        if (matches && filter_min_cost.has_value() && def_ptr->cost < *filter_min_cost)
          matches = false;
        if (matches && filter_max_cost.has_value() && def_ptr->cost > *filter_max_cost)
          matches = false;
        if (matches && filter_exact_cost.has_value() && def_ptr->cost != *filter_exact_cost)
          matches = false;

        if (matches) selected.push_back(card.instance_id);
      }
      set_context_var(out_key, selected);
    }
    break;
  }
  case InstructionOp::SELECT:
    handle_select(inst, state, card_db);
    break;
  case InstructionOp::GET_STAT:
    handle_get_stat(inst, state, card_db);
    break;
  case InstructionOp::MOVE:
    handle_move(inst, state);
    break;
  case InstructionOp::MODIFY:
    handle_modify(inst, state);
    break;
  case InstructionOp::IF:
    handle_if(inst, state, card_db);
    break;
  case InstructionOp::LOOP:
    handle_loop(inst, state, card_db);
    break;
  case InstructionOp::REPEAT:
    handle_loop(inst, state, card_db);
    break;
  case InstructionOp::COUNT:
  case InstructionOp::MATH:
    handle_calc(inst, state);
    break;
  case InstructionOp::PRINT:
    handle_print(inst, state);
    break;
  case InstructionOp::WAIT_INPUT:
    handle_wait_input(inst, state);
    break;
  default:
    break;
  }
}

int PipelineExecutor::resolve_int(const nlohmann::json &val) const {
  if (val.is_number())
    return val.get<int>();
  if (val.is_string()) {
    std::string s = val.get<std::string>();
    if (s.rfind("$", 0) == 0) {
      auto v = get_context_var(s);
      if (std::holds_alternative<int>(v))
        return std::get<int>(v);
    }
  }
  return 0;
}

std::string PipelineExecutor::resolve_string(const nlohmann::json &val) const {
  if (val.is_string()) {
    std::string s = val.get<std::string>();
    if (s.rfind("$", 0) == 0) {
      auto v = get_context_var(s);
      if (std::holds_alternative<std::string>(v))
        return std::get<std::string>(v);
    }
    return s;
  }
  return "";
}

void PipelineExecutor::execute_command(
    std::unique_ptr<dm::engine::game_command::GameCommand> cmd,
    core::GameState &state) {
  state.execute_command(std::move(cmd));
}

void PipelineExecutor::handle_select(
    const Instruction &inst, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  if (inst.args.is_null())
    return;
  std::string out_key = inst.args.value("out", "$selection");

  if (context.count(out_key))
    return;

  FilterDef filter = inst.args.value("filter", FilterDef{});
  std::vector<int> valid_targets;

  std::vector<Zone> zones;
  if (filter.zones.empty()) {
    zones = {Zone::BATTLE, Zone::HAND, Zone::MANA, Zone::SHIELD};
  } else {
    for (const auto &z_str : filter.zones) {
      if (z_str == "BATTLE_ZONE")
        zones.push_back(Zone::BATTLE);
      else if (z_str == "HAND")
        zones.push_back(Zone::HAND);
      else if (z_str == "MANA_ZONE")
        zones.push_back(Zone::MANA);
      else if (z_str == "SHIELD_ZONE")
        zones.push_back(Zone::SHIELD);
      else if (z_str == "GRAVEYARD")
        zones.push_back(Zone::GRAVEYARD);
      else if (z_str == "DECK")
        zones.push_back(Zone::DECK);
      else if (z_str == "EFFECT_BUFFER")
        zones.push_back(Zone::BUFFER);
    }
  }

  PlayerID player_id = state.active_player_id;

  std::vector<PlayerID> players_to_check;
  if (!filter.owner.has_value()) {
    // デフォルト: 発動プレイヤーのみを対象とする
    players_to_check.push_back(player_id);
  } else {
    const std::string &own = *filter.owner;
    if (own == "SELF") {
      players_to_check.push_back(player_id);
    } else if (own == "OPPONENT") {
      players_to_check.push_back(static_cast<PlayerID>(1 - player_id));
    } else if (own == "BOTH") {
      players_to_check.push_back(player_id);
      players_to_check.push_back(static_cast<PlayerID>(1 - player_id));
    } else {
      // Unknown owner string: fallback to active player only
      players_to_check.push_back(player_id);
    }
  }

  for (PlayerID pid : players_to_check) {
    for (Zone z : zones) {
      std::vector<int> zone_indices;
      if (z == Zone::BUFFER) {
        for (const auto &c : state.players[pid].effect_buffer)
          zone_indices.push_back(c.instance_id);
      } else {
        zone_indices = state.get_zone(pid, z);
      }

      for (int instance_id : zone_indices) {
        if (instance_id < 0)
          continue;
        const auto *card_ptr = state.get_card_instance(instance_id);
        if (!card_ptr && z == Zone::BUFFER) {
          const auto &buf = state.players[pid].effect_buffer;
          auto it = std::find_if(buf.begin(), buf.end(),
                                 [instance_id](const CardInstance &c) {
                                   return c.instance_id == instance_id;
                                 });
          if (it != buf.end())
            card_ptr = &(*it);
        }

        if (!card_ptr)
          continue;
        const auto &card = *card_ptr;

        if (card_db.count(card.card_id)) {
          const auto &def = card_db.at(card.card_id);
          if (dm::engine::utils::TargetUtils::is_valid_target(
                  card, def, filter, state, player_id, pid)) {
            valid_targets.push_back(instance_id);
          }
        } else if (card.card_id == 0) {
          if (dm::engine::utils::TargetUtils::is_valid_target(
                  card, CardDefinition(), filter, state, player_id, pid)) {
            valid_targets.push_back(instance_id);
          }
        }
      }
    }
  }

  auto count_val =
      inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
  int count = resolve_int(count_val);
  if (valid_targets.empty()) {
    set_context_var(out_key, std::vector<int>{});
    return;
  }

  // 再発防止: count=0 (ドロー0枚を選択した場合など) のとき valid_targets 全選択は誤り。
  //   count=0 のときは空ベクターを設定して次の命令へ進む。
  if (count <= 0) {
    set_context_var(out_key, std::vector<int>{});
    return;
  }

  if (count >= (int)valid_targets.size()) {
    set_context_var(out_key, valid_targets);
    return;
  }

  execution_paused = true;
  waiting_for_key = out_key;
  state.waiting_for_user_input = true;
  state.pending_query = GameState::QueryContext{
      0, "SELECT_TARGET", {{"min", count}, {"max", count}}, valid_targets, {}};
}

void PipelineExecutor::handle_get_stat(
    const Instruction &inst, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  if (inst.args.is_null())
    return;

  std::string stat_name = resolve_string(inst.args.value("stat", ""));
  std::string out_key = inst.args.value("out", "$stat_result");

  PlayerID controller_id = state.active_player_id;
  auto v = get_context_var("$controller");
  if (std::holds_alternative<int>(v)) {
    controller_id = std::get<int>(v);
  }

  const Player &controller = state.players[controller_id];
  int result = 0;

  if (stat_name == "MANA_CIVILIZATION_COUNT") {
    std::set<std::string> civs;
    for (const auto &c : controller.mana_zone) {
      if (card_db.count(c.card_id)) {
        const auto &cd = card_db.at(c.card_id);
        for (const auto &civ : cd.civilizations) {
          if (civ == Civilization::LIGHT)
            civs.insert("LIGHT");
          if (civ == Civilization::WATER)
            civs.insert("WATER");
          if (civ == Civilization::DARKNESS)
            civs.insert("DARKNESS");
          if (civ == Civilization::FIRE)
            civs.insert("FIRE");
          if (civ == Civilization::NATURE)
            civs.insert("NATURE");
          if (civ == Civilization::ZERO)
            civs.insert("ZERO");
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
  } else if (stat_name == "CREATURE_COUNT") {
    // 再発防止: CREATURE_COUNT はバトルゾーンのクリーチャー数（DM では常に battle_zone と同一）
    result = (int)controller.battle_zone.size();
  } else if (stat_name == "GRAVEYARD_COUNT") {
    result = (int)controller.graveyard.size();
  } else if (stat_name.rfind("OPPONENT_", 0) == 0) {
    // 再発防止: OPPONENT_* は相手プレイヤーの統計を返す。
    //   PlayerID は 0/1 の 2 値前提で相手を (1 - controller_id) で求める。
    PlayerID opp_id = (controller_id == PlayerID(0)) ? PlayerID(1) : PlayerID(0);
    const Player &opp = state.players[opp_id];
    std::string opp_stat = stat_name.substr(9); // "OPPONENT_" を除去
    if (opp_stat == "MANA_COUNT") {
      result = (int)opp.mana_zone.size();
    } else if (opp_stat == "CREATURE_COUNT" || opp_stat == "BATTLE_ZONE_COUNT") {
      result = (int)opp.battle_zone.size();
    } else if (opp_stat == "SHIELD_COUNT") {
      result = (int)opp.shield_zone.size();
    } else if (opp_stat == "HAND_COUNT") {
      result = (int)opp.hand.size();
    } else if (opp_stat == "GRAVEYARD_COUNT") {
      result = (int)opp.graveyard.size();
    }
  }

  set_context_var(out_key, result);
}

void PipelineExecutor::handle_move(const Instruction &inst, GameState &state) {
  if (inst.args.is_null())
    return;

  std::vector<int> targets;
  bool is_virtual_target = false;
  std::string virtual_target_type = "";
  int virtual_count = 0;

  if (inst.args.contains("target")) {
    auto target_val = inst.args["target"];
    if (target_val.is_string()) {
      std::string s = target_val.get<std::string>();
      // If caller provided a count that is actually a context variable
      // referring to an explicit list of targets (vector<int>), prefer
      // using that vector directly. Some command-generation paths may
      // incorrectly place the selected targets into the "count" field
      // (as a $var) instead of the "target" field; handle that here to
      // be tolerant and avoid dropping selections.
      if (inst.args.contains("count") && inst.args["count"].is_string()) {
        std::string count_ref = inst.args["count"].get<std::string>();
        if (!count_ref.empty() && count_ref.rfind("$", 0) == 0) {
          auto cv = get_context_var(count_ref);
          if (std::holds_alternative<std::vector<int>>(cv)) {
            targets = std::get<std::vector<int>>(cv);
            // Log resolved fallback for debugging
            try {
              std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
              if (lout) {
                lout << "RESOLVED_COUNT_AS_TARGETS count_ref=" << count_ref << " -> [";
                for (size_t _i = 0; _i < targets.size(); ++_i) {
                  if (_i) lout << ",";
                  lout << targets[_i];
                }
                lout << "]\n";
                lout.close();
              }
            } catch (...) {}
            std::cerr << "RESOLVED_COUNT_AS_TARGETS count_ref=" << count_ref << " -> [";
            for (size_t _i = 0; _i < targets.size(); ++_i) {
              if (_i) std::cerr << ",";
              std::cerr << targets[_i];
            }
            std::cerr << "]\n";
          }
        }
      }
      if (s.rfind("$", 0) == 0) {
        auto v = get_context_var(s);
        if (std::holds_alternative<std::vector<int>>(v)) {
          targets = std::get<std::vector<int>>(v);
        } else if (std::holds_alternative<int>(v)) {
          targets.push_back(std::get<int>(v));
        }
        // Debug: log resolved targets from context variable
        try {
          std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
          if (lout) {
            lout << "RESOLVED_TARGETS src=" << s << " -> [";
            for (size_t _i = 0; _i < targets.size(); ++_i) {
              if (_i) lout << ",";
              lout << targets[_i];
            }
            lout << "]\n";
            lout.close();
          }
        } catch (...) {}
        std::cerr << "RESOLVED_TARGETS src=" << s << " -> [";
        for (size_t _i = 0; _i < targets.size(); ++_i) {
          if (_i) std::cerr << ",";
          std::cerr << targets[_i];
        }
        std::cerr << "]\n";
      } else if (s == "DECK_TOP") {
        is_virtual_target = true;
        virtual_target_type = "DECK_TOP";
        auto c_val = inst.args.contains("count") ? inst.args["count"]
                                                 : nlohmann::json(1);
        virtual_count = resolve_int(c_val);
      } else if (s == "DECK_BOTTOM") {
        is_virtual_target = true;
        virtual_target_type = "DECK_BOTTOM";
        auto c_val = inst.args.contains("count") ? inst.args["count"]
                                                 : nlohmann::json(1);
        virtual_count = resolve_int(c_val);
      // 再発防止: HAND/MANA/BATTLE をバーチャルソースとして処理する。
      //   TRANSITION(from=HAND, input_value_key=...) が生成する MOVE 命令の
      //   target="HAND" を正しく解決するために追加。
      } else if (s == "HAND") {
        is_virtual_target = true;
        virtual_target_type = "HAND";
        // If a $count reference already resolved to an explicit vector of
        // targets above, avoid interpreting the count as a virtual_count
        // which would otherwise cause no-op when resolve_int returns 0.
        if (targets.empty()) {
          auto c_val = inst.args.contains("count") ? inst.args["count"]
                                                   : nlohmann::json(1);
          virtual_count = resolve_int(c_val);
        }
      } else if (s == "MANA") {
        is_virtual_target = true;
        virtual_target_type = "MANA";
        if (targets.empty()) {
          auto c_val = inst.args.contains("count") ? inst.args["count"]
                                                   : nlohmann::json(1);
          virtual_count = resolve_int(c_val);
        }
      } else if (s == "BATTLE") {
        is_virtual_target = true;
        virtual_target_type = "BATTLE";
        if (targets.empty()) {
          auto c_val = inst.args.contains("count") ? inst.args["count"]
                                                   : nlohmann::json(1);
          virtual_count = resolve_int(c_val);
        }
      } else if (s == "BUFFER_REMAIN") {
        // 再発防止: BUFFER_REMAIN は $buffer_select に含まれないバッファ残余カードをすべて移動する。
        //   MOVE_BUFFER_TO_ZONE が生成する残余デッキボトム戻し命令で使用。
        is_virtual_target = true;
        virtual_target_type = "BUFFER_REMAIN";
        virtual_count = INT_MAX;
      }
    } else if (target_val.is_number()) {
      targets.push_back(target_val.get<int>());
    }
  }

  // Support input_value_key as an alternative way to pass targets from previous
  // Query/Select
  if (targets.empty() && inst.args.contains("input_value_key")) {
    auto iv = inst.args["input_value_key"];
    if (iv.is_string()) {
      std::string key = iv.get<std::string>();
      if (key.empty() == false) {
        // Context keys are stored with a leading '$' in the pipeline
        std::string ctx_key = key.rfind("$", 0) == 0 ? key : ("$" + key);
        auto v = get_context_var(ctx_key);
        if (std::holds_alternative<std::vector<int>>(v))
          targets = std::get<std::vector<int>>(v);
        else if (std::holds_alternative<int>(v))
          targets.push_back(std::get<int>(v));
        // Debug: log resolved targets from input_value_key
        try {
          std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
          if (lout) {
            lout << "RESOLVED_INPUT_VALUE_KEY key=" << ctx_key << " -> [";
            for (size_t _i = 0; _i < targets.size(); ++_i) {
              if (_i) lout << ",";
              lout << targets[_i];
            }
            lout << "]\n";
            lout.close();
          }
        } catch (...) {}
        std::cerr << "RESOLVED_INPUT_VALUE_KEY key=" << ctx_key << " -> [";
        for (size_t _i = 0; _i < targets.size(); ++_i) {
          if (_i) std::cerr << ",";
          std::cerr << targets[_i];
        }
        std::cerr << "]\n";
      }
    }
  }

  // Diagnostic: if caller passed a count reference that resolves to a vector
  // (selected instance ids) but targets resolved to empty here, emit a
  // detailed trace to aid debugging. This helps catch cases where a
  // $var_selected exists in pipeline.context but was not picked up by the
  // MOVE resolution logic.
  try {
    if (targets.empty() && inst.args.contains("count") && inst.args["count"].is_string()) {
      std::string count_ref = inst.args["count"].get<std::string>();
      if (!count_ref.empty() && count_ref.rfind("$", 0) == 0) {
        auto cv = get_context_var(count_ref);
        if (std::holds_alternative<std::vector<int>>(cv)) {
          const auto &vec = std::get<std::vector<int>>(cv);
          std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
          if (lout) {
            lout << "DIAG_MOVE_MISS count_ref=" << count_ref << " vec=[";
            for (size_t i = 0; i < vec.size(); ++i) {
              if (i) lout << ",";
              lout << vec[i];
            }
            lout << "] inst_args=" << inst.args.dump() << " context=" << dump_context().dump() << "\n";
            lout.close();
          }
        }
      }
    }
  } catch (...) {}

  std::string to_zone_str = resolve_string(inst.args.value("to", ""));
  Zone to_zone = Zone::GRAVEYARD;
  bool to_bottom = false;
  if (to_zone_str == "HAND")
    to_zone = Zone::HAND;
  else if (to_zone_str == "MANA")
    to_zone = Zone::MANA;
  else if (to_zone_str == "BATTLE")
    to_zone = Zone::BATTLE;
  else if (to_zone_str == "SHIELD")
    to_zone = Zone::SHIELD;
  else if (to_zone_str == "DECK")
    to_zone = Zone::DECK;
  else if (to_zone_str == "DECK_BOTTOM") {
    to_zone = Zone::DECK;
    to_bottom = true;
  } else if (to_zone_str == "BUFFER")
    to_zone = Zone::BUFFER;
  else if (to_zone_str == "STACK")
    to_zone = Zone::STACK;

  if (inst.args.contains("to_bottom") && inst.args["to_bottom"].is_boolean()) {
    to_bottom = inst.args["to_bottom"];
  }

  if (is_virtual_target) {
    PlayerID pid = state.active_player_id;
    const auto &deck = state.players[pid].deck;

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
    } else if (virtual_target_type == "HAND") {
      // 再発防止: HAND バーチャルソース — 手札の末尾（最後に追加されたカード）から取る
      const auto &hand = state.players[pid].hand;
      int available = (int)hand.size();
      int count = std::min(virtual_count, available);
      for (int i = 0; i < count; ++i) {
        targets.push_back(hand[available - 1 - i].instance_id);
      }
    } else if (virtual_target_type == "MANA") {
      const auto &mana = state.players[pid].mana_zone;
      int available = (int)mana.size();
      int count = std::min(virtual_count, available);
      for (int i = 0; i < count; ++i) {
        targets.push_back(mana[available - 1 - i].instance_id);
      }
    } else if (virtual_target_type == "BATTLE") {
      const auto &bz = state.players[pid].battle_zone;
      int available = (int)bz.size();
      int count = std::min(virtual_count, available);
      for (int i = 0; i < count; ++i) {
        targets.push_back(bz[available - 1 - i].instance_id);
      }
    } else if (virtual_target_type == "BUFFER_REMAIN") {
      // 再発防止: $buffer_select に含まれないバッファ残余カードを収集する。
      //   SELECT_FROM_BUFFER または AUTO_SELECT_BUFFER で選択済みのカードを除外する。
      std::vector<int> selected;
      auto sv = get_context_var("$buffer_select");
      if (std::holds_alternative<std::vector<int>>(sv)) {
        selected = std::get<std::vector<int>>(sv);
      } else if (std::holds_alternative<int>(sv) && std::get<int>(sv) > 0) {
        selected.push_back(std::get<int>(sv));
      }
      for (const auto &c : state.players[pid].effect_buffer) {
        if (std::find(selected.begin(), selected.end(), c.instance_id) == selected.end()) {
          targets.push_back(c.instance_id);
        }
      }
    }
  }

  // Debug: record the MOVE instruction's resolved targets and args
  try {
    std::cerr << "MOVE_INSTRUCTION args=" << inst.args.dump() << " resolved_targets=[";
    for (size_t _i = 0; _i < targets.size(); ++_i) {
      if (_i) std::cerr << ",";
      std::cerr << targets[_i];
    }
    std::cerr << "] to_zone=" << to_zone_str << " to_bottom=" << to_bottom << "\n";
  } catch (...) {}

  for (int id : targets) {
    const CardInstance *card_ptr = state.get_card_instance(id);
    if (!card_ptr) {
      for (const auto &p : state.players) {
        for (const auto &c : p.effect_buffer) {
          if (c.instance_id == id) {
            card_ptr = &c;
            break;
          }
        }
        if (card_ptr)
          break;
      }
    }

    if (!card_ptr)
      continue;

    PlayerID owner = card_ptr->owner;
    // If owner is out-of-band, resolve from state; otherwise keep the
    // recorded owner. Previously this branch unconditionally overwrote
    // owner with the active player which caused moves to be applied to the
    // wrong player's zones. Keep existing owner when valid.
    if (owner > 1) {
      owner = state.get_card_owner(id);
    }

    Zone from_zone = Zone::GRAVEYARD;
    bool found = false;
    const Player &p = state.players[owner];

    for (const auto &c : p.hand)
      if (c.instance_id == id) {
        from_zone = Zone::HAND;
        found = true;
        break;
      }
    if (!found)
      for (const auto &c : p.battle_zone)
        if (c.instance_id == id) {
          from_zone = Zone::BATTLE;
          found = true;
          break;
        }
    if (!found)
      for (const auto &c : p.mana_zone)
        if (c.instance_id == id) {
          from_zone = Zone::MANA;
          found = true;
          break;
        }
    if (!found)
      for (const auto &c : p.shield_zone)
        if (c.instance_id == id) {
          from_zone = Zone::SHIELD;
          found = true;
          break;
        }
    if (!found)
      for (const auto &c : p.deck)
        if (c.instance_id == id) {
          from_zone = Zone::DECK;
          found = true;
          break;
        }
    if (!found)
      for (const auto &c : p.graveyard)
        if (c.instance_id == id) {
          from_zone = Zone::GRAVEYARD;
          found = true;
          break;
        }
    if (!found)
      for (const auto &c : p.effect_buffer)
        if (c.instance_id == id) {
          from_zone = Zone::BUFFER;
          found = true;
          break;
        }
    if (!found)
      for (const auto &c : p.stack)
        if (c.instance_id == id) {
          from_zone = Zone::STACK;
          found = true;
          break;
        }

    if (!found)
      continue;

    // 再発防止: CANNOT_LEAVE_BATTLE パッシブが有効な場合、バトルゾーンからの離脱を防ぐ
    if (from_zone == Zone::BATTLE) {
      const auto &db = dm::engine::infrastructure::CardRegistry::get_all_definitions();
      const CardInstance *check_card = state.get_card_instance(id);
      if (check_card && PassiveEffectSystem::instance().check_restriction(
                            state, *check_card, PassiveType::CANNOT_LEAVE_BATTLE, db)) {
        continue;
      }
    }

    int dest_idx = to_bottom ? 0 : -1;
    auto cmd = std::make_unique<TransitionCommand>(id, from_zone, to_zone,
                                                   owner, dest_idx);
    // Temp debug: record attempted transition details
    try {
      std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
      if (lout) {
        lout << "PIPELINE_MOVE id=" << id
             << " from=" << static_cast<int>(from_zone)
             << " to=" << static_cast<int>(to_zone) << " owner=" << owner
             << "\n";
        lout.close();
      }
    } catch (...) {
    }

    execute_command(std::move(cmd), state);
  }
}

void PipelineExecutor::handle_modify(const Instruction &inst,
                                     GameState &state) {
  if (inst.args.is_null())
    return;

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
    if (target_val.is_string() &&
        target_val.get<std::string>().rfind("$", 0) == 0) {
      auto v = get_context_var(target_val.get<std::string>());
      if (std::holds_alternative<std::vector<int>>(v))
        targets = std::get<std::vector<int>>(v);
      else if (std::holds_alternative<int>(v))
        targets.push_back(std::get<int>(v));
    } else if (target_val.is_number()) {
      targets.push_back(target_val.get<int>());
    }
  }

  MutateCommand::MutationType type;
  auto val_json =
      inst.args.contains("value") ? inst.args["value"] : nlohmann::json(0);
  int val = resolve_int(val_json);
  std::string str_val = resolve_string(inst.args.value("str_value", ""));

  if (mod_type_str == "TAP")
    type = MutateCommand::MutationType::TAP;
  else if (mod_type_str == "UNTAP")
    type = MutateCommand::MutationType::UNTAP;
  else if (mod_type_str == "POWER_ADD")
    type = MutateCommand::MutationType::POWER_MOD;
  else if (mod_type_str == "ADD_KEYWORD")
    type = MutateCommand::MutationType::ADD_KEYWORD;
  else if (mod_type_str == "REMOVE_KEYWORD")
    type = MutateCommand::MutationType::REMOVE_KEYWORD;
  else if (mod_type_str == "ADD_PASSIVE")
    type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
  else if (mod_type_str == "ADD_COST_MODIFIER")
    type = MutateCommand::MutationType::ADD_COST_MODIFIER;
  else if (mod_type_str == "STAT") {
    StatCommand::StatType s_type;
    std::string stat_name = resolve_string(inst.args.value("stat", ""));
    if (stat_name == "CARDS_DRAWN")
      s_type = StatCommand::StatType::CARDS_DRAWN;
    else if (stat_name == "CARDS_DISCARDED")
      s_type = StatCommand::StatType::CARDS_DISCARDED;
    else if (stat_name == "CREATURES_PLAYED")
      s_type = StatCommand::StatType::CREATURES_PLAYED;
    else if (stat_name == "SPELLS_CAST")
      s_type = StatCommand::StatType::SPELLS_CAST;
    else
      return;

    auto cmd = std::make_unique<StatCommand>(s_type, val);
    execute_command(std::move(cmd), state);
    return;
  } else
    return;

  // --- NEW: Intercept specific keywords to create PASSIVE EFFECTS instead ---
  // Also support passing them directly via ADD_PASSIVE to streamline
  // ModifierHandler
  bool is_passive_keyword =
      (type == MutateCommand::MutationType::ADD_KEYWORD &&
       (str_val == "CANNOT_ATTACK" || str_val == "CANNOT_BLOCK" ||
        str_val == "CANNOT_ATTACK_OR_BLOCK" || str_val == "CANNOT_LEAVE_BATTLE"));

  if (is_passive_keyword) {
    // Treat as ADD_PASSIVE flow below but with specific target iteration
  }

  // Handling ADD_PASSIVE for both Filter-based (global/broad) and Target-based
  // (specific)
  if (type == MutateCommand::MutationType::ADD_PASSIVE_EFFECT ||
      is_passive_keyword) {
    std::vector<PassiveType> p_types;
    if (str_val == "LOCK_SPELL")
      p_types.push_back(PassiveType::CANNOT_USE_SPELLS);
    else if (str_val == "POWER")
      p_types.push_back(PassiveType::POWER_MODIFIER);
    else if (str_val == "CANNOT_ATTACK")
      p_types.push_back(PassiveType::CANNOT_ATTACK);
    else if (str_val == "CANNOT_BLOCK")
      p_types.push_back(PassiveType::CANNOT_BLOCK);
    else if (str_val == "CANNOT_ATTACK_OR_BLOCK") {
      p_types.push_back(PassiveType::CANNOT_ATTACK);
      p_types.push_back(PassiveType::CANNOT_BLOCK);
    } else if (str_val == "IGNORE_ABILITY") {
      p_types.push_back(PassiveType::IGNORE_ABILITIES);
    } else if (str_val == "CANNOT_LEAVE_BATTLE") {
      p_types.push_back(PassiveType::CANNOT_LEAVE_BATTLE);
    } else {
      // Fallback or unknown
    }

    if (!p_types.empty()) {
      int duration = inst.args.value("duration", 1);
      int source_id = -1;
      auto v = get_context_var("$source");
      if (std::holds_alternative<int>(v))
        source_id = std::get<int>(v);

      // Resolve filter references from execution context
      FilterDef filter = inst.args.value("filter", FilterDef{});
      if (filter.cost_ref.has_value()) {
        const auto &key = filter.cost_ref.value();
        auto ctx_val = get_context_var(key);
        if (std::holds_alternative<int>(ctx_val)) {
          filter.exact_cost = std::get<int>(ctx_val);
        }
      }

      // If targets are provided (e.g. from Select or specific target logic), we
      // apply specifically
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

            auto cmd = std::make_unique<MutateCommand>(
                -1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
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

          auto cmd = std::make_unique<MutateCommand>(
              -1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
          cmd->passive_effect = eff;
          execute_command(std::move(cmd), state);
        }
      }
      return;
    }
    // If type matches but str_val not handled, fall through?
    // Or if it was supposed to be simple ADD_PASSIVE without str_val mapping
    // logic (unlikely in current design).
  }

  if (type == MutateCommand::MutationType::ADD_PASSIVE_EFFECT) {
    // Fallback for cases not handled above (e.g. direct integer type set?)
    // Currently we rely on str_val. If str_val is empty or unknown, we might
    // have an issue. But existing code only handled LOCK_SPELL and POWER. Let's
    // keep the old block logic just in case but integrated. The above block
    // covers LOCK_SPELL and POWER. So we effectively replaced the logic.
    return;
  }

  if (type == MutateCommand::MutationType::ADD_COST_MODIFIER) {
    CostModifier mod;
    mod.reduction_amount = val;
    mod.condition_filter = inst.args.value("filter", FilterDef{});
    mod.turns_remaining = inst.args.value("duration", 1);
    int source_id = -1;
    auto v = get_context_var("$source");
    if (std::holds_alternative<int>(v))
      source_id = std::get<int>(v);
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

void PipelineExecutor::handle_if(
    const Instruction &inst, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  if (inst.args.is_null() || !inst.args.contains("cond")) {
    call_stack.back().pc++;
    return;
  }

  bool res = check_condition(inst.args["cond"], state, card_db);

  auto block =
      res ? std::make_shared<std::vector<Instruction>>(inst.then_block)
          : std::make_shared<std::vector<Instruction>>(inst.else_block);

  call_stack.back().pc++;

  {
    size_t before_size = call_stack.size();
    int parent_idx = (before_size > 0) ? (int)before_size - 1 : -1;
    int parent_pc = -1;
    if (parent_idx >= 0)
      parent_pc = call_stack[parent_idx].pc;
    std::string inst_dump = "{}";
    if (block && !block->empty())
      inst_dump = (*block)[0].args.dump();
    std::fprintf(stderr,
                 "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d parent_pc=%d "
                 "inst=%s\n",
                 __FILE__, __LINE__, before_size, parent_idx, parent_pc,
                 inst_dump.c_str());
    call_stack.push_back({block, 0, LoopContext{}});
    size_t after_size = call_stack.size();
    std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__,
                 __LINE__, after_size);
  }
}

void PipelineExecutor::handle_loop(
    const Instruction &inst, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  auto &frame = call_stack.back();
  auto &ctx = frame.loop_ctx;

  if (!ctx.active) {
    ctx.active = true;
    ctx.index = 0;

    if (inst.op == InstructionOp::REPEAT ||
        (inst.args.contains("count") && !inst.args.contains("in"))) {
      auto c_val =
          inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
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
      if (parent_idx >= 0)
        parent_pc = call_stack[parent_idx].pc;
      std::string inst_dump = "{}";
      if (block && !block->empty())
        inst_dump = (*block)[0].args.dump();
      std::fprintf(stderr,
                   "[DIAG PUSH] %s:%d before_size=%zu parent_idx=%d "
                   "parent_pc=%d inst=%s\n",
                   __FILE__, __LINE__, before_size, parent_idx, parent_pc,
                   inst_dump.c_str());
      call_stack.push_back({block, 0, LoopContext{}});
      size_t after_size = call_stack.size();
      std::fprintf(stderr, "[DIAG PUSH] %s:%d after_size=%zu\n", __FILE__,
                   __LINE__, after_size);
    }
  } else {
    ctx.active = false;
    frame.pc++;
  }
}

void PipelineExecutor::handle_calc(const Instruction &inst,
                                   GameState & /*state*/) {
  if (inst.args.is_null())
    return;
  std::string out_key = inst.args.value("out", "$result");
  if (inst.op == InstructionOp::MATH) {
    auto l_val =
        inst.args.contains("lhs") ? inst.args["lhs"] : nlohmann::json(0);
    auto r_val =
        inst.args.contains("rhs") ? inst.args["rhs"] : nlohmann::json(0);
    int lhs = resolve_int(l_val);
    int rhs = resolve_int(r_val);
    std::string op = inst.args.value("op", "+");
    int res = 0;
    if (op == "+")
      res = lhs + rhs;
    else if (op == "-")
      res = lhs - rhs;
    else if (op == "*")
      res = lhs * rhs;
    else if (op == "/")
      res = (rhs != 0) ? lhs / rhs : 0;

    set_context_var(out_key, res);
  } else if (inst.op == InstructionOp::COUNT) {
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

void PipelineExecutor::handle_print(const Instruction &inst,
                                    GameState & /*state*/) {
  if (inst.args.is_null())
    return;
  std::cout << "[Pipeline] " << resolve_string(inst.args.value("msg", ""))
            << std::endl;
}

void PipelineExecutor::handle_wait_input(const Instruction &inst,
                                         GameState &state) {
  std::cerr << "[PipelineExecutor::handle_wait_input] Called" << std::endl;
  if (inst.args.is_null()) {
    std::cerr << "[PipelineExecutor::handle_wait_input] args is null, returning"
              << std::endl;
    return;
  }

  std::string out_key = inst.args.value("out", "$input");
  std::cerr << "[PipelineExecutor::handle_wait_input] out_key=" << out_key
            << std::endl;

  // If we already have the value (via resume), don't pause again.
  // Use get_context_var which handles $-prefixed vs non-prefixed keys.
  auto existing_val = get_context_var(out_key);
  if (!std::holds_alternative<std::monostate>(existing_val)) {
    std::cerr << "[PipelineExecutor::handle_wait_input] Already have value in context, returning" << std::endl;
    // Dump full pipeline context for correlation debugging
    try {
      auto j = dump_context();
      std::cerr << "[PipelineExecutor::handle_wait_input] FULL_CONTEXT=" << j.dump() << std::endl;
      try {
        std::filesystem::create_directories("logs");
        std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
        if (lout) {
          lout << "[PipelineExecutor::handle_wait_input] FULL_CONTEXT=" << j.dump() << "\n";
          lout.close();
        }
      } catch (...) {}
    } catch (...) {}
    return;
  }

  std::string query_type = inst.args.value("query_type", "NONE");
  std::cerr << "[PipelineExecutor::handle_wait_input] query_type=" << query_type
            << std::endl;
  std::vector<std::string> options;
  if (inst.args.contains("options")) {
    for (const auto &opt : inst.args["options"]) {
      if (opt.is_string())
        options.push_back(opt.get<std::string>());
    }
  }

  execution_paused = true;
  // Normalize waiting_for_key to always include leading '$' so resume() and
  // set_context_var use a consistent key form.
  waiting_for_key = (out_key.rfind("$", 0) == 0) ? out_key : ("$" + out_key);
  state.waiting_for_user_input = true;
  std::cerr << "[PipelineExecutor::handle_wait_input] Setting "
               "execution_paused=true, waiting_for_user_input=true"
            << std::endl;

  // Dump context at the moment we set pending_query for later correlation
  try {
    auto j2 = dump_context();
    std::cerr << "[PipelineExecutor::handle_wait_input] CONTEXT_AT_PENDING=" << j2.dump() << std::endl;
    try {
      std::filesystem::create_directories("logs");
      std::ofstream lout2("logs/pipeline_trace.txt", std::ios::app);
      if (lout2) {
        lout2 << "[PipelineExecutor::handle_wait_input] CONTEXT_AT_PENDING=" << j2.dump() << "\n";
        lout2.close();
      }
    } catch (...) {}
  } catch (...) {}

  // Setup pending query with min/max for SELECT_NUMBER
  std::map<std::string, int> param_map;
  if (inst.args.contains("min")) {
    int min_val = resolve_int(inst.args["min"]);
    param_map["min"] = min_val;
    std::cerr << "[PipelineExecutor::handle_wait_input] min=" << min_val
              << std::endl;
  }
  if (inst.args.contains("max")) {
    int max_val = resolve_int(inst.args["max"]);
    param_map["max"] = max_val;
    std::cerr << "[PipelineExecutor::handle_wait_input] max=" << max_val
              << std::endl;
  }

  // Ensure valid_targets is initialized as an empty array (vector<int>), not
  // an empty JSON object. Construct QueryContext with correctly-typed
  // parameters to avoid aggregate-initializer type/order mismatches.
  state.pending_query = GameState::QueryContext{0, query_type, param_map,
                                                std::vector<int>{}, options};
  std::cerr
      << "[PipelineExecutor::handle_wait_input] pending_query set: query_type="
      << state.pending_query.query_type << std::endl;
}

bool PipelineExecutor::check_condition(
    const nlohmann::json &cond, GameState &state,
    const std::map<core::CardID, core::CardDefinition> &card_db) {
  if (cond.is_null())
    return false;

  if (cond.contains("type")) {
    std::string type = cond.value("type", "NONE");
    if (type != "NONE") {
      core::ConditionDef def;
      // Use centralized JSON parsing to ensure all fields (including filter)
      // are handled
      dm::core::from_json(cond, def);

      int source_id = -1;
      auto v = get_context_var("$source");
      if (std::holds_alternative<int>(v))
        source_id = std::get<int>(v);
      else if (std::holds_alternative<std::vector<int>>(v)) {
        const auto &vec = std::get<std::vector<int>>(v);
        if (!vec.empty())
          source_id = vec[0];
      }

      std::map<std::string, int> exec_ctx;
      for (const auto &kv : context) {
        if (std::holds_alternative<int>(kv.second)) {
          exec_ctx[kv.first] = std::get<int>(kv.second);
        }
      }

      return dm::engine::rules::ConditionSystem::instance().evaluate_def(
          state, def, source_id, card_db, exec_ctx);
    }
  }

  if (cond.contains("exists")) {
    std::string key = cond["exists"];
    auto v = get_context_var(key);
    if (std::holds_alternative<std::vector<int>>(v)) {
      return !std::get<std::vector<int>>(v).empty();
    }
    if (std::holds_alternative<int>(v))
      return true;
  }

  if (cond.contains("op")) {
    int lhs = 0;
    if (cond.contains("lhs"))
      lhs = resolve_int(cond["lhs"]);

    int rhs = 0;
    if (cond.contains("rhs"))
      rhs = resolve_int(cond["rhs"]);

    std::string op = cond.value("op", "==");
    if (op == "==")
      return lhs == rhs;
    if (op == ">")
      return lhs > rhs;
    if (op == "<")
      return lhs < rhs;
    if (op == ">=")
      return lhs >= rhs;
    if (op == "<=")
      return lhs <= rhs;
    if (op == "!=")
      return lhs != rhs;
  }
  return false;
}

} // namespace dm::engine::systems
