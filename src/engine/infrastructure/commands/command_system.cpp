#include "command_system.hpp"
#include "engine/infrastructure/data/card_registry.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"
#include "engine/systems/rules/condition_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <set>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <unordered_set>


namespace dm::engine::systems {

using namespace dm::core;

Zone parse_zone_string(const std::string &zone_str);

// Count cards logic (same as before)
static int count_matching_cards(GameState &state, const CommandDef &cmd,
                                PlayerID player_id,
                                std::map<std::string, int> &execution_context) {
  std::vector<PlayerID> players_to_check;
  if (cmd.target_group == TargetScope::PLAYER_SELF ||
      cmd.target_group == TargetScope::SELF) {
    players_to_check.push_back(player_id);
  } else if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
    players_to_check.push_back(1 - player_id);
  } else if (cmd.target_group == TargetScope::ALL_PLAYERS) {
    players_to_check.push_back(player_id);
    players_to_check.push_back(1 - player_id);
  }

  if (players_to_check.empty()) {
    players_to_check.push_back(player_id);
  }

  core::FilterDef filter = cmd.target_filter;
  filter.count.reset();
  if (filter.zones.empty()) {
    filter.zones.push_back("BATTLE_ZONE");
  }

  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();
  int total = 0;

  for (PlayerID pid : players_to_check) {
    for (const auto &zone_str : filter.zones) {
      Zone zone_enum = parse_zone_string(zone_str);
      const std::vector<CardInstance> *container = nullptr;
      switch (zone_enum) {
      case Zone::HAND:
        container = &state.players[pid].hand;
        break;
      case Zone::MANA:
        container = &state.players[pid].mana_zone;
        break;
      case Zone::BATTLE:
        container = &state.players[pid].battle_zone;
        break;
      case Zone::GRAVEYARD:
        container = &state.players[pid].graveyard;
        break;
      case Zone::SHIELD:
        container = &state.players[pid].shield_zone;
        break;
      case Zone::DECK:
        container = &state.players[pid].deck;
        break;
      default:
        break;
      }
      if (!container)
        continue;

      for (const auto &card : *container) {
        auto def_it = card_db.find(card.card_id);
        if (def_it != card_db.end()) {
          if (dm::engine::utils::TargetUtils::is_valid_target(
                  card, def_it->second, filter, state, player_id, pid, false,
                  &execution_context)) {
            total++;
          }
        } else if (card.card_id == 0) {
          if (dm::engine::utils::TargetUtils::is_valid_target(
                  card, CardDefinition(), filter, state, player_id, pid, false,
                  &execution_context)) {
            total++;
          }
        }
      }
    }
  }
  return total;
}

Zone parse_zone_string(const std::string &zone_str) {
  if (zone_str == "DECK" || zone_str == "DECK_BOTTOM")
    return Zone::DECK;
  if (zone_str == "HAND")
    return Zone::HAND;
  if (zone_str == "MANA" || zone_str == "MANA_ZONE")
    return Zone::MANA;
  if (zone_str == "BATTLE" || zone_str == "BATTLE_ZONE")
    return Zone::BATTLE;
  if (zone_str == "GRAVEYARD")
    return Zone::GRAVEYARD;
  if (zone_str == "SHIELD" || zone_str == "SHIELD_ZONE")
    return Zone::SHIELD;
  return Zone::GRAVEYARD;
}

int CommandSystem::resolve_amount(
    const CommandDef &cmd,
    const std::map<std::string, int> &execution_context) {
  if (!cmd.input_value_key.empty()) {
    auto it = execution_context.find(cmd.input_value_key);
    if (it != execution_context.end()) {
      return it->second;
    }
  }
  return cmd.amount;
}

void CommandSystem::execute_command(
    GameState &state, const CommandDef &cmd, int source_instance_id,
    PlayerID player_id, std::map<std::string, int> &execution_context) {
  // Legacy support: Generate instructions and execute via temporary pipeline
  auto instructions = generate_instructions(state, cmd, source_instance_id,
                                            player_id, execution_context);

  dm::engine::systems::PipelineExecutor pipeline;
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  // Seed pipeline context with current execution_context
  for (const auto &kv : execution_context) {
    pipeline.set_context_var(kv.first, kv.second);
  }

  // Debug: dump generated instructions for this command to stderr
  try {
    std::cerr << "GENERATED_INSTRUCTIONS for cmd.type=" << static_cast<int>(cmd.type) << " count=" << instructions.size() << "\n";
    for (const auto &inst : instructions) {
      try { std::cerr << "  inst.op=" << static_cast<int>(inst.op) << " args=" << inst.args.dump() << "\n"; } catch(...) {}
    }
  } catch (...) {}

  pipeline.execute(instructions, state, card_db);

  // After execution, dump the pipeline context for debugging to stderr and pipeline_trace
  try {
    auto ctx = pipeline.dump_context();
    std::cerr << "PIPELINE_CONTEXT_DUMP: " << ctx.dump() << "\n";
    try {
      std::filesystem::create_directories("logs");
      std::ofstream lout("logs/pipeline_trace.txt", std::ios::app);
      if (lout) {
        lout << "PIPELINE_CONTEXT_DUMP: " << ctx.dump() << "\n";
        lout.close();
      }
    } catch (...) {}
  } catch (...) {}

  // Write back context output if needed (simple int/string/vec mapping)
  // PipelineExecutor context uses variants, simplified back mapping:
  for (const auto &kv : pipeline.context) {
    if (std::holds_alternative<int>(kv.second)) {
      execution_context[kv.first] = std::get<int>(kv.second);
    } else if (std::holds_alternative<std::vector<int>>(kv.second)) {
      // Write back the count of selected targets into execution_context as
      // an integer under the plain key name (no leading '$'). This provides
      // downstream code that expects an integer count access to the number
      // of selected targets without coercing or losing the actual vector in
      // pipeline.context. Do NOT overwrite existing int values if present.
      const auto &vec = std::get<std::vector<int>>(kv.second);
      std::string key = kv.first;
      std::string plain = (!key.empty() && key[0] == '$') ? key.substr(1) : key;
      // Only write back the count if execution_context doesn't already have a
      // meaningful value for this plain key to avoid clobbering caller data.
      if (!plain.empty() && execution_context.count(plain) == 0) {
        execution_context[plain] = static_cast<int>(vec.size());
      }
      continue;
    }
    // Note: string / other types are ignored for now.
  }

  // Also write back keys without leading '$' to avoid $-prefixed vs non-prefixed
  // mismatches when callers look up execution_context by raw key name.
  for (const auto &kv : pipeline.context) {
    std::string key = kv.first;
    if (!key.empty() && key[0] == '$') {
      std::string plain = key.substr(1);
      if (std::holds_alternative<int>(kv.second)) {
        execution_context[plain] = std::get<int>(kv.second);
      } else if (std::holds_alternative<std::vector<int>>(kv.second)) {
        // Also provide a plain-key count for vectors if missing (same semantics
        // as the loop above). Avoid overwriting existing plain-key ints.
        const auto &vec = std::get<std::vector<int>>(kv.second);
        if (!plain.empty() && execution_context.count(plain) == 0) {
          execution_context[plain] = static_cast<int>(vec.size());
        }
        continue;
      }
    }
  }
}

std::vector<Instruction> CommandSystem::generate_instructions(
    GameState &state, const CommandDef &cmd, int source_instance_id,
    PlayerID player_id, std::map<std::string, int> &execution_context) {
  std::vector<Instruction> out;

  // Ensure defaults are loaded
  dm::engine::rules::ConditionSystem::instance().initialize_defaults();

  switch (cmd.type) {
  // Primitive Commands
  case core::CommandType::TRANSITION:
  case core::CommandType::MUTATE:
  case core::CommandType::FLOW:
  case core::CommandType::IF:      // 再発防止: IF/IF_ELSE は FLOW と同様に generate_primitive_instructions で処理
  case core::CommandType::IF_ELSE: // target_filter または condition で条件を記述する
  case core::CommandType::ELSE:
  case core::CommandType::QUERY:
  case core::CommandType::SHUFFLE_DECK:
    generate_primitive_instructions(out, state, cmd, source_instance_id,
                                    player_id, execution_context);
    break;

  case core::CommandType::MANA_CHARGE: {
    nlohmann::json args;
    args["card"] =
        (cmd.instance_id != -1) ? cmd.instance_id : source_instance_id;
    Instruction inst(InstructionOp::GAME_ACTION, args);
    inst.args["type"] = "MANA_CHARGE";
    out.push_back(inst);
  } break;

  // Macro Commands
  case core::CommandType::DRAW_CARD:
  case core::CommandType::BOOST_MANA:
  case core::CommandType::ADD_MANA:
  case core::CommandType::DESTROY:
  case core::CommandType::DISCARD:
  case core::CommandType::TAP:
  case core::CommandType::UNTAP:
  case core::CommandType::RETURN_TO_HAND:
  case core::CommandType::BREAK_SHIELD:
  case core::CommandType::POWER_MOD:
  case core::CommandType::ADD_KEYWORD:
  case core::CommandType::GRANT_KEYWORD:  // 再発防止: GRANT_KEYWORD は ADD_KEYWORD と同様に処理
  case core::CommandType::SEARCH_DECK:
  case core::CommandType::SEND_TO_MANA:
  case core::CommandType::SELECT_NUMBER:  // 再発防止: SELECT_NUMBER は WAIT_INPUT を生成
  case core::CommandType::SELECT_OPTION:  // 再発防止: SELECT_OPTION は CHOICE と同様に処理
  case core::CommandType::APPLY_MODIFIER: // 再発防止: APPLY_MODIFIER は一時キーワード付与にマップ
  case core::CommandType::PUT_CREATURE:   // 再発防止: PUT_CREATURE は SUMMON_TOKEN 的に処理
  case core::CommandType::CAST_SPELL:     // 再発防止: CAST_SPELL は GAME_ACTION(CAST_SPELL) を生成
  case core::CommandType::ADD_RESTRICTION:
  case core::CommandType::REPLACE_CARD_MOVE:  // 再発防止: REPLACE_CARD_MOVE はゾーン置換効果
  case core::CommandType::DRAW:               // 再発防止: DRAW は DRAW_CARD と同一ブロックで処理 (generate_macro_instructions 内で統合済)
  case core::CommandType::REPLACE_MOVE_CARD:  // 再発防止: REPLACE_MOVE_CARD はマグナム系置換効果をインストール
  case core::CommandType::LOOK_TO_BUFFER:      // 再発防止: デッキ上から指定枚数を「見る」バッファ展開 (REVEAL_TO_BUFFER と同一実装)
  case core::CommandType::REVEAL_TO_BUFFER:   // 再発防止: デッキ上からバッファへ表向きに展開
  case core::CommandType::SELECT_FROM_BUFFER: // 再発防止: バッファからプレイヤーが選択
  case core::CommandType::MOVE_BUFFER_TO_ZONE: // 再発防止: バッファ残予を指定ゾーンへ移動
  case core::CommandType::MOVE_BUFFER_REMAIN_TO_ZONE: // 再発防止: バッファ残余（選択外）をすべて指定ゾーンへ移動
  case core::CommandType::REGISTER_DELAYED_EFFECT: // 遅延効果登録
  // 再発防止: 制限コマンド — ADD_PASSIVE 命令にマップしてパッシブ効果を登録する
  case core::CommandType::LOCK_SPELL:
  case core::CommandType::SPELL_RESTRICTION:
  case core::CommandType::CANNOT_PUT_CREATURE:
  case core::CommandType::CANNOT_SUMMON_CREATURE:
  case core::CommandType::PLAYER_CANNOT_ATTACK:
  case core::CommandType::IGNORE_ABILITY:
    generate_macro_instructions(out, state, cmd, source_instance_id, player_id,
                                execution_context);
    break;

  // 再発防止: IF/IF_ELSE/ELSE は上部の primitive block に含まれているため重複 case を排除すること

  case core::CommandType::REVOLUTION_CHANGE:
    // 再発防止: REVOLUTION_CHANGE は cards.json の effects[].commands[] に「能力宣言」として
    //   記述されており、generate_instructions ではMOVE命令を生成してはならない。
    //   実際のカードスワップ処理は以下の経路で行われる:
    //     trigger_manager.cpp → ReactionType::REVOLUTION_CHANGE を reaction スタックに登録
    //     → intent_generator.cpp が USE_ABILITY CommandDef を生成
    //     → commands.cpp::DeclareReactionCommand → PendingEffect(TRIGGER_ABILITY)
    //     → pending_strategy.cpp が swap 命令を生成
    //   ここで MOVE 命令を生成すると ON_PLAY など無関係な効果起動時に
    //   不正なカード移動が起きてテスト失敗の原因になる（発生済みバグ #RC-001)。
    break;

  default:
    break;
  }
  return out;
}

void CommandSystem::generate_primitive_instructions(
    std::vector<Instruction> &out, GameState &state, const CommandDef &cmd,
    int source_instance_id, PlayerID player_id,
    std::map<std::string, int> &execution_context) {
  if (cmd.type == core::CommandType::FLOW ||
      cmd.type == core::CommandType::IF ||
      cmd.type == core::CommandType::IF_ELSE ||
      cmd.type == core::CommandType::ELSE) {
    // 再発防止: IF/IF_ELSE は cards.json で target_filter に条件を記述する場合がある。
    //   FLOW は condition フィールドを使用する。
    //   どちらの形式も if_true/if_false ブランチは共通。
    bool cond_result = true;
    if (cmd.condition.has_value()) {
      const auto &card_db =
          dm::engine::infrastructure::CardRegistry::get_all_definitions();
      cond_result = dm::engine::rules::ConditionSystem::instance().evaluate_def(
          state, cmd.condition.value(), source_instance_id, card_db,
          execution_context);
    } else if (cmd.target_filter.type.has_value()) {
      // 再発防止: cards.json の IF コマンドは target_filter に条件データを埋め込む場合がある。
      //   FilterDef.typeで条件タイプを出力し、ConditionDefに変換して評価する。
      //   cmd.input_value_key は COMPARE_INPUT 条件の変数名として使用する。
      core::ConditionDef derived_cond;
      derived_cond.type  = *cmd.target_filter.type;
      derived_cond.value = cmd.target_filter.value.value_or(0);
      derived_cond.op    = cmd.target_filter.op.value_or(">=");
      // COMPARE_INPUT / PLAYED_WITHOUT_MANA_TARGET: stat_key = cmd.input_value_key
      // 再発防止: 入力リンクを参照する条件タイプを追加したら、この分岐にも必ず追記すること。
      if ((derived_cond.type == "COMPARE_INPUT" || derived_cond.type == "PLAYED_WITHOUT_MANA_TARGET")
          && !cmd.input_value_key.empty()) {
        derived_cond.stat_key = cmd.input_value_key;
      }
      const auto &card_db =
          dm::engine::infrastructure::CardRegistry::get_all_definitions();
      cond_result = dm::engine::rules::ConditionSystem::instance().evaluate_def(
          state, derived_cond, source_instance_id, card_db, execution_context);
    }
    // ELSE は常に if_true ブランチを実行する
    if (cmd.type == core::CommandType::ELSE) {
      cond_result = true;
    }

    const auto &branch = cond_result ? cmd.if_true : cmd.if_false;
    for (const auto &child_cmd : branch) {
      auto sub = generate_instructions(state, child_cmd, source_instance_id,
                                       player_id, execution_context);
      out.insert(out.end(), sub.begin(), sub.end());
    }

  } else if (cmd.type == core::CommandType::TRANSITION) {
    std::vector<int> targets;
    // If this TRANSITION expects input_value_key (user-provided amount/targets),
    // do not pre-resolve targets at generation time. Resolving now would cause
    // the engine to select default targets prematurely and skip runtime SELECT
    // instruction generation. Keep targets empty so the input_value_key path
    // below can generate SELECT/MOVE sequence as needed.
    if (cmd.input_value_key.empty()) {
      targets = resolve_targets(state, cmd, source_instance_id,
                                player_id, execution_context);
    }
    Zone from_z = parse_zone_string(cmd.from_zone);
    std::string to_z_str = cmd.to_zone; // Pass string to instruction

    // 再発防止: input_value_key が指定されている場合、生成時にコンテキスト変数を解決しない。
    // WAIT_INPUT で決定される値は生成時には未確定（ゼロ）のため。
    if (targets.empty() && !cmd.input_value_key.empty()) {
      std::string ctx_key = cmd.input_value_key.rfind("$", 0) == 0
                                ? cmd.input_value_key
                                : "$" + cmd.input_value_key;
      // 再発防止: input_value_usage はデータ由来で大文字小文字/空白ゆらぎがあり得るため、
      //   比較前に正規化して分岐ミス（TARGETS が else に落ちる）を防ぐ。
      std::string usage_norm = cmd.input_value_usage;
      usage_norm.erase(
          std::remove_if(usage_norm.begin(), usage_norm.end(),
                         [](unsigned char ch) { return std::isspace(ch) != 0; }),
          usage_norm.end());
      std::transform(usage_norm.begin(), usage_norm.end(), usage_norm.begin(),
                     [](unsigned char ch) { return (char)std::toupper(ch); });

      if (cmd.from_zone.empty()) {
        // 再発防止: from_zone が空: input_value_key = インスタンスIDリスト → 直接 MOVE
        Instruction move(InstructionOp::MOVE);
        move.args["target"] = ctx_key;
        move.args["to"] = to_z_str;
        out.push_back(move);
      } else if (usage_norm == "AMOUNT") {
        // 再発防止: from_zone + AMOUNT モード: input_value_key = 移動枚数(ユーザーが選択した数)
        //   ユーザーが from_zone から N 枚を選択してから MOVE する。
        //   旧実装 (MOVE(target=from_zone, count=N)) はユーザー選択を省略するバグがあった。
        //   SELECT 命令で対象を選ばせ、ctx_key+"_selected" に格納してから MOVE する。
        std::string select_out_key = ctx_key + "_selected";
        Instruction select_inst(InstructionOp::SELECT);
        nlohmann::json filter_json = nlohmann::json::object();
        filter_json["zones"] = nlohmann::json::array({cmd.from_zone});
        filter_json["owner"] = "SELF";
        select_inst.args["filter"] = filter_json;
        select_inst.args["count"] = ctx_key;  // コンテキスト変数参照 ($ プレフィックス済み)
        select_inst.args["out"] = select_out_key;
        out.push_back(select_inst);

        Instruction move(InstructionOp::MOVE);
        move.args["target"] = select_out_key;
        move.args["to"] = to_z_str;
        out.push_back(move);
      } else {
        // 再発防止: from_zone + input_value_key の既定は TARGETS 扱いにする。
        //   usage が空/不正でも、ユーザー選択済みの instance_id リストを
        //   直接 target に渡せば MOVE 側で vector<int> を解決できる。
        //   (AMOUNT のみ上の分岐で明示処理)
        Instruction move(InstructionOp::MOVE);
        move.args["target"] = ctx_key;
        move.args["to"] = to_z_str;
        out.push_back(move);
      }
      return;
    }

    if (targets.empty() && cmd.amount > 0 && from_z != Zone::GRAVEYARD) {
      int count = resolve_amount(cmd, execution_context);
      if (from_z == Zone::DECK) {
        Instruction move(InstructionOp::MOVE);
        move.args["target"] = "DECK_TOP";
        move.args["count"] = count;
        move.args["to"] = to_z_str;
        out.push_back(move);

        if (!cmd.output_value_key.empty()) {
          Instruction calc(InstructionOp::MATH);
          calc.args["lhs"] = count;
          calc.args["op"] = "+";
          calc.args["rhs"] = 0;
          calc.args["out"] = cmd.output_value_key;
          out.push_back(calc);
        }
        return;
      }
    }

    for (int target_id : targets) {
      Instruction move(InstructionOp::MOVE);
      move.args["target"] = target_id;
      move.args["to"] = to_z_str;
      out.push_back(move);
    }

    if (!cmd.output_value_key.empty()) {
      Instruction calc(InstructionOp::MATH);
      calc.args["lhs"] = (int)targets.size();
      calc.args["op"] = "+";
      calc.args["rhs"] = 0;
      calc.args["out"] = cmd.output_value_key;
      out.push_back(calc);
    }

  } else if (cmd.type == core::CommandType::QUERY) {
    std::string query_type =
        cmd.str_param.empty() ? "SELECT_TARGET" : cmd.str_param;
    if (query_type == "CARDS_MATCHING_FILTER") {
      int count =
          count_matching_cards(state, cmd, player_id, execution_context);
      if (!cmd.output_value_key.empty()) {
        Instruction calc(InstructionOp::MATH);
        calc.args["lhs"] = count;
        calc.args["op"] = "+";
        calc.args["rhs"] = 0;
        calc.args["out"] = cmd.output_value_key;
        out.push_back(calc);
      }
    } else if (query_type == "MANA_CIVILIZATION_COUNT") {
      Instruction get(InstructionOp::GET_STAT);
      get.args["stat"] = "MANA_CIVILIZATION_COUNT";
      get.args["out"] =
          cmd.output_value_key.empty() ? "$temp" : cmd.output_value_key;
      out.push_back(get);
    } else if (query_type == "SELECT_OPTION") {
      // 再発防止: QUERY(SELECT_OPTION) は新仕様で「カード選択」を行う。
      //   target_filter + amount/input_value_key を使い、選択結果を output_value_key へ保存する。
      //   旧仕様（str_val の文字列選択肢）は filter 未指定時のみ後方互換で維持する。
      const FilterDef& select_filter = cmd.target_filter;
      const bool has_filter =
          !select_filter.zones.empty() || !select_filter.types.empty() ||
          !select_filter.civilizations.empty() || !select_filter.races.empty() ||
          select_filter.owner.has_value() || select_filter.min_cost.has_value() ||
          select_filter.max_cost.has_value() || select_filter.exact_cost.has_value() ||
          select_filter.min_power.has_value() || select_filter.max_power.has_value() ||
          select_filter.is_tapped.has_value() || select_filter.is_blocker.has_value() ||
          select_filter.is_evolution.has_value() || !select_filter.and_conditions.empty();

      if (!has_filter && !cmd.str_val.empty()) {
        Instruction wait(InstructionOp::WAIT_INPUT);
        wait.args["query_type"] = "SELECT_OPTION";
        nlohmann::json opt_labels = nlohmann::json::array();
        std::istringstream ss(cmd.str_val);
        std::string line;
        while (std::getline(ss, line)) {
          if (!line.empty()) opt_labels.push_back(line);
        }
        int branch_count = (int)cmd.options.size();
        while ((int)opt_labels.size() < branch_count) {
          opt_labels.push_back("Option " + std::to_string(opt_labels.size() + 1));
        }
        wait.args["options"] = opt_labels;
        wait.args["out"] = cmd.output_value_key.empty() ? "$select_result" : cmd.output_value_key;
        out.push_back(wait);
      } else {
        Instruction select_inst(InstructionOp::SELECT);
        nlohmann::json filter_json = select_filter;
        if (cmd.target_group == TargetScope::PLAYER_SELF) {
          filter_json["owner"] = std::string("SELF");
        } else if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
          filter_json["owner"] = std::string("OPPONENT");
        } else if (cmd.target_group == TargetScope::ALL_PLAYERS) {
          filter_json["owner"] = std::string("BOTH");
        }
        select_inst.args["filter"] = filter_json;

        if (!cmd.input_value_key.empty()) {
          std::string in_key = cmd.input_value_key.rfind("$", 0) == 0
                                 ? cmd.input_value_key
                                 : ("$" + cmd.input_value_key);
          select_inst.args["count"] = in_key;
        } else {
          int amount = resolve_amount(cmd, execution_context);
          select_inst.args["count"] = amount > 0 ? amount : 1;
        }

        select_inst.args["out"] =
            cmd.output_value_key.empty() ? "$selected_cards" : cmd.output_value_key;
        out.push_back(select_inst);
      }
    } else {
      // 再発防止: 既知の GET_STAT モードは GET_STAT 命令を生成する。
      //   OPPONENT_ プレフィックス付きは executor 側で相手プレイヤーとして処理。
      static const std::unordered_set<std::string> STAT_MODES = {
        "MANA_COUNT", "CREATURE_COUNT", "SHIELD_COUNT", "HAND_COUNT",
        "GRAVEYARD_COUNT", "BATTLE_ZONE_COUNT", "CARDS_DRAWN_THIS_TURN",
        "OPPONENT_MANA_COUNT", "OPPONENT_CREATURE_COUNT", "OPPONENT_SHIELD_COUNT",
        "OPPONENT_HAND_COUNT", "OPPONENT_GRAVEYARD_COUNT", "OPPONENT_BATTLE_ZONE_COUNT"
      };
      if (STAT_MODES.count(query_type)) {
        Instruction get(InstructionOp::GET_STAT);
        get.args["stat"] = query_type;
        get.args["out"] =
            cmd.output_value_key.empty() ? "$temp" : cmd.output_value_key;
        out.push_back(get);
      } else {
        // SELECT_TARGET など: 実行時対象選択
      // 再発防止: QUERY(SELECT_TARGET, target_filter={zones:["HAND"]}, input_value_key="var_N",
      //   output_value_key="var_out") は実行時対象列挙が必要なため
      //   InstructionOp::SELECT を生成する。
      //   GAME_ACTION("SELECT_TARGET") は生成時解決が必要で単一パイプラインパターンに
      //   適合しない。InstructionOp::SELECT は実行時に対象を列挙し、count=
      //   input_value_key でコンテキスト変数を参照する。
      const FilterDef& select_filter = cmd.target_filter;
      Instruction select_inst(InstructionOp::SELECT);
      // filter を JSON に手動シリアライズ (必要なフィールドのみ)
      nlohmann::json filter_json = nlohmann::json::object();
      if (!select_filter.zones.empty()) {
        filter_json["zones"] = select_filter.zones;
      }
      // 再発防止: CommandDef の target_group が指定されている場合は owner を明示する
      // しないと実行時に両プレイヤーを走査してしまい、対象候補が過剰に生成される。
      if (cmd.target_group == TargetScope::PLAYER_SELF) {
        filter_json["owner"] = std::string("SELF");
      } else if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
        filter_json["owner"] = std::string("OPPONENT");
      } else if (cmd.target_group == TargetScope::ALL_PLAYERS) {
        filter_json["owner"] = std::string("BOTH");
      }
      if (select_filter.owner.has_value()) {
        filter_json["owner"] = *select_filter.owner;
      }
      select_inst.args["filter"] = filter_json;
      if (!cmd.input_value_key.empty()) {
        // count をコンテキスト変数参照として指定（$ プレフィックスは get_context_var フォールバックで対応）
        // 再発防止: resolve_int は $ プレフィックスがある文字列のみコンテキスト参照として処理する。
        //   $ なし文字列を渡すと即座に 0 を返すため、$ を明示的に付加する。
        select_inst.args["count"] = "$" + cmd.input_value_key;
      } else {
        int amount = resolve_amount(cmd, execution_context);
        select_inst.args["count"] = amount;
      }
      select_inst.args["out"] = cmd.output_value_key;
      out.push_back(select_inst);
      }  // end inner else (SELECT_TARGET)
    }  // end outer else (non-GET_STAT query types)

  } else if (cmd.type == core::CommandType::MUTATE) {
    std::vector<int> targets = resolve_targets(state, cmd, source_instance_id,
                                               player_id, execution_context);
    int val = resolve_amount(cmd, execution_context);

    Instruction modify(InstructionOp::MODIFY);
    modify.args["type"] = cmd.mutation_kind;
    modify.args["value"] = val;
    modify.args["str_value"] = cmd.str_param;

    for (int target_id : targets) {
      modify.args["target"] = target_id;
      out.push_back(modify);
    }

  } else if (cmd.type == core::CommandType::SHUFFLE_DECK) {
    Instruction modify(InstructionOp::MODIFY);
    modify.args["type"] = "SHUFFLE";
    modify.args["target"] = "DECK";
    out.push_back(modify);
  }
}

void CommandSystem::generate_macro_instructions(
    std::vector<Instruction> &out, GameState &state, const CommandDef &cmd,
    int source_instance_id, PlayerID player_id,
    std::map<std::string, int> &execution_context) {
  int count = resolve_amount(cmd, execution_context);

  // 再発防止: DRAW は DRAW_CARD の完全エイリアス。両者を同一ブロックで処理する。
  if (cmd.type == core::CommandType::DRAW_CARD ||
      cmd.type == core::CommandType::DRAW) {
    std::string out_key =
        cmd.output_value_key.empty() ? "$draw_choice" : cmd.output_value_key;
    std::string count_val_key = "";

    if (cmd.up_to && count > 0) {
      Instruction select(InstructionOp::WAIT_INPUT);
      select.args["query_type"] = "SELECT_NUMBER";
      select.args["min"] = 0;
      select.args["max"] = count;
      select.args["out"] = out_key;
      out.push_back(select);

      count_val_key = out_key;
    }

    Instruction move(InstructionOp::MOVE);
    move.args["target"] = "DECK_TOP";
    if (!count_val_key.empty()) {
      // 再発防止: コンテキスト変数参照は $ プレフィックスが必要。
      //   resolve_int が "$var_X" 形式のみコンテキスト参照として処理するため。
      //   get_context_var は $ あり/なし両方でフォールバック検索するが、
      //   resolve_int 自体が $ なし文字列を即座に 0 返するため $ を明示する。
      move.args["count"] = "$" + count_val_key;
    } else {
      move.args["count"] = count;
    }
    move.args["to"] = "HAND";
    out.push_back(move);

    // Ensure turn-stat tracking for draws: emit a STAT modify so that
    // CARDS_DRAWN is incremented in a central, undoable way. Use a
    // context reference when the draw amount comes from a $-variable.
    {
      Instruction stat_mod(InstructionOp::MODIFY);
      stat_mod.args["type"] = "STAT";
      stat_mod.args["stat"] = "CARDS_DRAWN";
      if (!count_val_key.empty()) {
        // count chosen by player (SELECT_NUMBER) stored in count_val_key
        stat_mod.args["value"] = std::string("$") + count_val_key;
      } else {
        stat_mod.args["value"] = count;
      }
      out.push_back(stat_mod);
    }

    if (!cmd.output_value_key.empty() && count_val_key.empty()) {
      Instruction calc(InstructionOp::MATH);
      calc.args["lhs"] = count;
      calc.args["op"] = "+";
      calc.args["rhs"] = 0;
      calc.args["out"] = cmd.output_value_key;
      out.push_back(calc);
    }
    return;
  }

  // 再発防止: DESTROY と DISCARD はどちらも対象を墓地へ送る。同一ブロックで処理する。
  //   output_value_key が指定されている場合は破壊/捨てた枚数を格納する。
  if (cmd.type == core::CommandType::DESTROY ||
      cmd.type == core::CommandType::DISCARD) {
    std::vector<int> targets = resolve_targets(state, cmd, source_instance_id,
                                               player_id, execution_context);
    for (int target_id : targets) {
      Instruction move(InstructionOp::MOVE);
      move.args["target"] = target_id;
      move.args["to"] = "GRAVEYARD";
      out.push_back(move);
    }
    if (!cmd.output_value_key.empty()) {
      Instruction calc(InstructionOp::MATH);
      calc.args["lhs"] = (int)targets.size();
      calc.args["op"] = "+";
      calc.args["rhs"] = 0;
      calc.args["out"] = cmd.output_value_key;
      out.push_back(calc);
    }
    return;
  }

  // Fallback: Use Primitive Logic via dispatch if macros map cleanly to
  // Primitives For BOOST_MANA, DISCARD, etc. Re-use primitive generator logic
  // by creating a primitive-like structure? Or just implement them.

  if (cmd.type == core::CommandType::BOOST_MANA ||
      cmd.type == core::CommandType::ADD_MANA) {
    Instruction move(InstructionOp::MOVE);
    move.args["target"] = "DECK_TOP";
    move.args["count"] = count;
    move.args["to"] = "MANA";
    out.push_back(move);

  } else if (cmd.type == core::CommandType::TAP ||
             cmd.type == core::CommandType::UNTAP) {
    std::vector<int> targets = resolve_targets(state, cmd, source_instance_id,
                                               player_id, execution_context);
    std::string type = (cmd.type == core::CommandType::TAP) ? "TAP" : "UNTAP";
    for (int target_id : targets) {
      Instruction mod(InstructionOp::MODIFY);
      mod.args["type"] = type;
      mod.args["target"] = target_id;
      out.push_back(mod);
    }
  } else if (cmd.type == core::CommandType::BREAK_SHIELD) {
    std::vector<int> targets = resolve_targets(state, cmd, source_instance_id,
                                               player_id, execution_context);
    // BREAK_SHIELD involves checking triggers.
    // Use GAME_ACTION "BREAK_SHIELD" which is handled by ShieldSystem
    nlohmann::json args;
    args["type"] = "BREAK_SHIELD";
    args["shields"] = targets;
    args["source_id"] = source_instance_id;

    Instruction inst(InstructionOp::GAME_ACTION, args);
    out.push_back(inst);
  } else if (cmd.type == core::CommandType::SELECT_NUMBER) {
    // 再発防止: SELECT_NUMBER は プレイヤーに数値入力を求める WAIT_INPUT 命令を生成する。
    //   amount が最大値として使われる。output_value_key に結果を格納する。
    std::string out_key =
        cmd.output_value_key.empty() ? "$selected_number" : cmd.output_value_key;
    int max_val = (count > 0) ? count : 6;

    Instruction select(InstructionOp::WAIT_INPUT);
    select.args["query_type"] = "SELECT_NUMBER";
    select.args["min"] = 0;
    select.args["max"] = max_val;
    select.args["out"] = out_key;
    out.push_back(select);

  } else if (cmd.type == core::CommandType::SELECT_OPTION) {
    // 再発防止: SELECT_OPTION は WAIT_INPUT(SELECT_OPTION) でプレイヤーに選択させ
    //   選択されたインデックスを out_key に格納する。
    //   intent_generator が CHOICE コマンドを生成し、dispatch_command が
    //   pipeline.waiting_for_key へ選択インデックスをセットしてパイプラインを再開する。
    //   旧実装の「常に option[0] を実行するフォールバック」は不正確なため削除。
    std::string out_key = cmd.output_value_key.empty() ? "$select_result" : cmd.output_value_key;

    // オプションラベルを str_val（改行区切り）または自動生成で構築
    nlohmann::json opt_labels = nlohmann::json::array();
    if (!cmd.str_val.empty()) {
      std::istringstream ss(cmd.str_val);
      std::string line;
      while (std::getline(ss, line)) {
        if (!line.empty()) opt_labels.push_back(line);
      }
    }
    int branch_count = (int)cmd.options.size();
    while ((int)opt_labels.size() < branch_count) {
      opt_labels.push_back("Option " + std::to_string(opt_labels.size() + 1));
    }

    Instruction wait(InstructionOp::WAIT_INPUT);
    wait.args["query_type"] = "SELECT_OPTION";
    wait.args["options"] = opt_labels;
    wait.args["out"] = out_key;
    out.push_back(wait);

    // 各ブランチを IF(選択インデックス == i) でガードして順次追加
    // 再発防止: check_condition は "var_eq" 型を処理しないため、
    //   現時点では pipeline_executor 側で IF 命令の cond を extend 拡張する必要がある。
    //   未実装の場合でも WAIT_INPUT で正しく一時停止し AI が CHOICE を送信できる。
    for (int opt_idx = 0; opt_idx < branch_count; ++opt_idx) {
      if (cmd.options[opt_idx].empty()) continue;
      // sub-instructions for this branch
      std::vector<Instruction> branch_insts;
      for (const auto &child_cmd : cmd.options[opt_idx]) {
        auto sub = generate_instructions(state, child_cmd, source_instance_id,
                                         player_id, execution_context);
        branch_insts.insert(branch_insts.end(), sub.begin(), sub.end());
      }
      // IF 命令で選択インデックスと比較してガード
      // 再発防止: check_condition の "op" 形式（lhs/$var == rhs/int）を使用する
      //   出力変数キーに "$" がないとき resolve_int がコンテキスト参照しないので "$" を付加する
      Instruction if_inst(InstructionOp::IF);
      nlohmann::json cond;
      cond["op"] = "==";
      // $ プレフィックスがなければ付加して context lookup が機能するようにする
      std::string lhs_var = (out_key.rfind("$", 0) == 0) ? out_key : ("$" + out_key);
      cond["lhs"] = lhs_var;
      cond["rhs"] = opt_idx;
      if_inst.args["cond"] = cond;
      if_inst.then_block = branch_insts;
      out.push_back(if_inst);
    }

  } else if (cmd.type == core::CommandType::CAST_SPELL) {
    // 再発防止: CAST_SPELL は手札/墓地から呪文を無料または通常コストで唱える。
    //   target_filter で対象呪文を絞り込み、GAME_ACTION(CAST_SPELL) で実行。
    nlohmann::json args;
    args["type"] = "CAST_SPELL";
    args["source_id"] = source_instance_id;
    args["owner_id"] = static_cast<int>(player_id);
    // フィルター情報をシリアライズ（TargetUtils で解決）
    if (!cmd.target_filter.zones.empty()) {
      args["source_zone"] = cmd.target_filter.zones[0];
    }
    if (cmd.target_filter.max_cost.has_value()) {
      args["max_cost"] = *cmd.target_filter.max_cost;
    }

    Instruction inst(InstructionOp::GAME_ACTION, args);
    out.push_back(inst);

  } else if (cmd.type == core::CommandType::APPLY_MODIFIER ||
             cmd.type == core::CommandType::GRANT_KEYWORD ||
             cmd.type == core::CommandType::ADD_RESTRICTION) {
    // 再発防止: APPLY_MODIFIER/GRANT_KEYWORD/ADD_RESTRICTION はすべて「対象ごとにMODIFY命令を生成」する
    //   同じパターンで処理し、type文字列のみ分岐する。
    //   APPLY_MODIFIER/GRANT_KEYWORD: mod_type = "ADD_KEYWORD"
    //   ADD_RESTRICTION:              mod_type = "ADD_RESTRICTION"
    const std::string mod_type =
        (cmd.type == core::CommandType::ADD_RESTRICTION) ? "ADD_RESTRICTION"
                                                         : "ADD_KEYWORD";
    std::vector<int> targets = resolve_targets(state, cmd, source_instance_id,
                                               player_id, execution_context);
    if (!cmd.str_param.empty()) {
      for (int target_id : targets) {
        Instruction mod(InstructionOp::MODIFY);
        mod.args["type"] = mod_type;
        mod.args["target"] = target_id;
        mod.args["str_value"] = cmd.str_param;
        out.push_back(mod);
      }
    }

  } else if (cmd.type == core::CommandType::LOCK_SPELL ||
             cmd.type == core::CommandType::SPELL_RESTRICTION ||
             cmd.type == core::CommandType::CANNOT_PUT_CREATURE ||
             cmd.type == core::CommandType::CANNOT_SUMMON_CREATURE ||
             cmd.type == core::CommandType::PLAYER_CANNOT_ATTACK ||
             cmd.type == core::CommandType::IGNORE_ABILITY) {
    // 再発防止: 制限コマンドは ADD_PASSIVE 命令でパッシブ効果を登録する。
    //   duration 文字列（"THIS_TURN" 等）を整数ターン数に変換する。
    //   target_group から対象プレイヤーを自動設定。
    auto dur_str_to_int = [](const std::string& d) -> int {
      if (d == "PERMANENT") return -1;
      if (d == "UNTIL_END_OF_OPPONENT_TURN" || d == "UNTIL_START_OF_OPPONENT_TURN") return 2;
      if (d == "DURING_OPPONENT_TURN") return 1;
      return 1; // THIS_TURN およびその他は 1 ターン
    };
    int dur = dur_str_to_int(cmd.duration);

    // target_group から filter.owner を設定する
    // 再発防止: TargetScope に ALL は存在しない。ALL_PLAYERS を使用すること。
    std::string owner_str = "OPPONENT"; // デフォルト: 相手を対象
    if (cmd.target_group == core::TargetScope::PLAYER_SELF || cmd.target_group == core::TargetScope::SELF)
      owner_str = "SELF";
    else if (cmd.target_group == core::TargetScope::ALL_PLAYERS)
      owner_str = "BOTH";

    Instruction mod(InstructionOp::MODIFY);
    mod.args["duration"] = dur;
    mod.args["type"] = "ADD_PASSIVE";

    if (cmd.type == core::CommandType::LOCK_SPELL) {
      mod.args["str_value"] = "LOCK_SPELL";
      mod.args["filter"]["owner"] = owner_str;
      mod.args["filter"]["types"] = nlohmann::json::array({"SPELL"});
    } else if (cmd.type == core::CommandType::SPELL_RESTRICTION) {
      mod.args["str_value"] = "LOCK_SPELL";
      mod.args["filter"]["owner"] = owner_str;
      mod.args["filter"]["types"] = nlohmann::json::array({"SPELL"});
      // target_filter のコスト指定を引き継ぐ
      if (cmd.target_filter.exact_cost.has_value())
        mod.args["filter"]["exact_cost"] = *cmd.target_filter.exact_cost;
      if (cmd.target_filter.min_cost.has_value())
        mod.args["filter"]["min_cost"] = *cmd.target_filter.min_cost;
      if (cmd.target_filter.max_cost.has_value())
        mod.args["filter"]["max_cost"] = *cmd.target_filter.max_cost;
      if (!cmd.input_value_key.empty()) {
        std::string in_key = cmd.input_value_key.rfind("$", 0) == 0
                               ? cmd.input_value_key
                               : ("$" + cmd.input_value_key);
        mod.args["filter"]["cost_ref"] = in_key;
      }
    } else if (cmd.type == core::CommandType::CANNOT_PUT_CREATURE ||
               cmd.type == core::CommandType::CANNOT_SUMMON_CREATURE) {
      mod.args["str_value"] = "CANNOT_SUMMON";
      mod.args["filter"]["owner"] = owner_str;
      mod.args["filter"]["types"] = nlohmann::json::array({"CREATURE"});
    } else if (cmd.type == core::CommandType::PLAYER_CANNOT_ATTACK) {
      mod.args["str_value"] = "CANNOT_ATTACK";
      mod.args["filter"]["owner"] = owner_str;
      mod.args["filter"]["types"] = nlohmann::json::array({"CREATURE"});
    } else if (cmd.type == core::CommandType::IGNORE_ABILITY) {
      mod.args["str_value"] = "IGNORE_ABILITY";
      mod.args["filter"] = cmd.target_filter;
      mod.args["filter"]["owner"] = owner_str;
      if (!cmd.input_value_key.empty()) {
        std::string in_key = cmd.input_value_key.rfind("$", 0) == 0
                               ? cmd.input_value_key
                               : ("$" + cmd.input_value_key);
        mod.args["filter"]["cost_ref"] = in_key;
      }
    }

    out.push_back(mod);

  } else if (cmd.type == core::CommandType::PUT_CREATURE) {
    // 再発防止: PUT_CREATURE はクリーチャーを場に直接出す（コスト支払なし）。
    //   SUMMON_TOKEN 的な動作: str_param でカードID/トークン名を指定する。
    nlohmann::json args;
    args["type"] = "SUMMON_CREATURE";
    args["card_id"] = cmd.str_param;
    args["owner_id"] = static_cast<int>(player_id);
    args["source_id"] = source_instance_id;

    Instruction inst(InstructionOp::GAME_ACTION, args);
    out.push_back(inst);

  } else if (cmd.type == core::CommandType::REPLACE_CARD_MOVE) {
    // 再発防止: REPLACE_CARD_MOVE はカード移動の置換効果。
    //   from_zone のかわりに to_zone にカードを送る（例: 墓地へ行くかわりに手札へ）。
    //   target_filter で対象カードを絞り込む。
    std::vector<int> targets = resolve_targets(state, cmd, source_instance_id,
                                               player_id, execution_context);
    for (int target_id : targets) {
      Instruction move(InstructionOp::MOVE);
      move.args["target"] = target_id;
      move.args["to"] = cmd.to_zone.empty() ? "HAND" : cmd.to_zone;
      out.push_back(move);
    }
  } else if (cmd.type == core::CommandType::REPLACE_MOVE_CARD) {
    // 再発防止: REPLACE_MOVE_CARD はマグナム系置換効果。
    //   相手カードの移動先を墓地に変更する。
    //   INSTALL_REPLACEMENT_EFFECT 指示を生成し、Pipeline がゲームステートに登録する。
    nlohmann::json args;
    args["type"] = "REPLACE_MOVE_CARD";
    args["source_id"] = source_instance_id;
    args["owner_id"] = static_cast<int>(player_id);
    Instruction inst(InstructionOp::GAME_ACTION, args);
    out.push_back(inst);
  } else if (cmd.type == core::CommandType::REGISTER_DELAYED_EFFECT) {
    // Map REGISTER_DELAYED_EFFECT to a GAME_ACTION so the pipeline can register
    // the delayed effect into the game systems. Include minimal metadata.
    nlohmann::json args;
    args["type"] = "REGISTER_DELAYED_EFFECT";
    args["source_id"] = source_instance_id;
    args["owner_id"] = static_cast<int>(player_id);
    if (!cmd.str_param.empty()) {
      args["effect_key"] = cmd.str_param;
    }
    // amount can represent duration/count for the delayed effect
    if (count > 0) {
      args["amount"] = count;
    }
    // Serialize simple filter information if present
    if (!cmd.target_filter.zones.empty()) {
      nlohmann::json f = nlohmann::json::object();
      f["zones"] = cmd.target_filter.zones;
      if (cmd.target_filter.owner.has_value()) f["owner"] = *cmd.target_filter.owner;
      args["filter"] = f;
    }
    Instruction inst_delay(InstructionOp::GAME_ACTION, args);
    out.push_back(inst_delay);
  } else if (cmd.type == core::CommandType::LOOK_TO_BUFFER ||
             cmd.type == core::CommandType::REVEAL_TO_BUFFER) {
    // 再発防止: LOOK_TO_BUFFER / REVEAL_TO_BUFFER はどちらもデッキ上から指定枚数をバッファへ展開する。
    //   LOOK は手番プレイヤーのみ閲覧、REVEAL は全員に公開するという意味の違いだが
    //   エンジン実装はどちらも同じ MOVE(DECK_TOP → BUFFER) で処理する。
    //   ※後続の MOVE_BUFFER_TO_ZONE で各 to_zone への振り分けを行う。
    int reveal_count = (count > 0) ? count : 4;
    Instruction move(InstructionOp::MOVE);
    move.args["target"] = "DECK_TOP";
    move.args["count"] = reveal_count;
    move.args["to"] = "BUFFER";
    out.push_back(move);
    if (!cmd.output_value_key.empty()) {
      Instruction calc(InstructionOp::MATH);
      calc.args["lhs"] = reveal_count;
      calc.args["op"] = "+";
      calc.args["rhs"] = 0;
      calc.args["out"] = cmd.output_value_key;
      out.push_back(calc);
    }
  } else if (cmd.type == core::CommandType::SELECT_FROM_BUFFER) {
    // 再発防止: SELECT_FROM_BUFFER はバッファからプレイヤーに選択させる。
    std::string out_key =
        cmd.output_value_key.empty() ? "$buffer_select" : cmd.output_value_key;
    Instruction wait(InstructionOp::WAIT_INPUT);
    wait.args["query_type"] = "SELECT_FROM_BUFFER";
    wait.args["count"] = (count > 0) ? count : 1;
    wait.args["out"] = out_key;
    out.push_back(wait);

  } else if (cmd.type == core::CommandType::MOVE_BUFFER_TO_ZONE) {
    // 再発防止: MOVE_BUFFER_TO_ZONE は amount と target_filter の組み合わせで3パターン動作。
    auto &f = cmd.target_filter;
    bool has_filter = !f.civilizations.empty() || !f.types.empty() ||
                      !f.races.empty() || !f.zones.empty() ||
                      f.min_cost.has_value() || f.max_cost.has_value() ||
                      f.exact_cost.has_value();

    if (has_filter) {
      nlohmann::json auto_args;
      auto_args["type"] = "AUTO_SELECT_BUFFER";
      nlohmann::json filter_json;
      filter_json["civilizations"] = f.civilizations;
      filter_json["types"] = f.types;
      filter_json["races"] = f.races;
      if (f.min_cost.has_value())
        filter_json["min_cost"] = *f.min_cost;
      if (f.max_cost.has_value())
        filter_json["max_cost"] = *f.max_cost;
      if (f.exact_cost.has_value())
        filter_json["exact_cost"] = *f.exact_cost;
      auto_args["filter"] = filter_json;
      auto_args["out"] = "$buffer_select";
      out.push_back(Instruction(InstructionOp::GAME_ACTION, auto_args));

      Instruction move(InstructionOp::MOVE);
      move.args["target"] = "$buffer_select";
      move.args["to"] = cmd.to_zone.empty() ? "HAND" : cmd.to_zone;
      out.push_back(move);

      Instruction ret(InstructionOp::MOVE);
      ret.args["target"] = "BUFFER_REMAIN";
      ret.args["to"] = "DECK_BOTTOM";
      out.push_back(ret);

    } else if (count > 0) {
      std::string sel_key = "$buffer_select";
      Instruction wait(InstructionOp::WAIT_INPUT);
      wait.args["query_type"] = "SELECT_FROM_BUFFER";
      wait.args["count"] = count;
      wait.args["out"] = sel_key;
      out.push_back(wait);

      Instruction move(InstructionOp::MOVE);
      move.args["target"] = sel_key;
      move.args["to"] = cmd.to_zone.empty() ? "HAND" : cmd.to_zone;
      out.push_back(move);

    } else {
      Instruction move(InstructionOp::MOVE);
      move.args["target"] = "$buffer_select";
      move.args["to"] = cmd.to_zone.empty() ? "HAND" : cmd.to_zone;
      out.push_back(move);

      Instruction ret(InstructionOp::MOVE);
      ret.args["target"] = "BUFFER_REMAIN";
      ret.args["to"] = "DECK_BOTTOM";
      out.push_back(ret);
    }

  } else if (cmd.type == core::CommandType::MOVE_BUFFER_REMAIN_TO_ZONE) {
    // 再発防止: バッファ残余カード（$buffer_select に含まれない）をすべて指定ゾーンへ移動。
    Instruction remain(InstructionOp::MOVE);
    remain.args["target"] = "BUFFER_REMAIN";
    remain.args["to"] = cmd.to_zone.empty() ? "DECK_BOTTOM" : cmd.to_zone;
    out.push_back(remain);
  }
}

std::vector<int>
CommandSystem::resolve_targets(GameState &state, const CommandDef &cmd,
                               int source_instance_id, PlayerID player_id,
                               std::map<std::string, int> &execution_context) {
  // ... (Same implementation as before)
  std::vector<int> targets;

    // 再発防止: input_value_key が設定されている場合は execution_context から
    //   カードインスタンスIDを直接取得して対象として使用する。
    //   TAP/UNTAP などで SELECT 結果を入力として受け取る際に使用。
    if (!cmd.input_value_key.empty()) {
      auto it = execution_context.find(cmd.input_value_key);
      if (it != execution_context.end() && it->second > 0) {
        targets.push_back(it->second);
        return targets;
      }
    }
  std::vector<PlayerID> players_to_check;

  if (cmd.target_group == TargetScope::PLAYER_SELF) {
    players_to_check.push_back(player_id);
  } else if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
    players_to_check.push_back(1 - player_id);
  } else if (cmd.target_group == TargetScope::ALL_PLAYERS) {
    players_to_check.push_back(player_id);
    players_to_check.push_back(1 - player_id);
  } else if (cmd.target_group == TargetScope::SELF) {
    players_to_check.push_back(player_id);
  }

  const auto &filter = cmd.target_filter;
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  for (PlayerID pid : players_to_check) {
    if (filter.zones.empty()) {
      if (cmd.target_group == TargetScope::SELF && source_instance_id != -1) {
        CardInstance *inst = state.get_card_instance(source_instance_id);
        if (inst && card_db.find(inst->card_id) != card_db.end()) {
          if (dm::engine::utils::TargetUtils::is_valid_target(
                  *inst, card_db.at(inst->card_id), filter, state, player_id,
                  pid, false, &execution_context)) {
            targets.push_back(source_instance_id);
          }
        }
      }
      continue;
    }

    for (const std::string &zone_str : filter.zones) {
      Zone zone_enum = parse_zone_string(zone_str);

      const std::vector<CardInstance> *container = nullptr;
      switch (zone_enum) {
      case Zone::HAND:
        container = &state.players[pid].hand;
        break;
      case Zone::MANA:
        container = &state.players[pid].mana_zone;
        break;
      case Zone::BATTLE:
        container = &state.players[pid].battle_zone;
        break;
      case Zone::GRAVEYARD:
        container = &state.players[pid].graveyard;
        break;
      case Zone::SHIELD:
        container = &state.players[pid].shield_zone;
        break;
      case Zone::DECK:
        container = &state.players[pid].deck;
        break;
      default:
        break;
      }

      if (container) {
        for (const auto &card : *container) {
          if (card_db.find(card.card_id) != card_db.end()) {
            const auto &def = card_db.at(card.card_id);
            if (dm::engine::utils::TargetUtils::is_valid_target(
                    card, def, filter, state, player_id, pid, false,
                    &execution_context)) {
              targets.push_back(card.instance_id);
            }
          }
        }
      }
    }
  }

  if (filter.count.has_value()) {
    int n = filter.count.value();
    if (targets.size() > (size_t)n) {
      targets.resize(n);
    }
  }

  return targets;
}

} // namespace dm::engine::systems
