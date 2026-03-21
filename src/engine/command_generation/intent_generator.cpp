#include "intent_generator.hpp"
#include "core/constants.hpp"
#include "engine/systems/effects/reaction_window.hpp"
#include "engine/systems/effects/continuous_effect_system.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace dm::engine {

    using namespace dm::core;

    std::vector<CommandDef> IntentGenerator::generate_legal_commands(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        try {
            // Ensure continuous effects (PASSIVE/STATIC) are up-to-date before
            // generating legal commands. This enforces the contract that any
            // state change which could affect costs or targets has had
            // ContinuousEffectSystem::recalculate run.
            dm::engine::systems::ContinuousEffectSystem::recalculate(game_state, card_db);

            std::filesystem::create_directories("logs");
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "INTENT_GEN_CMD entry player=" << static_cast<int>(game_state.active_player_id)
                     << " phase=" << static_cast<int>(game_state.current_phase) << "\n";
                diag.close();
            }
        } catch(...) {}

        auto dump_actions = [&](const std::vector<CommandDef>& actions, const std::string& context) {
            try {
                std::filesystem::create_directories("logs");
                std::ofstream ofs("logs/intent_actions.txt", std::ios::app);
                if (!ofs) return;
                std::ostringstream ss;
                ss << "[IntentCommands] context=" << context << " player=" << static_cast<int>(game_state.active_player_id)
                   << " phase=" << static_cast<int>(game_state.current_phase) << " count=" << actions.size() << "\n";
                for (size_t i = 0; i < actions.size(); ++i) {
                    const auto& a = actions[i];
                    ss << "  #" << i << " type=" << static_cast<int>(a.type)
                       << " inst=" << a.instance_id
                       << " tgt=" << a.target_instance
                       << " amt=" << a.amount << "\n";
                }
                ofs << ss.str();
            } catch (...) {}
        };

        // Phase 6: Handle Waiting for User Input (Query Response)
        if (game_state.waiting_for_user_input) {
            std::vector<CommandDef> actions;
            const auto& query = game_state.pending_query;

            if (query.query_type == "SELECT_TARGET") {
                for (int target_id : query.valid_targets) {
                    CommandDef cmd;
                    cmd.type = CommandType::SELECT_TARGET;
                    cmd.instance_id = target_id;
                    actions.push_back(cmd);
                }
            }
            else if (query.query_type == "SELECT_OPTION") {
                for (size_t i = 0; i < query.options.size(); ++i) {
                    CommandDef cmd;
                    cmd.type = CommandType::CHOICE;
                    cmd.target_instance = static_cast<int>(i);
                    actions.push_back(cmd);
                }
            } else if (query.query_type == "SELECT_NUMBER") {
                // Assuming min/max params in query
                int min_val = 0;
                int max_val = 0;
                if (query.params.count("min")) min_val = query.params.at("min");
                if (query.params.count("max")) max_val = query.params.at("max");

                for (int val = min_val; val <= max_val; ++val) {
                    CommandDef cmd;
                    cmd.type = CommandType::SELECT_NUMBER;
                    cmd.target_instance = val;
                    actions.push_back(cmd);
                }
            // NOTE: 再発防止 — SELECT_NUMBER の for ループ後に閉じ括弧が必要。
            // ここでブロックを閉じないと else if が dangling else になりコンパイルエラー。
            } else if (query.query_type == "SELECT_FROM_BUFFER") {
                // 再発防止: SELECT_FROM_BUFFER のケース
                // - 目的: バッファ内カードをユーザー/AI が選べるように単純な選択コマンドを生成する
                // - 出力キーは一意の既定値を用いて後段が暗黙キーを期待しないようにする
                // - owner_id を明示して「どのプレイヤーの選択か」を確定させる
                // - バッファが空のときは PASS を返すことで安全にパイプラインを進める
                const auto& buf = game_state.players[game_state.active_player_id].effect_buffer;
                // Use the canonical context key used by CommandSystem/MOVE_BUFFER_TO_ZONE
                // CommandSystem historically expects "$buffer_select" as the out key
                const std::string default_out_key = "$buffer_select"; // 統一キー
                const int owner = static_cast<int>(game_state.active_player_id);

                for (const auto& card : buf) {
                    CommandDef cmd;
                    cmd.type = CommandType::SELECT_FROM_BUFFER;
                    // instance_id は選択対象のカードインスタンスID
                    cmd.instance_id = card.instance_id;
                    // 明示的に owner をつけて、どのプレイヤーのバッファかを示す
                    cmd.owner_id = owner;
                    // 後段はこのキーで選択結果（vector<int> を想定）を受け取る
                    cmd.output_value_key = default_out_key;
                    actions.push_back(cmd);
                }
                if (actions.empty()) {
                    // バッファが空の場合は PASS（安全フォールバック）。
                    // 注意: PASS は空バッファの正常経路であり、エラー隠蔽にならないよう
                    //       呼び出し元のログ/テストで検出できるようにすること。
                    CommandDef pass_cmd;
                    pass_cmd.type = CommandType::PASS;
                    actions.push_back(pass_cmd);
                }
            }

            dump_actions(actions, "waiting_for_user_input");
            return actions;
        }

        // 再発防止: WAITING_FOR_REACTION 中は通常フェーズ戦略へ進まず、
        // reaction_stack から専用の合法手を返す。ここを通さないと革命チェンジが
        // USE_ABILITY として提示されず、AI が反応を宣言できない。
        if (game_state.status == GameState::Status::WAITING_FOR_REACTION &&
            !game_state.reaction_stack.empty()) {
            std::vector<CommandDef> actions;
            const auto& window = game_state.reaction_stack.back();

            for (const auto& candidate : window.candidates) {
                if (candidate.player_id != game_state.active_player_id) {
                    continue;
                }

                if (candidate.type == dm::engine::systems::ReactionType::REVOLUTION_CHANGE) {
                    CommandDef use;
                    use.type = CommandType::USE_ABILITY;
                    use.instance_id = candidate.instance_id;
                    use.target_instance = game_state.current_attack.source_instance_id;
                    actions.push_back(use);
                } else if (candidate.type == dm::engine::systems::ReactionType::SHIELD_TRIGGER) {
                    CommandDef st;
                    st.type = CommandType::SHIELD_TRIGGER;
                    st.instance_id = candidate.instance_id;
                    actions.push_back(st);
                }
            }

            CommandDef pass;
            pass.type = CommandType::PASS;
            actions.push_back(pass);

            dump_actions(actions, "waiting_for_reaction");
            return actions;
        }

        // Prepare Context
        CommandGenContext ctx = { game_state, card_db, game_state.active_player_id };

        // 1. Pending Effects
        if (!game_state.pending_effects.empty()) {
            PendingEffectStrategy pending_strategy;
            auto res = pending_strategy.generate(ctx);
            dump_actions(res, "pending_effects");
            return res;
        }

        // 2. Stack (Atomic Action Flow)
        {
            StackStrategy stack_strategy;
            auto stack_actions = stack_strategy.generate(ctx);
            if (!stack_actions.empty()) {
                dump_actions(stack_actions, "stack_actions");
                return stack_actions;
            }
        }

        // 3. Phase Specific Strategies
        if (game_state.current_phase == Phase::BLOCK) {
            BlockPhaseStrategy block_strategy;
            auto res = block_strategy.generate(ctx);
            dump_actions(res, "block_phase");
            return res;
        }

        switch (game_state.current_phase) {
            case Phase::START_OF_TURN:
            case Phase::DRAW:
                {
                    std::vector<CommandDef> empty;
                    dump_actions(empty, "auto_advance_phase");
                    return empty;
                }
            case Phase::MANA:
                {
                    ManaPhaseStrategy mana_strategy;
                    auto res = mana_strategy.generate(ctx);
                    dump_actions(res, "mana_phase");
                    return res;
                }
            case Phase::MAIN:
                {
                    MainPhaseStrategy main_strategy;
                    auto res = main_strategy.generate(ctx);
                    dump_actions(res, "main_phase");
                    return res;
                }
            case Phase::ATTACK:
                {
                    AttackPhaseStrategy attack_strategy;
                    auto res = attack_strategy.generate(ctx);
                    dump_actions(res, "attack_phase");
                    return res;
                }
            default:
                {
                    std::vector<CommandDef> empty;
                    dump_actions(empty, "default_empty");
                    return {};
                }
        }
    }

}
