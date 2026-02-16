#include "intent_generator.hpp"
#include "core/constants.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace dm::engine {

    using namespace dm::core;

    std::vector<CommandDef> IntentGenerator::generate_legal_commands(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        try {
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
            }

            dump_actions(actions, "waiting_for_user_input");
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
