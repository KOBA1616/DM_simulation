#include "intent_generator.hpp"
#include "core/constants.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> IntentGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        // Helper to dump actions for debugging
        auto dump_actions = [&](const std::vector<Action>& actions, const std::string& context) {
            try {
                std::filesystem::create_directories("logs");
                std::ofstream ofs("logs/intent_actions.txt", std::ios::app);
                if (!ofs) return;
                std::ostringstream ss;
                ss << "[IntentActions] context=" << context << " player=" << static_cast<int>(game_state.active_player_id)
                   << " phase=" << static_cast<int>(game_state.current_phase) << " count=" << actions.size() << "\n";
                for (size_t i = 0; i < actions.size(); ++i) {
                    const auto& a = actions[i];
                    ss << "  #" << i << " type=" << static_cast<int>(a.type)
                       << " card_id=" << a.card_id
                       << " src_iid=" << a.source_instance_id
                       << " tgt_iid=" << a.target_instance_id
                       << " tgt_player=" << static_cast<int>(a.target_player)
                       << " slot=" << a.slot_index << " tgt_slot=" << a.target_slot_index << "\n";
                }
                ofs << ss.str();
            } catch (...) {
                // Best-effort logging; swallow errors to not affect engine flow
            }
        };

        // Phase 6: Handle Waiting for User Input (Query Response)
        if (game_state.waiting_for_user_input && game_state.waiting_for_user_input) {
            std::vector<Action> actions;
            const auto& query = game_state.pending_query;

            if (query.query_type == "SELECT_TARGET") {
                for (int target_id : query.valid_targets) {
                    Action act;
                    act.type = PlayerIntent::SELECT_TARGET;
                    act.target_instance_id = target_id;
                    // We might need to encode query_id to ensure we are answering the right query,
                    // but Action struct is limited. The engine state implies the context.
                    actions.push_back(act);
                }
            }
            else if (query.query_type == "SELECT_OPTION") {
                for (size_t i = 0; i < query.options.size(); ++i) {
                    Action act;
                    act.type = PlayerIntent::SELECT_OPTION;
                    act.target_slot_index = static_cast<int>(i);
                    // Optionally store string value in a future Action extension
                    actions.push_back(act);
                }
            }

            dump_actions(actions, "waiting_for_user_input");
            return actions;
        }

        // Prepare Context
        ActionGenContext ctx = { game_state, card_db, game_state.active_player_id };

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
                    // No actions logic required for these phases in current design, just PASS
                    std::vector<Action> actions;
                    Action pass;
                    pass.type = PlayerIntent::PASS;
                    actions.push_back(pass);
                    dump_actions(actions, "pass_phase");
                    return actions;
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
                    std::vector<Action> empty;
                    dump_actions(empty, "default_empty");
                    return {};
                }
        }
    }

}
