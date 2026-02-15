#include "game_instance.hpp"
#include "systems/flow/phase_manager.hpp"
#include "engine/actions/intent_generator.hpp"
#include "engine/ai/simple_ai.hpp" // Added for SimpleAI
#include "engine/game_command/game_command.hpp"
#include "engine/game_command/action_commands.hpp" // Added
#include "engine/game_command/commands.hpp" // Added for DeclareReactionCommand
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/continuous_effect_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/card/effect_system.hpp" // Added for EffectSystem
#include "engine/systems/command_system.hpp" // Added for CommandSystem
#include "diag_win32.h"
#include <functional>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace dm::engine {

    using namespace dm::core;
    // using namespace dm::engine::game_command; // Removed to avoid CommandType ambiguity

    GameInstance::GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db)
        : state(seed), card_db(db) {
        std::ofstream diag("c:\\temp\\game_instance_constructor.txt", std::ios::app);
        if (diag) {
            diag << "GameInstance constructor called with seed=" << seed << std::endl;
            diag.close();
        }
        initial_seed_ = seed;
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Refactored: Use TriggerManager to setup event handling logic
        systems::TriggerManager::setup_event_handling(state, trigger_manager, card_db);
    }

    GameInstance::GameInstance(uint32_t seed)
        : state(seed), card_db(CardRegistry::get_all_definitions_ptr()) {
        initial_seed_ = seed;
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Refactored: Use TriggerManager to setup event handling logic
        systems::TriggerManager::setup_event_handling(state, trigger_manager, card_db);
    }

    GameInstance::~GameInstance() {
        // Use low-level Win32 write only to avoid CRT/heap activity during destructor
        try { diag_write_win32(std::string("GameInstance::~GameInstance seed=") + std::to_string(initial_seed_)); } catch(...) {}
    }

    void GameInstance::start_game() {
        if (card_db) {
            PhaseManager::start_game(state, *card_db);
        }
    }

    bool GameInstance::step() {
        // Check if game is over
        if (state.game_over) {
            std::cout << "[step] Game is over, returning false\n";
            return false;
        }

        // Check if current player is human - skip auto-step
        PlayerID active_pid = state.active_player_id;
        if (state.is_human_player(active_pid)) {
            std::cout << "[step] Human player turn (P" << (int)active_pid << "), returning false\n";
            return false;
        }

        // Generate legal actions
        auto actions = IntentGenerator::generate_legal_commands(state, *card_db);
        std::cout << "[step] Generated " << actions.size() << " commands at Turn " << state.turn_number
                  << ", Phase " << static_cast<int>(state.current_phase) << ", Player " << state.active_player_id << "\n";
        
        // Log action types (first 5)
        for (size_t i = 0; i < std::min(size_t(5), actions.size()); ++i) {
            std::cout << "  Command " << i << ": type=" << static_cast<int>(actions[i].type) << "\n";
        }
        if (actions.size() > 5) {
            std::cout << "  ... and " << (actions.size() - 5) << " more\n";
        }

        if (actions.empty()) {
            std::cout << "[step] No actions, calling fast_forward...\n";
            // No actions available - call fast_forward to progress to next decision point
            PhaseManager::fast_forward(state, *card_db);
            std::cout << "[step] After fast_forward: Turn " << state.turn_number 
                      << ", Phase " << static_cast<int>(state.current_phase) << "\n";
            
            // Check if we're stuck (still no actions after fast_forward)
            actions = IntentGenerator::generate_legal_commands(state, *card_db);
            std::cout << "[step] After fast_forward, re-generated " << actions.size() << " commands at Turn "
                      << state.turn_number << ", Phase " << static_cast<int>(state.current_phase) 
                      << ", Player " << state.active_player_id << "\n";
            
            // Log action types (first 5)
            for (size_t i = 0; i < std::min(size_t(5), actions.size()); ++i) {
                std::cout << "  Command " << i << ": type=" << static_cast<int>(actions[i].type) << "\n";
            }
            if (actions.size() > 5) {
                std::cout << "  ... and " << (actions.size() - 5) << " more\n";
            }
            
            if (actions.empty()) {
                std::cout << "[step] Still no actions after fast_forward, game stuck\n";
                // Game might be over or stuck
                return false;
            }
        }

        // Use SimpleAI to select action based on priority
        auto selected_idx = ai::SimpleAI::select_action(actions, state);
        
        if (selected_idx.has_value()) {
            const auto& selected = actions[*selected_idx];
            std::cout << "[step] Executing command type " << static_cast<int>(selected.type) << "\n";
            resolve_command(selected);
            return true;
        }

        std::cout << "[step] No action selected, returning false\n";
        return false;
    }

    void GameInstance::resolve_command(const core::CommandDef& cmd_def) {
        using namespace dm::engine::game_command;
        // Delegate to GameLogicSystem for new CommandDef-based handling
        
        if (!pipeline) {
            std::cerr << "FATAL: pipeline is null!" << std::endl;
            return;
        }
        if (!card_db) {
             std::cerr << "FATAL: card_db is null!" << std::endl;
             return;
        }
        // state.active_pipeline = pipeline; // Removed as GameState is stateless

        // Guard: prevent repeated re-entry for the same resolve action signature.
        auto make_sig = [&](const core::CommandDef& c)->uint64_t {
            uint64_t sig = 0;
            sig |= (uint64_t)state.turn_number << 32;
            sig |= (uint64_t)((int)c.type & 0xFFFF) << 16;
            sig |= (uint64_t)((uint32_t)c.instance_id & 0xFFFF);
            return sig;
        };

        const size_t HARD_MAX = 150;
        bool inserted_sig = false;
        uint64_t sig = make_sig(cmd_def);

        try {
            if (pipeline->call_stack.size() > 0) {
                if (cmd_def.type == core::CommandType::RESOLVE_PLAY) {
                    // If we are already resolving the same signature, skip to avoid runaway loops
                    if (resolving_action_sigs.count(sig)) {
                        std::filesystem::create_directories("logs");
                        std::string trace_file = "logs/game_trace_" + std::to_string(initial_seed_) + ".jsonl";
                        std::ofstream lout(trace_file, std::ios::app);
                        if (lout) {
                            lout << "{\"event\":\"resolve_action_skipped_repeat\",";
                            lout << "\"turn\":" << state.turn_number << ",";
                            lout << "\"phase\":" << static_cast<int>(state.current_phase) << ",";
                            lout << "\"reason\":\"reentrant_signature\",";
                            lout << "\"call_stack_size\":" << pipeline->call_stack.size() << "}" << std::endl;
                            lout.close();
                        }
                        return;
                    }
                }

                if (pipeline->call_stack.size() > HARD_MAX) {
                    std::filesystem::create_directories("logs");
                    std::string trace_file = "logs/game_trace_" + std::to_string(initial_seed_) + ".jsonl";
                    std::ofstream lout(trace_file, std::ios::app);
                    if (lout) {
                        lout << "{\"event\":\"resolve_action_aborted\",";
                        lout << "\"turn\":" << state.turn_number << ",";
                        lout << "\"phase\":" << static_cast<int>(state.current_phase) << ",";
                        lout << "\"reason\":\"call_stack_overflow\",";
                        lout << "\"call_stack_size\":" << pipeline->call_stack.size() << "}" << std::endl;
                        lout.close();
                    }
                    return;
                }
            }
        } catch(...) {}

        // If this is a RESOLVE_PLAY and we're about to process, insert signature to prevent reentry.
        if (cmd_def.type == core::CommandType::RESOLVE_PLAY) {
            resolving_action_sigs.insert(sig);
            inserted_sig = true;
        }

        // RAII guard to ensure signature removal on all exits
        struct ScopedSigRemover {
            GameInstance* inst;
            uint64_t s;
            bool active;
            ~ScopedSigRemover() {
                if (active) inst->resolving_action_sigs.erase(s);
            }
        } scoped_remover{this, sig, inserted_sig};

        // Ensure logs directory exists
        try {
            std::filesystem::create_directories("logs");
        } catch (...) {}
        // Open per-game trace log (append)
        std::string trace_file = "logs/game_trace_" + std::to_string(initial_seed_) + ".jsonl";
        try {
            std::ofstream lout(trace_file, std::ios::app);
            if (lout) {
                lout << "{\"event\":\"resolve_action_start\",";
                lout << "\"turn\":" << state.turn_number << ",";
                lout << "\"phase\":" << static_cast<int>(state.current_phase) << ",";
                lout << "\"active\":" << state.active_player_id << ",";
                lout << "\"command_type\":" << (int)cmd_def.type << ",";
                lout << "\"call_stack_size\":" << (pipeline ? pipeline->call_stack.size() : 0) << "}" << std::endl;
                lout.close();
            }
        } catch (...) {}

        std::unique_ptr<GameCommand> cmd = nullptr;

        // Debug logging before switch
        {
            std::ofstream dbg("c:\\temp\\resolve_debug.txt", std::ios::app);
            if (dbg) {
                dbg << "Before switch: cmd.type=" << (int)cmd_def.type << std::endl;
                dbg.flush();
                dbg.close();
            }
        }

        switch (cmd_def.type) {
            case core::CommandType::PLAY_FROM_ZONE:
                {
                    auto p_cmd = std::make_unique<PlayCardCommand>(cmd_def.instance_id);
                    p_cmd->is_spell_side = (cmd_def.amount == 1);
                    // spawn_source not in CommandDef, relies on inference
                    p_cmd->target_slot_index = cmd_def.target_slot_index;
                    cmd = std::move(p_cmd);
                }
                break;
            case core::CommandType::ATTACK_CREATURE:
                cmd = std::make_unique<AttackCommand>(cmd_def.instance_id, cmd_def.target_instance);
                break;
            case core::CommandType::ATTACK_PLAYER:
                {
                    PlayerID tgt_p = 1 - state.active_player_id; // Default opponent
                    cmd = std::make_unique<AttackCommand>(cmd_def.instance_id, -1, tgt_p);
                }
                break;
            case core::CommandType::BLOCK:
                cmd = std::make_unique<BlockCommand>(cmd_def.instance_id);
                break;
            case core::CommandType::MANA_CHARGE:
                cmd = std::make_unique<ManaChargeCommand>(cmd_def.instance_id);
                break;
            case core::CommandType::PASS:
                cmd = std::make_unique<PassCommand>();
                break;
            case core::CommandType::USE_ABILITY:
                cmd = std::make_unique<UseAbilityCommand>(cmd_def.instance_id, cmd_def.target_instance);
                break;
            case core::CommandType::CHOICE:
                {
                    int effect_idx = cmd_def.slot_index;
                    int option_idx = cmd_def.target_instance;
                    (void)option_idx; // Unused for now
                    
                    if (effect_idx >= 0 && effect_idx < (int)state.pending_effects.size()) {
                        state.pending_effects.erase(state.pending_effects.begin() + effect_idx);
                    }
                }
                break;
            case core::CommandType::SELECT_NUMBER:
                {
                    int effect_idx = cmd_def.slot_index;
                    int chosen_number = cmd_def.target_instance;
                    
                    if (effect_idx >= 0 && effect_idx < (int)state.pending_effects.size()) {
                        auto& pe = state.pending_effects[effect_idx];
                        pe.execution_context["_selected_number"] = chosen_number;
                        if (pe.effect_def.has_value() && !pe.effect_def->condition.str_val.empty()) {
                            std::string output_key = pe.effect_def->condition.str_val;
                            pe.execution_context[output_key] = chosen_number;
                        }
                        if (pe.effect_def.has_value()) {
                            core::PlayerID controller = dm::engine::EffectSystem::get_controller(state, pe.source_instance_id);
                            for (const auto& c : pe.effect_def->commands) {
                                dm::engine::systems::CommandSystem::execute_command(state, c, pe.source_instance_id, controller, pe.execution_context);
                            }
                        }
                        state.pending_effects.erase(state.pending_effects.begin() + effect_idx);
                    }
                }
                break;
            case core::CommandType::RESOLVE_EFFECT:
                systems::GameLogicSystem::dispatch_command(*pipeline, state, cmd_def, *card_db);
                systems::ContinuousEffectSystem::recalculate(state, *card_db);
                try { if (pipeline) pipeline->execute(nullptr, state, *card_db); } catch (...) {}
                return;

            default:
                // Fallback to dispatch_command
                systems::GameLogicSystem::dispatch_command(*pipeline, state, cmd_def, *card_db);
                systems::ContinuousEffectSystem::recalculate(state, *card_db);
                try { if (pipeline) pipeline->execute(nullptr, state, *card_db); } catch (...) {}
                return;
        }

        if (cmd) {
            state.execute_command(std::move(cmd));
        } else {
             systems::GameLogicSystem::dispatch_command(*pipeline, state, cmd_def, *card_db);
        }

        // Execute any instructions enqueued onto the pipeline during dispatch/command
        try {
            if (pipeline) pipeline->execute(nullptr, state, *card_db);
        } catch (...) {}

        // Log after execution
        try {
            std::ofstream lout(trace_file, std::ios::app);
            if (lout) {
                lout << "{\"event\":\"resolve_action_end\",";
                lout << "\"turn\":" << state.turn_number << ",";
                lout << "\"phase\":" << static_cast<int>(state.current_phase) << ",";
                lout << "\"active\":" << state.active_player_id << ",";
                lout << "\"winner\":" << static_cast<int>(state.winner) << ",";
                lout << "\"call_stack_size\":" << (pipeline ? pipeline->call_stack.size() : 0) << "}" << std::endl;
                lout.close();
            }
        } catch (...) {}

        systems::ContinuousEffectSystem::recalculate(state, *card_db);
    }

    void GameInstance::undo() {
        if (state.command_history.empty()) return;

        auto& cmd = state.command_history.back();
        cmd->invert(state);
        state.command_history.pop_back();
    }

    // ... [rest of file unchanged]

    void GameInstance::initialize_card_stats(int deck_size) {
        if (card_db) {
            state.initialize_card_stats(*card_db, deck_size);
        }
    }

    void GameInstance::reset_with_scenario(const ScenarioConfig& config) {
        // [Existing implementation remains unchanged]
        // 1. Reset Game State
        state.turn_number = 5;
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN;
        state.winner = GameResult::NONE;
        state.pending_effects.clear();
        state.current_attack = AttackState(); // Reset attack context

        // Clear all zones for both players
        for (auto& p : state.players) {
            p.hand.clear();
            p.battle_zone.clear();
            p.mana_zone.clear();
            p.graveyard.clear();
            p.shield_zone.clear();
            p.deck.clear();
        }

        // Instance ID counter
        int instance_id_counter = 0;

        // Fill decks
        // Player 0 (Me)
        if (!config.my_deck.empty()) {
             for (int cid : config.my_deck) {
                  state.players[0].deck.emplace_back((CardID)cid, instance_id_counter++, (PlayerID)0);
             }
        } else {
             // Fallback to dummy deck
             for(int i=0; i<30; ++i) {
                  state.players[0].deck.emplace_back((CardID)1, instance_id_counter++, (PlayerID)0);
             }
        }

        // Player 1 (Enemy)
        if (!config.enemy_deck.empty()) {
             for (int cid : config.enemy_deck) {
                  state.players[1].deck.emplace_back((CardID)cid, instance_id_counter++, (PlayerID)1);
             }
        } else {
             // Fallback to dummy deck
             for(int i=0; i<30; ++i) {
                  state.players[1].deck.emplace_back((CardID)1, instance_id_counter++, (PlayerID)1);
             }
        }

        // 2. Setup My Resources (Player 0)
        Player& me = state.players[0];

        // Hand
        for (int cid : config.my_hand_cards) {
            me.hand.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // Battle Zone
        for (int cid : config.my_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, me.id);
            c.summoning_sickness = false; // Assume creatures on board are ready
            me.battle_zone.push_back(c);
        }

        // Mana Zone
        for (int cid : config.my_mana_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, me.id);
            c.is_tapped = false;
            me.mana_zone.push_back(c);
        }

        if (config.my_mana_zone.empty() && config.my_mana > 0) {
            for (int i = 0; i < config.my_mana; ++i) {
                me.mana_zone.emplace_back(1, instance_id_counter++, me.id);
            }
        }

        // Graveyard
        for (int cid : config.my_grave_yard) {
            me.graveyard.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // My Shields (Player 0)
        for (int cid : config.my_shields) {
             me.shield_zone.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // 3. Setup Enemy Resources (Player 1)
        Player& enemy = state.players[1];

        // Enemy Battle Zone
        for (int cid : config.enemy_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, enemy.id);
            c.summoning_sickness = false;
            enemy.battle_zone.push_back(c);
        }

        // Enemy Shields
        for (int i = 0; i < config.enemy_shield_count; ++i) {
             enemy.shield_zone.emplace_back(1, instance_id_counter++, enemy.id);
        }

        // Initialize Owner Map [Phase A]
        state.card_owner_map.resize(instance_id_counter);

        auto populate_owner = [&](const Player& p) {
            auto register_cards = [&](const std::vector<CardInstance>& cards) {
                for (const auto& c : cards) {
                    if (c.instance_id >= 0) {
                        state.set_card_owner(c.instance_id, p.id);
                    }
                }
            };
            register_cards(p.hand);
            register_cards(p.battle_zone);
            register_cards(p.mana_zone);
            register_cards(p.graveyard);
            register_cards(p.shield_zone);
            register_cards(p.deck);
            register_cards(p.hyper_spatial_zone);
            register_cards(p.gr_deck);
        };

        populate_owner(state.players[0]);
        populate_owner(state.players[1]);
    }

    // A. Interactive Processing implementation
    void GameInstance::resume_processing(const std::vector<int>& inputs) {
         if (pipeline && pipeline->execution_paused) {
             // Basic support for single integer input for now (common case)
             dm::engine::systems::ContextValue val;
             if (inputs.empty()) {
                  // No input?
             } else if (inputs.size() == 1) {
                  val = inputs[0];
             } else {
                  val = inputs;
             }
             pipeline->resume(state, *card_db, val);
             systems::ContinuousEffectSystem::recalculate(state, *card_db);
         }
    }

}
