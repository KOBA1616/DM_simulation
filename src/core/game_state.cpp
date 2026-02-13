#include "game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp" // Include for cloning
#include "engine/diag_win32.h"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include <fstream>

namespace dm::core {

    GameState::GameState(int seed) : players(2) {
        rng.seed(seed);
        for (size_t i = 0; i < players.size(); ++i) {
            players[i].id = static_cast<PlayerID>(i);
        }
    }

    GameState::~GameState() {
        // Avoid C++ stream/CRT allocations in destructor path; use low-level write only.
        try { diag_write_win32(std::string("GameState::~GameState turn=") + std::to_string(turn_number) + " card_owner_map_sz=" + std::to_string(card_owner_map.size())); } catch(...) {}
    }
    GameState::GameState(GameState&&) noexcept = default;
    GameState& GameState::operator=(GameState&&) noexcept = default;

    void GameState::setup_test_duel() {
        // Simple setup for tests
        players.resize(2);
        for (size_t i = 0; i < players.size(); ++i) {
            players[i].id = static_cast<PlayerID>(i);
        }
        for(auto& p : players) {
            p.hand.clear();
            p.mana_zone.clear();
            p.battle_zone.clear();
            p.shield_zone.clear();
            p.graveyard.clear();
            p.deck.clear();
        }
        // Clear card_owner_map to reset instance_id counter
        card_owner_map.clear();
        turn_number = 1;
        active_player_id = 0;
        current_phase = Phase::START_OF_TURN;
    }

    GameState GameState::clone() const {
        GameState new_state;
        new_state.turn_number = turn_number;
        new_state.active_player_id = active_player_id;
        new_state.current_phase = current_phase;
        new_state.players = players;
        new_state.game_over = game_over;
        new_state.winner = winner;
        new_state.pending_effects = pending_effects;
        new_state.reaction_stack = reaction_stack;
        new_state.active_modifiers = active_modifiers;
        new_state.passive_effects = passive_effects;
        new_state.current_attack = current_attack;
        new_state.rng = rng;
        new_state.card_owner_map = card_owner_map;
        new_state.turn_stats = turn_stats;
        new_state.stats_recorded = stats_recorded;
        new_state.played_cards_history_this_game = played_cards_history_this_game;
        new_state.waiting_for_user_input = waiting_for_user_input;
        new_state.pending_query = pending_query;
        new_state.status = status;
        // Do NOT copy event_dispatcher. It may contain callbacks to UI or logger that are not thread-safe or needed for simulation.
        // new_state.event_dispatcher = event_dispatcher;

        // Do NOT copy command_history. It is only for replay/undo and grows indefinitely.
        // new_state.command_history = command_history;

        // Deep copy pipeline if active
        if (active_pipeline) {
            auto ptr = std::static_pointer_cast<dm::engine::systems::PipelineExecutor>(active_pipeline);
            new_state.active_pipeline = ptr->clone();
        }

        // Stats
        new_state.historical_card_stats = historical_card_stats; // Share
        new_state.global_card_stats = global_card_stats; // Copy (now small)
        new_state.initial_deck_stats_sum = initial_deck_stats_sum;
        new_state.visible_stats_sum = visible_stats_sum;
        new_state.initial_deck_count = initial_deck_count;
        new_state.visible_card_count = visible_card_count;
        new_state.hash_history = hash_history;
        new_state.loop_proven = loop_proven;

        return new_state;
    }

    void GameState::add_card_to_zone(const CardInstance& card, Zone zone, PlayerID pid) {
        // Delegate to GameCommand to ensure history tracking
        auto cmd = std::make_shared<dm::engine::game_command::AddCardCommand>(card, zone, pid);
        execute_command(cmd);
    }

    void GameState::add_passive_effect(const PassiveEffect& effect) {
        passive_effects.push_back(effect);
    }

    void GameState::register_card_instance(const CardInstance& card) {
        if (card.instance_id < 0) return;
        if (card.instance_id >= (int)card_owner_map.size()) {
            try { diag_write_win32(std::string("RESIZE card_owner_map before_sz=") + std::to_string(card_owner_map.size()) + " target_id=" + std::to_string(card.instance_id)); } catch(...) {}
            // Conservative resize: ensure exactly enough space to hold this id and use -1 as unknown owner
            size_t new_sz = (size_t)card.instance_id + 1;
            card_owner_map.resize(new_sz, static_cast<PlayerID>(-1));
            try { diag_write_win32(std::string("RESIZE card_owner_map after_sz=") + std::to_string(card_owner_map.size())); } catch(...) {}
        }
        try { diag_write_win32(std::string("REGISTER_CARD instance_id=") + std::to_string(card.instance_id) + " owner=" + std::to_string((int)card.owner)); } catch(...) {}
        // Defensive write: only write when index valid
        if (card.instance_id >= 0 && card.instance_id < (int)card_owner_map.size()) {
            card_owner_map[card.instance_id] = card.owner;
        } else {
            try { diag_write_win32(std::string("REGISTER_CARD SKIPPED OOB instance_id=") + std::to_string(card.instance_id)); } catch(...) {}
        }
    }

    void GameState::ensure_owner_map_for(int instance_id) {
        if (instance_id < 0) return;
        if ((size_t)instance_id >= card_owner_map.size()) {
            try { diag_write_win32(std::string("ENSURE_OWNER_MAP for id=") + std::to_string(instance_id) + " before_sz=" + std::to_string(card_owner_map.size())); } catch(...) {}
            card_owner_map.resize((size_t)instance_id + 1, static_cast<PlayerID>(-1));
            try { diag_write_win32(std::string("ENSURE_OWNER_MAP after_sz=") + std::to_string(card_owner_map.size())); } catch(...) {}
        }
    }

    void GameState::set_card_owner(int instance_id, PlayerID owner) {
        if (instance_id < 0) return;
        ensure_owner_map_for(instance_id);
        if (instance_id >= 0 && (size_t)instance_id < card_owner_map.size()) {
            card_owner_map[instance_id] = owner;
        } else {
            try { diag_write_win32(std::string("SET_CARD_OWNER SKIPPED OOB id=") + std::to_string(instance_id)); } catch(...) {}
        }
    }

    PlayerID GameState::get_card_owner(int instance_id) const {
        if (instance_id < 0) return static_cast<PlayerID>(-1);
        if ((size_t)instance_id >= card_owner_map.size()) return static_cast<PlayerID>(-1);
        return card_owner_map[instance_id];
    }

    CardInstance* GameState::get_card_instance(int instance_id) {
        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "get_card_instance entry id=" << instance_id << " owner_map_sz=" << card_owner_map.size() << "\n";
                diag.close();
            }
        } catch(...) {}
        if(instance_id < 0 || instance_id >= (int)card_owner_map.size()) return nullptr;
        PlayerID pid = card_owner_map[instance_id];
        if(pid >= players.size()) return nullptr;

        auto find_in = [&](std::vector<CardInstance>& v) -> CardInstance* {
            for(auto& c : v) {
                if(c.instance_id == instance_id) {
                    try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"GET_CARD found id="<<instance_id<<" owner="<<(int)pid<<"\n"; d.flush(); d.close(); } } catch(...) {}
                    return &c;
                }
            }
            return nullptr;
        };

        if(auto* c = find_in(players[pid].battle_zone)) return c;
        if(auto* c = find_in(players[pid].hand)) return c;
        if(auto* c = find_in(players[pid].mana_zone)) return c;
        if(auto* c = find_in(players[pid].shield_zone)) return c;
        if(auto* c = find_in(players[pid].graveyard)) return c;
        if(auto* c = find_in(players[pid].deck)) return c;
        if(auto* c = find_in(players[pid].effect_buffer)) return c;
        if(auto* c = find_in(players[pid].stack)) return c;
        if(auto* c = find_in(players[pid].hyper_spatial_zone)) return c;
        if(auto* c = find_in(players[pid].gr_deck)) return c;
        return nullptr;
    }

    const CardInstance* GameState::get_card_instance(int instance_id) const {
        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "get_card_instance const entry id=" << instance_id << " owner_map_sz=" << card_owner_map.size() << "\n";
                diag.close();
            }
        } catch(...) {}
        if(instance_id < 0 || instance_id >= (int)card_owner_map.size()) return nullptr;
        PlayerID pid = card_owner_map[instance_id];
        if(pid >= players.size()) return nullptr;

        auto find_in = [&](const std::vector<CardInstance>& v) -> const CardInstance* {
            for(const auto& c : v) {
                if(c.instance_id == instance_id) {
                    try { std::ofstream d("logs/crash_diag.txt", std::ios::app); if(d){ d<<"GET_CARD_CONST found id="<<instance_id<<" owner="<<(int)pid<<"\n"; d.flush(); d.close(); } } catch(...) {}
                    return &c;
                }
            }
            return nullptr;
        };

        if(auto* c = find_in(players[pid].battle_zone)) return c;
        if(auto* c = find_in(players[pid].hand)) return c;
        if(auto* c = find_in(players[pid].mana_zone)) return c;
        if(auto* c = find_in(players[pid].shield_zone)) return c;
        if(auto* c = find_in(players[pid].graveyard)) return c;
        if(auto* c = find_in(players[pid].deck)) return c;
        if(auto* c = find_in(players[pid].effect_buffer)) return c;
        if(auto* c = find_in(players[pid].stack)) return c;
        if(auto* c = find_in(players[pid].hyper_spatial_zone)) return c;
        if(auto* c = find_in(players[pid].gr_deck)) return c;
        return nullptr;
    }

    std::vector<int> GameState::get_zone(PlayerID pid, Zone zone) const {
        std::vector<int> ids;
        if(pid >= players.size()) return ids;
        const auto& p = players[pid];
        const std::vector<CardInstance>* z = nullptr;
        switch(zone) {
            case Zone::HAND: z = &p.hand; break;
            case Zone::MANA: z = &p.mana_zone; break;
            case Zone::BATTLE: z = &p.battle_zone; break;
            case Zone::SHIELD: z = &p.shield_zone; break;
            case Zone::GRAVEYARD: z = &p.graveyard; break;
            case Zone::DECK: z = &p.deck; break;
            case Zone::BUFFER: z = &p.effect_buffer; break;
            case Zone::STACK: z = &p.stack; break;
            case Zone::HYPER_SPATIAL: z = &p.hyper_spatial_zone; break;
            case Zone::GR_DECK: z = &p.gr_deck; break;
            default: break;
        }
        if(z) {
            for(const auto& c : *z) ids.push_back(c.instance_id);
        }
        return ids;
    }

    void GameState::execute_command(std::shared_ptr<dm::engine::game_command::GameCommand> cmd) {
        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "EXEC_CMD entry type=" << static_cast<int>(cmd->get_type())
                     << " history_sz=" << command_history.size() << "\n";
                diag.flush();
                diag.close();
            }
        } catch(...) {}

        try { diag_write_win32(std::string("EXEC_CMD BEFORE_EXEC type=") + std::to_string((int)cmd->get_type()) + " history_sz=" + std::to_string(command_history.size()) + " history_sentinel=" + std::to_string(history_sentinel)); } catch(...) {}
        // Also write a flushed text record before executing the command to help post-mortem analysis.
        try {
            std::ofstream pre("logs/crash_diag.txt", std::ios::app);
            if (pre) {
                pre << "EXEC_CMD TEXT_BEFORE type=" << static_cast<int>(cmd->get_type()) << " history_sz=" << command_history.size() << " history_sentinel=" << history_sentinel << "\n";
                pre.flush();
                pre.close();
            }
        } catch(...) {}

        cmd->execute(*this);

        // Low-level and text marker immediately after command execution
        try { diag_write_win32(std::string("EXEC_CMD AFTER_EXEC type=") + std::to_string((int)cmd->get_type()) + " history_sz=" + std::to_string(command_history.size()) + " history_sentinel=" + std::to_string(history_sentinel)); } catch(...) {}
        try {
            std::ofstream post("logs/crash_diag.txt", std::ios::app);
            if (post) {
                post << "EXEC_CMD TEXT_AFTER type=" << static_cast<int>(cmd->get_type()) << " history_sz=" << command_history.size() << " history_sentinel=" << history_sentinel << "\n";
                post.flush();
                post.close();
            }
        } catch(...) {}

        if (command_redirect_target) {
            try { diag_write_win32(std::string("EXEC_CMD REDIRECT push history_sz_before=") + std::to_string(command_history.size()) + " history_sentinel=" + std::to_string(history_sentinel)); } catch(...) {}
            command_redirect_target->push_back(std::move(cmd));
            // update sentinel after mutation
            try { ++history_sentinel; } catch(...) {}
            try { diag_write_win32(std::string("EXEC_CMD REDIRECT pushed history_sz_after=") + std::to_string(command_history.size()) + " history_sentinel=" + std::to_string(history_sentinel)); } catch(...) {}
        } else {
            try { diag_write_win32(std::string("EXEC_CMD push history_sz_before=") + std::to_string(command_history.size()) + " history_sentinel=" + std::to_string(history_sentinel)); } catch(...) {}
            command_history.push_back(std::move(cmd));
            // update sentinel after mutation
            try { ++history_sentinel; } catch(...) {}
            try { diag_write_win32(std::string("EXEC_CMD pushed history_sz_after=") + std::to_string(command_history.size()) + " history_sentinel=" + std::to_string(history_sentinel)); } catch(...) {}
        }

        try {
            std::ofstream diag("logs/crash_diag.txt", std::ios::app);
            if (diag) {
                diag << "EXEC_CMD exit history_sz=" << command_history.size() << "\n";
                diag.flush();
                diag.close();
            }
        } catch(...) {}
    }

    // Removed unique_ptr overload implementation

    void GameState::undo() {
        if (command_history.empty()) return;
        auto& cmd = command_history.back();
        cmd->invert(*this);
        command_history.pop_back();
    }

    CardStats GameState::get_card_stats(CardID cid) const {
        // Priority: Local > Historical > Empty
        auto it = global_card_stats.find(cid);
        if (it != global_card_stats.end()) {
            return it->second;
        }
        if (historical_card_stats) {
            auto it_hist = historical_card_stats->find(cid);
            if (it_hist != historical_card_stats->end()) {
                return it_hist->second;
            }
        }
        return CardStats{};
    }

    CardStats& GameState::get_mutable_card_stats(CardID cid) {
        // If present locally, return it.
        // If not, copy from historical (if exists) or create new, insert into local, return it.
        auto it = global_card_stats.find(cid);
        if (it == global_card_stats.end()) {
            CardStats cs;
            if (historical_card_stats) {
                auto it_hist = historical_card_stats->find(cid);
                if (it_hist != historical_card_stats->end()) {
                    cs = it_hist->second;
                }
            }
            // Insert and get iterator
            it = global_card_stats.emplace(cid, cs).first;
        }
        return it->second;
    }

    float GameState::calculate_board_advantage(PlayerID player_id, const std::map<CardID, CardDefinition>& card_db) const {
        // Simple fallback heuristic used by some AI paths: sum powers on board for player minus opponent.
        try {
            if (player_id < 0 || player_id >= (PlayerID)players.size()) return 0.0f;
            int my_power = 0;
            int opp_power = 0;
            for (size_t pid = 0; pid < players.size(); ++pid) {
                int sum = 0;
                for (const auto& c : players[pid].battle_zone) {
                    auto it = card_db.find(c.card_id);
                    if (it != card_db.end()) sum += it->second.power;
                }
                if ((PlayerID)pid == player_id) my_power = sum; else opp_power = sum;
            }
            return static_cast<float>(my_power - opp_power);
        } catch(...) {
            return 0.0f;
        }
    }

    GameState::StateSnapshot GameState::create_snapshot() const {
        StateSnapshot snap;
        snap.commands_since_snapshot = command_history;
        snap.hash_at_snapshot = calculate_hash();
        return snap;
    }

    void GameState::restore_snapshot(const StateSnapshot& snap) {
        if (command_history.size() < snap.commands_since_snapshot.size()) {
             return;
        }
        while (command_history.size() > snap.commands_since_snapshot.size()) {
            undo();
        }
    }

    void GameState::make_move(const CommandDef& cmd) {
        execute_turn_command(cmd);
    }

    void GameState::execute_turn_command(const CommandDef& cmd) {
        move_start_indices.push_back(command_history.size());
        const auto& card_db = dm::engine::CardRegistry::get_all_definitions();

        if (!active_pipeline) {
            active_pipeline = std::make_shared<dm::engine::systems::PipelineExecutor>();
        } else {
            // Only clear stack if not paused (waiting for input)
            if (!active_pipeline->execution_paused) {
                active_pipeline->call_stack.clear();
                active_pipeline->execution_paused = false;
            }
        }

        dm::engine::systems::GameLogicSystem::dispatch_command(*active_pipeline, *this, cmd, card_db);
        active_pipeline->execute(nullptr, *this, card_db);
    }

    void GameState::unmake_move() {
        if (move_start_indices.empty()) return;
        size_t target_size = move_start_indices.back();
        move_start_indices.pop_back();

        while (command_history.size() > target_size) {
            undo();
        }
    }

}
