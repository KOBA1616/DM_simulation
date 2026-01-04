#include "game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp" // Include for cloning

namespace dm::core {

    GameState::GameState(int seed) : players(2) {
        rng.seed(seed);
    }

    GameState::~GameState() = default;
    GameState::GameState(GameState&&) noexcept = default;
    GameState& GameState::operator=(GameState&&) noexcept = default;

    void GameState::setup_test_duel() {
        // Simple setup for tests
        players.resize(2);
        for(auto& p : players) {
            p.hand.clear();
            p.mana_zone.clear();
            p.battle_zone.clear();
            p.shield_zone.clear();
            p.graveyard.clear();
            p.deck.clear();
        }
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

    void GameState::register_card_instance(const CardInstance& card) {
        if (card.instance_id < 0) return;
        if (card.instance_id >= (int)card_owner_map.size()) {
            card_owner_map.resize(card.instance_id + 100, 0); // Proactively resize
        }
        card_owner_map[card.instance_id] = card.owner;
    }

    CardInstance* GameState::get_card_instance(int instance_id) {
        if(instance_id < 0 || instance_id >= (int)card_owner_map.size()) return nullptr;
        PlayerID pid = card_owner_map[instance_id];
        if(pid >= players.size()) return nullptr;

        auto find_in = [&](std::vector<CardInstance>& v) -> CardInstance* {
            for(auto& c : v) if(c.instance_id == instance_id) return &c;
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
        if(instance_id < 0 || instance_id >= (int)card_owner_map.size()) return nullptr;
        PlayerID pid = card_owner_map[instance_id];
        if(pid >= players.size()) return nullptr;

        auto find_in = [&](const std::vector<CardInstance>& v) -> const CardInstance* {
            for(const auto& c : v) if(c.instance_id == instance_id) return &c;
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
        cmd->execute(*this);
        if (command_redirect_target) {
            command_redirect_target->push_back(std::move(cmd));
        } else {
            command_history.push_back(std::move(cmd));
        }
    }

    void GameState::execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd) {
        execute_command(std::shared_ptr<dm::engine::game_command::GameCommand>(std::move(cmd)));
    }

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

}
