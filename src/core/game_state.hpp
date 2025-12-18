#ifndef DM_CORE_GAME_STATE_HPP
#define DM_CORE_GAME_STATE_HPP

#include "types.hpp"
#include "card_def.hpp"
#include "card_stats.hpp"
#include "pending_effect.hpp"
#include "modifiers.hpp"
#include "game_event.hpp" // Added and removed struct GameEvent def
#include "engine/systems/trigger_system/reaction_window.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <random>
#include <optional>
#include <functional>

// Forward declarations
namespace dm::engine::systems {
    class PipelineExecutor;
}

namespace dm::engine::game_command {
    class GameCommand;
}

namespace dm::core {

    // Note: Zone, Phase, EffectType defined in types.hpp

    struct Player {
        PlayerID id = 0; // Added ID member
        std::vector<CardInstance> hand;
        std::vector<CardInstance> mana_zone;
        std::vector<CardInstance> battle_zone;
        std::vector<CardInstance> shield_zone;
        std::vector<CardInstance> graveyard;
        std::vector<CardInstance> deck;
        std::vector<CardInstance> hyper_spatial_zone;
        std::vector<CardInstance> gr_deck;

        std::vector<CardInstance> effect_buffer;
    };

    struct AttackState {
        int source_instance_id = -1;
        int target_instance_id = -1; // -1 for player
        int target_player_id = -1;
        bool blocked = false;
        int blocking_creature_id = -1;
    };

    class GameState {
    public:
        enum class Status {
            PLAYING,
            WAITING_FOR_REACTION,
            GAME_OVER
        };
        Status status = Status::PLAYING;

        int turn_number = 1;
        PlayerID active_player_id = 0;
        Phase current_phase = Phase::START_OF_TURN;
        std::vector<Player> players;

        bool game_over = false;
        GameResult winner = GameResult::NONE;
        
        std::vector<PendingEffect> pending_effects;
        std::vector<dm::engine::systems::ReactionWindow> reaction_stack;

        std::vector<CostModifier> active_modifiers;
        std::vector<PassiveEffect> passive_effects;

        // Current attack state (for hash and logic)
        AttackState current_attack;

        std::mt19937 rng;

        std::vector<PlayerID> card_owner_map;

        TurnStats turn_stats;
        bool stats_recorded = false;
        std::map<PlayerID, std::vector<std::pair<CardID, int>>> played_cards_history_this_game;

        // Requires full definition of GameCommand in .cpp for deletion
        std::vector<std::unique_ptr<dm::engine::game_command::GameCommand>> command_history;

        struct QueryContext {
            int query_id = 0;
            std::string query_type;
            std::map<std::string, int> params;
            std::vector<int> valid_targets;
            std::vector<std::string> options;
        };

        bool waiting_for_user_input = false;
        QueryContext pending_query;

        std::shared_ptr<void> active_pipeline; // Can store PipelineExecutor generic pointer

        // Event Dispatcher callback
        std::function<void(const GameEvent&)> event_dispatcher;

        // Stats members needed for compilation
        std::map<CardID, CardStats> global_card_stats;
        CardStats initial_deck_stats_sum;
        CardStats visible_stats_sum;
        int initial_deck_count = 0;
        int visible_card_count = 0;
        std::vector<size_t> hash_history;
        bool loop_proven = false;

        GameState(int seed = 0);
        ~GameState();

        // Move-only due to unique_ptr
        GameState(GameState&&) noexcept;
        GameState& operator=(GameState&&) noexcept;
        GameState(const GameState&) = delete;
        GameState& operator=(const GameState&) = delete;

        void setup_test_duel();
        CardInstance* get_card_instance(int instance_id);
        const CardInstance* get_card_instance(int instance_id) const;
        std::vector<int> get_zone(PlayerID pid, Zone zone) const;
        void execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd);

        GameState clone() const;
        size_t calculate_hash() const;

        void add_card_to_zone(const CardInstance& card, Zone zone, PlayerID pid);

        // Declarations for methods implemented in other files
        void update_loop_check();
        void initialize_card_stats(const std::map<CardID, CardDefinition>& card_db, int deck_size);
        bool load_card_stats_from_json(const std::string& filepath);
        void compute_initial_deck_sums(const std::vector<CardID>& deck_list);
        void on_card_reveal(CardID cid);
        void on_card_play(CardID cid, int turn, bool is_trigger, int cost_diff, PlayerID pid);
        void on_game_finished(GameResult result);
        std::vector<float> vectorize_card_stats(CardID cid) const;
        std::vector<float> get_library_potential() const;
    };

}

#endif // DM_CORE_GAME_STATE_HPP
