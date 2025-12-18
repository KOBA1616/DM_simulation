#ifndef DM_CORE_GAME_STATE_HPP
#define DM_CORE_GAME_STATE_HPP

#include "types.hpp"
#include "card_def.hpp"
#include "card_stats.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <random>
#include <optional>

// Include full command definition to avoid incomplete type errors in unique_ptr
#include "engine/game_command/commands.hpp"

// Forward declarations
namespace dm::engine::systems {
    class PipelineExecutor;
}

namespace dm::core {

    // Note: Zone, Phase, EffectType defined in types.hpp

    struct PendingEffect {
        EffectType type;
        int source_instance_id;
        PlayerID controller;
        ResolveType resolve_type = ResolveType::NORMAL;

        FilterDef filter;
        int num_targets_needed = 0;
        std::vector<int> target_instance_ids;
        bool optional = false;

        std::optional<EffectDef> effect_def;
        std::map<std::string, int> execution_context;

        int chain_depth = 0;

        PendingEffect(EffectType t, int src, PlayerID ctrl)
            : type(t), source_instance_id(src), controller(ctrl) {}
    };

    struct CostModifier {
        int reduction_amount = 0;
        FilterDef condition_filter;
        int turns_remaining = 0;
        int source_instance_id = -1;
        PlayerID controller = 0;
    };

    enum class PassiveType {
        NONE,
        POWER_MODIFIER,
        ADD_RACE,
        ADD_CIVILIZATION,
        KEYWORD_GRANT,
        BLOCKER_GRANT,
        SPEED_ATTACKER_GRANT,
        SLAYER_GRANT,
        CANNOT_USE_SPELLS,
        CANNOT_ATTACK_PLAYER,
        CANNOT_ATTACK_CREATURE
    };

    struct PassiveEffect {
        PassiveType type = PassiveType::NONE;
        int value = 0;
        std::string str_value;
        FilterDef target_filter;
        int turns_remaining = 0;
        int source_instance_id = -1;
        PlayerID controller = 0;
    };

    struct Player {
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

    class GameState {
    public:
        int turn_number = 1;
        PlayerID active_player_id = 0;
        Phase current_phase = Phase::START_OF_TURN;
        std::vector<Player> players;

        bool game_over = false;
        GameResult winner = GameResult::NONE;
        
        std::vector<PendingEffect> pending_effects;

        std::vector<CostModifier> active_modifiers;
        std::vector<PassiveEffect> passive_effects;

        std::mt19937 rng;

        std::vector<PlayerID> card_owner_map;

        TurnStats turn_stats;
        bool stats_recorded = false;
        std::vector<std::pair<CardID, int>> played_cards_history_this_game;

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

        std::shared_ptr<void> active_pipeline;

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
