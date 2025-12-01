#pragma once
#include "types.hpp"
#include "constants.hpp"
#include "card_stats.hpp"
#include "card_def.hpp"
#include <vector>
#include <array>
#include <random>
#include <stdexcept>
#include <optional>
#include <map>
#include "card_json_types.hpp"

namespace dm::core {

    struct CardInstance {
        CardID card_id; // The definition ID
        int instance_id; // Unique ID for this instance during the game
        bool is_tapped = false;
        bool summoning_sickness = false;
        bool is_face_down = false; // For shields or terror pit etc? Shields are face down by default.
        int power_mod = 0; // temporary/ongoing power modifications applied by effects
        
        // Constructors
        CardInstance() : card_id(0), instance_id(-1) {}
        CardInstance(CardID cid, int iid) : card_id(cid), instance_id(iid) {}
    };

    struct PendingEffect {
        EffectType type;
        int source_instance_id;
        PlayerID controller;

        // Targeting
        std::vector<int> target_instance_ids;
        int num_targets_needed = 0;

        // Optional: carry the EffectDef (from JSON) for later resolution after target selection
        std::optional<EffectDef> effect_def;

        PendingEffect(EffectType t, int src, PlayerID p) 
            : type(t), source_instance_id(src), controller(p) {}
    };

    struct AttackRequest {
        int source_instance_id;
        int target_instance_id; // -1 if attacking player
        PlayerID target_player; // Valid if target_instance_id == -1
        bool is_blocked = false;
        int blocker_instance_id = -1;
    };

    struct Player {
        PlayerID id;
        std::vector<CardInstance> hand;
        std::vector<CardInstance> mana_zone;
        std::vector<CardInstance> battle_zone;
        std::vector<CardInstance> graveyard;
        std::vector<CardInstance> shield_zone;
        std::vector<CardInstance> deck;
        std::vector<CardInstance> hyper_spatial_zone;
        std::vector<CardInstance> gr_deck;
    };

    struct GameState {
        int turn_number = 1;
        PlayerID active_player_id = 0;
        Phase current_phase = Phase::START_OF_TURN;
        GameResult winner = GameResult::NONE;
        std::array<Player, 2> players;
        
        // Pending Effects Pool [Spec 4.1, 4.2]
        std::vector<PendingEffect> pending_effects;

        // Current Attack Context
        AttackRequest current_attack;

        // Determinism: std::mt19937 inside State [Q69]
        std::mt19937 rng;

        // Result Stats / POMDP support
        // Map CardID -> aggregated CardStats (sums and counts)
        std::map<CardID, CardStats> global_card_stats;

        // Initial deck aggregate sums (sum over cards placed in initial deck)
        CardStats initial_deck_stats_sum;
        CardStats visible_stats_sum;
        int initial_deck_count = 40;
        int visible_card_count = 0;

        GameState(uint32_t seed) : rng(seed) {
            players[0].id = 0;
            players[1].id = 1;
        }

        // POMDP helpers
        void on_card_reveal(CardID cid);
        std::vector<float> vectorize_card_stats(CardID cid) const;
        std::vector<float> get_library_potential() const;
        // Initialize card stats map from card DB (creates entries for all known CardIDs)
        void initialize_card_stats(const std::map<CardID, CardDefinition>& card_db, int deck_size = 40);
        // Load historical card stats from JSON file (format: array of {id, play_count, averages:[16] or sums:[16]})
        bool load_card_stats_from_json(const std::string& filepath);
        // Given an explicit deck list (vector of CardIDs), compute initial_deck_stats_sum as sum of per-card averages
        void compute_initial_deck_sums(const std::vector<CardID>& deck_list);

        Player& get_active_player() {
            return players[active_player_id];
        }

        Player& get_non_active_player() {
            return players[1 - active_player_id];
        }
        
        // Error Handling [Q43, Q87]
        void panic(const char* message) const {
            throw std::runtime_error(message);
        }
    };
}
