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
#include "game_command_fwd.hpp"
#include <memory>

namespace dm::core {

    struct CardInstance {
        CardID card_id; // The definition ID
        int instance_id; // Unique ID for this instance during the game
        bool is_tapped = false;
        bool summoning_sickness = false;
        bool is_face_down = false; // For shields or terror pit etc? Shields are face down by default.
        int power_mod = 0; // temporary/ongoing power modifications applied by effects
        int turn_played = -1; // Turn number when this card was put into the battle zone (for Just Diver/SS)
        PlayerID owner = 255; // 255 = None/Unknown

        // Step 2-1: Hierarchy Support
        // Cards stacked underneath this card (Evolution sources, etc.)
        std::vector<CardInstance> underlying_cards;

        // Step 4-3: Twinpact State (Set when placed in stack)
        bool is_spell_side_mode = false;

        // Phase 4: Cost Payment Metadata (e.g. number of creatures tapped for reduction)
        int cost_payment_meta = 0;

        // Constructors
        CardInstance() : card_id(0), instance_id(-1), is_tapped(false), summoning_sickness(true), is_face_down(false) {}
        CardInstance(CardID cid, int iid) : card_id(cid), instance_id(iid), is_tapped(false), summoning_sickness(true), is_face_down(false) {}
    };

    struct PendingEffect {
        EffectType type;
        int source_instance_id;
        PlayerID controller;

        // Targeting
        std::vector<int> target_instance_ids;
        int num_targets_needed = 0;
        ResolveType resolve_type = ResolveType::NONE;

        FilterDef filter; // The filter used for selection
        bool optional = false; // If true, can choose to select nothing (PASS)

        // Optional: carry the EffectDef (from JSON) for later resolution after target selection
        std::optional<EffectDef> effect_def;

        // For SELECT_OPTION: Store the choices
        std::vector<std::vector<ActionDef>> options;

        // Phase 5: Execution Context (Variable Linking)
        std::map<std::string, int> execution_context;

        // Optional context for REACTION_WINDOW
        struct ReactionContext {
            std::string trigger_event; // The event being reacted to (e.g., "ON_BLOCK_OR_ATTACK")
            int attacking_creature_id = -1; // Instance ID of attacker
            int blocked_creature_id = -1;
        };
        std::optional<ReactionContext> reaction_context;

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

    struct CostModifier {
        int reduction_amount;
        // Condition/Filter for which cards this modifier applies to
        FilterDef condition_filter;
        int turns_remaining; // 1 = this turn only, >1 = persistent, -1 = indefinite
        int source_instance_id; // To track where it came from (e.g. Cocco Lupia)
        PlayerID controller;
    };

    // Step 5-1: Passive Effects
    enum class PassiveType {
        POWER_MODIFIER,
        // KEYWORD_GRANT was missing here, but present in handlers.
        // It must be defined for compilation.
        KEYWORD_GRANT,
        COST_REDUCTION, // Merged with CostModifier? Or keep separate?
        // CostModifier is for HAND/MANA cards. PassiveEffect is usually for BATTLE ZONE.
        // We will focus on BATTLE ZONE passives here.
        BLOCKER_GRANT,
        SPEED_ATTACKER_GRANT,
        SLAYER_GRANT,
        // Step 3-4: Attack/Block Restriction
        CANNOT_ATTACK,
        CANNOT_BLOCK,
        CANNOT_USE_SPELLS, // Step 3-x: "Cannot cast spell" (locking)
        LOCK_SPELL_BY_COST // "Declare Number -> Prohibit Spells"
    };

    struct PassiveEffect {
        PassiveType type;
        int value; // e.g. +1000 power
        std::string str_value; // e.g. "SPEED_ATTACKER"
        FilterDef target_filter; // Which creatures get this buff?
        ConditionDef condition; // "If shields 0..."
        int source_instance_id; // The source of the effect (e.g. Rose Castle)
        PlayerID controller;
        int turns_remaining = -1; // -1 = permanent, 1 = this turn only
    };

    // Phase 5: Turn Stats for Meta Counter logic
    struct TurnStats {
        bool played_without_mana = false; // True if a card was played with 0 actual mana paid (except cost reduction down to >= 1)
        int cards_drawn_this_turn = 0;
        int cards_discarded_this_turn = 0;
        int creatures_played_this_turn = 0;
        int spells_cast_this_turn = 0;
        int attacks_declared_this_turn = 0; // Step 2-3: Track attacks
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

        // Step 3-1: Per-Player Effect Buffer [Stability Fix]
        std::vector<CardInstance> effect_buffer;
    };

    struct GameState {
        int turn_number = 1;
        PlayerID active_player_id = 0;
        Phase current_phase = Phase::START_OF_TURN;
        GameResult winner = GameResult::NONE;

        // Loop Detection
        std::vector<uint64_t> hash_history;
        bool loop_proven = false;
        void update_loop_check();

        std::array<Player, 2> players;
        
        // Pending Effects Pool [Spec 4.1, 4.2]
        std::vector<PendingEffect> pending_effects;

        // Current Attack Context
        AttackRequest current_attack;

        // Active Cost Modifiers (e.g. Cocco Lupia, Fairy Gift)
        std::vector<CostModifier> active_modifiers;

        // Passive Effects (Step 5-1)
        std::vector<PassiveEffect> passive_effects;

        // Stack Zone for declared cards (waiting for cost payment) [PLAN-002]
        std::vector<CardInstance> stack_zone;

        // Effect Buffer moved to Player struct [Step 3-1]

        // Turn Stats [Phase 5]
        TurnStats turn_stats;

        // Determinism: std::mt19937 inside State [Q69]
        std::mt19937 rng;

        // Owner Map [Phase A]
        // Index is instance_id, Value is owner PlayerID
        std::vector<PlayerID> card_owner_map;

        // Result Stats / POMDP support
        // Map CardID -> aggregated CardStats (sums and counts)
        std::map<CardID, CardStats> global_card_stats;

        // Initial deck aggregate sums (sum over cards placed in initial deck)
        CardStats initial_deck_stats_sum;
        CardStats visible_stats_sum;
        int initial_deck_count = 40;
        int visible_card_count = 0;

        // Phase 6: GameCommand History
        std::vector<std::shared_ptr<dm::engine::game_command::GameCommand>> command_history;

        // Phase 6: Query/Decide
        bool waiting_for_user_input = false;
        struct QueryContext {
             int query_id = 0; // Incrementing ID
             std::string query_type; // e.g., "SELECT_TARGET", "SELECT_OPTION"
             std::map<std::string, int> params; // e.g., "min_count": 1, "max_count": 1
             std::vector<int> valid_target_ids;
             std::vector<std::string> options;
        };
        std::optional<QueryContext> pending_query;

        GameState() : rng(0) {
            players[0].id = 0;
            players[1].id = 1;
        }

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

        // Stats tracking
        // Stores pair of (CardID, TurnNumber)
        std::vector<std::pair<CardID, int>> played_cards_history_this_game[2];
        bool stats_recorded = false;
        void on_card_play(CardID cid, int turn, bool is_trigger, int cost_diff, PlayerID pid);
        void on_game_finished(GameResult result);

        Player& get_active_player() {
            return players[active_player_id];
        }

        Player& get_non_active_player() {
            return players[1 - active_player_id];
        }

        const Player& get_active_player() const {
            return players[active_player_id];
        }

        const Player& get_non_active_player() const {
             return players[1 - active_player_id];
        }

        // Loop detection / State Identity
        uint64_t calculate_hash() const;

        // Instance Lookup Helper (O(1) owner check, O(N) zone scan)
        // Returns pointer to instance or nullptr if not found
        // Optimized by checking owner first
        CardInstance* get_card_instance(int instance_id) {
             if (instance_id < 0 || instance_id >= (int)card_owner_map.size()) return nullptr;

             PlayerID owner = card_owner_map[instance_id];
             if (owner > 1) return nullptr; // Invalid owner?

             Player& p = players[owner];
             // Check zones in order of likelihood
             for (auto& c : p.battle_zone) {
                 if (c.instance_id == instance_id) return &c;
                 for (auto& u : c.underlying_cards) if (u.instance_id == instance_id) return &u;
             }
             for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
             for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
             for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
             for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
             for (auto& c : p.deck) if (c.instance_id == instance_id) return &c;

             // Check effect buffers of BOTH players (card might be in either buffer)
             // Prioritize the owner's buffer, then opponent's buffer
             for (auto& c : players[0].effect_buffer) if (c.instance_id == instance_id) return &c;
             for (auto& c : players[1].effect_buffer) if (c.instance_id == instance_id) return &c;

             return nullptr;
        }

        const CardInstance* get_card_instance(int instance_id) const {
             if (instance_id < 0 || instance_id >= (int)card_owner_map.size()) return nullptr;

             PlayerID owner = card_owner_map[instance_id];
             if (owner > 1) return nullptr;

             const Player& p = players[owner];
             for (const auto& c : p.battle_zone) {
                 if (c.instance_id == instance_id) return &c;
                 for (const auto& u : c.underlying_cards) if (u.instance_id == instance_id) return &u;
             }
             for (const auto& c : p.hand) if (c.instance_id == instance_id) return &c;
             for (const auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
             for (const auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
             for (const auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
             for (const auto& c : p.deck) if (c.instance_id == instance_id) return &c;

             for (const auto& c : players[0].effect_buffer) if (c.instance_id == instance_id) return &c;
             for (const auto& c : players[1].effect_buffer) if (c.instance_id == instance_id) return &c;

             return nullptr;
        }
        
        // Error Handling [Q43, Q87]
        void panic(const char* message) const {
            throw std::runtime_error(message);
        }
    };
}
