#pragma once
#include "game_command.hpp"
#include "core/types.hpp"
#include "core/modifiers.hpp"
#include "core/pending_effect.hpp"
#include "engine/systems/trigger_system/reaction_window.hpp"

namespace dm::engine::game_command {

    class TransitionCommand : public GameCommand {
    public:
        int card_instance_id;
        core::Zone from_zone;
        core::Zone to_zone;
        core::PlayerID owner_id;
        int destination_index; // -1 for append

        // Context to restore exact position in from_zone for undo
        int original_index;

        TransitionCommand(int instance_id, core::Zone from, core::Zone to, core::PlayerID owner, int dest_idx = -1)
            : card_instance_id(instance_id), from_zone(from), to_zone(to), owner_id(owner), destination_index(dest_idx), original_index(-1) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::TRANSITION; }
    };

    class MutateCommand : public GameCommand {
    public:
        enum class MutationType {
            TAP,
            UNTAP,
            POWER_MOD,
            ADD_KEYWORD,
            REMOVE_KEYWORD,
            ADD_PASSIVE_EFFECT,
            ADD_COST_MODIFIER,
            ADD_PENDING_EFFECT
        };

        int target_instance_id;
        MutationType mutation_type;
        int int_value; // Power value or simple int param
        std::string str_value; // Keyword string etc.

        // Extended payloads for modifiers
        std::optional<core::PassiveEffect> passive_effect;
        std::optional<core::CostModifier> cost_modifier;
        std::optional<core::PendingEffect> pending_effect;

        // Undo context
        int previous_int_value;
        bool previous_bool_value;

        MutateCommand(int instance_id, MutationType type, int val = 0, std::string str = "")
            : target_instance_id(instance_id), mutation_type(type), int_value(val), str_value(str), previous_int_value(0), previous_bool_value(false) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::MUTATE; }
    };

    // B. Complex Card Processing (Attach/Evolution)
    class AttachCommand : public GameCommand {
    public:
        int card_to_attach_id;   // The card moving (e.g. from Hand)
        int target_base_card_id; // The card on the field being evolved/cross-geared

        // We assume attachment means "put on top" (Evolution) or "under" (Soul)?
        // Evolution puts the new card ON TOP. The target becomes UNDER.
        // But `underlying_cards` vector is usually "cards under this instance".
        // If we Evolve A onto B:
        // A is the new top instance. B is removed from Battle Zone and added to A.underlying_cards.
        // Wait, Instance ID persistence?
        // If A enters play, it gets a new Instance ID.
        // B already has Instance ID X.
        // If we replace B with A... B is gone (from zone). A is in zone.
        // B is inside A.
        // BUT, usually Evolution Creature keeps the state (tapped/untapped) of the base?
        // "Evolution creature is put on top".
        // So we might want to keep the Instance ID of the base, and just change the Card Definition?
        // No, that's messy.
        // Better: A is a new card entering.
        // B is moved to "Under A".
        // A inherits B's state (tapped/untapped, summoning sickness status).

        core::Zone source_zone; // Where A comes from (usually Hand)

        // Undo State
        core::Zone original_zone;
        int original_zone_index;
        bool target_was_tapped;
        bool target_was_sick;

        AttachCommand(int attach_id, int base_id, core::Zone src_zone)
            : card_to_attach_id(attach_id), target_base_card_id(base_id), source_zone(src_zone),
              original_zone(core::Zone::HAND), original_zone_index(-1), target_was_tapped(false), target_was_sick(true) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::ATTACH; }
    };

    class FlowCommand : public GameCommand {
    public:
        enum class FlowType {
            PHASE_CHANGE,
            TURN_CHANGE,
            STEP_CHANGE, // Future use
            SET_ATTACK_SOURCE,
            SET_ATTACK_TARGET,
            SET_ATTACK_PLAYER,
            SET_ACTIVE_PLAYER
        };

        FlowType flow_type;
        int new_value; // Phase enum or Turn number

        // Undo context
        int previous_value;

        FlowCommand(FlowType type, int val)
            : flow_type(type), new_value(val), previous_value(0) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::FLOW; }
    };

    class QueryCommand : public GameCommand {
    public:
        std::string query_type;
        std::vector<int> valid_targets;
        std::map<std::string, int> params;

        QueryCommand(std::string type, std::vector<int> targets = {}, std::map<std::string, int> p = {})
            : query_type(type), valid_targets(targets), params(p) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override; // Clears the query
        CommandType get_type() const override { return CommandType::QUERY; }
    };

    class DecideCommand : public GameCommand {
    public:
        int query_id; // To match the query
        std::vector<int> selected_indices; // or instance_ids
        int selected_option_index;

        // Undo context
        bool was_waiting;
        std::optional<core::GameState::QueryContext> previous_query;

        DecideCommand(int q_id, std::vector<int> selection = {}, int option = -1)
            : query_id(q_id), selected_indices(selection), selected_option_index(option), was_waiting(false) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::DECIDE; }
    };

    class DeclareReactionCommand : public GameCommand {
    public:
        bool pass; // True if player passes (uses nothing)
        int reaction_index; // Index in ReactionWindow::candidates, -1 if pass
        core::PlayerID player_id;

        // Undo context
        bool was_waiting;
        core::GameState::Status previous_status;
        std::vector<dm::engine::systems::ReactionWindow> previous_stack;

        DeclareReactionCommand(core::PlayerID pid, bool is_pass, int idx = -1)
            : pass(is_pass), reaction_index(idx), player_id(pid), was_waiting(false), previous_status(core::GameState::Status::PLAYING) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::DECLARE_REACTION; }
    };

    class StatCommand : public GameCommand {
    public:
        enum class StatType {
            CARDS_DRAWN,
            CARDS_DISCARDED,
            CREATURES_PLAYED,
            SPELLS_CAST
        };
        StatType stat;
        int amount;

        // Undo context
        int previous_value;

        StatCommand(StatType s, int amt) : stat(s), amount(amt), previous_value(0) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::STAT; }
    };

    class GameResultCommand : public GameCommand {
    public:
        core::GameResult result;

        // Undo context
        core::GameResult previous_result;

        GameResultCommand(core::GameResult res) : result(res), previous_result(core::GameResult::NONE) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::GAME_RESULT; }
    };

}
