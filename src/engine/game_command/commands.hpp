#pragma once
#include "game_command.hpp"
#include <string>
#include <vector>
#include <map>

namespace dm::engine::game_command {

    // TransitionCommand: Moves a card from one zone to another
    class TransitionCommand : public GameCommand {
    public:
        // Full Constructor
        TransitionCommand(int instance_id, int source_player, int source_zone,
                      int dest_player, int dest_zone, int dest_index);

        // Legacy/Simplified Constructor
        TransitionCommand(int instance_id, int source_zone, int dest_zone, int player_id, int dest_index);

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::TRANSITION; }

        int get_instance_id() const { return instance_id_; }
        int get_dest_zone() const { return dest_zone_; }

    private:
        int instance_id_;
        int source_player_;
        int source_zone_;
        int dest_player_;
        int dest_zone_;
        int dest_index_;

        int previous_index_;
        bool was_tapped_ = false;
        bool was_face_down_ = false;
        int previous_owner_ = -1;
    };

    // MutateCommand: Changes a property of a game object
    class MutateCommand : public GameCommand {
    public:
        enum class MutationType {
            TAP,
            UNTAP,
            POWER_MOD,
            BREAK_SHIELD,
            ADD_MODIFIER,
            ADD_PASSIVE,
            ADD_KEYWORD,
            REMOVE_KEYWORD
        };

        // Standard Constructor (Value + Duration). Default value=0 allows (id, type) calls.
        MutateCommand(int target_id, MutationType type, int value = 0, int duration = 0);

        // String Payload Constructor
        MutateCommand(int target_id, MutationType type, int value, const std::string& str_value);

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::MUTATE; }

    private:
        int target_id_;
        MutationType type_;
        int value_;
        int duration_;
        std::string str_value_;

        int previous_value_ = 0;
        bool previous_bool_ = false;
    };

    // FlowCommand: Controls game flow
    class FlowCommand : public GameCommand {
    public:
        enum class FlowType {
            PHASE_CHANGE,
            NEXT_TURN,
            GAME_OVER,
            SET_ATTACK_SOURCE,
            SET_ATTACK_TARGET,
            SET_ATTACK_PLAYER
        };

        FlowCommand(FlowType type, int next_value);
        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::FLOW; }

    private:
        FlowType type_;
        int next_value_;

        int prev_value_ = 0;
        int prev_phase_ = 0;
        int prev_turn_ = 0;
        int prev_active_player_ = 0;
        int prev_winner_ = 0;
    };

    // QueryCommand: Requests input
    class QueryCommand : public GameCommand {
    public:
        QueryCommand(const std::string& query_type, const std::map<std::string, int>& params, const std::vector<int>& valid_targets);
        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::QUERY; }

    private:
        std::string query_type_;
        std::map<std::string, int> params_;
        std::vector<int> valid_targets_;
    };

    // DecideCommand: Records input
    class DecideCommand : public GameCommand {
    public:
        DecideCommand(int query_id, const std::vector<int>& selected_indices);
        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::DECIDE; }

    private:
        int query_id_;
        std::vector<int> selected_indices_;
    };

}
