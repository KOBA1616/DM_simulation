#pragma once
#include "engine/game_command/game_command.hpp"
#include "core/types.hpp"

namespace dm::engine::game_command {

    class TransitionCommand : public GameCommand {
    public:
        TransitionCommand(dm::core::CardID card_id, int instance_id, dm::core::Zone from_zone, dm::core::Zone to_zone, dm::core::PlayerID player_id, int from_index = -1, int to_index = -1);

        void execute(dm::core::GameState& state) override;
        void undo(dm::core::GameState& state) override;
        CommandType get_type() const override { return CommandType::TRANSITION; }
        std::string to_string() const override;

    private:
        dm::core::CardID card_id;
        int instance_id;
        dm::core::Zone from_zone;
        dm::core::Zone to_zone;
        dm::core::PlayerID player_id;
        int from_index;
        int to_index;
    };

}
