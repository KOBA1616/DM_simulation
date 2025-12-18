#pragma once
#include "core/game_state.hpp"
#include <memory>

namespace dm::engine::game_command {

    enum class CommandType {
        TRANSITION,
        MUTATE,
        FLOW,
        QUERY,
        DECIDE,
        DECLARE_REACTION,
        STAT,
        GAME_RESULT,
        ATTACH
    };

    class GameCommand {
    public:
        virtual ~GameCommand() = default;

        // Execute the command, modifying the game state
        virtual void execute(core::GameState& state) = 0;

        // Invert the command to support rollback/undo
        // This should reverse the changes made by execute()
        virtual void invert(core::GameState& state) = 0;

        virtual CommandType get_type() const = 0;
    };

}
