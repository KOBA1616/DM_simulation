#include "core/game_state.hpp"
#include "engine/game_command/game_command.hpp"
#include <iostream>

namespace dm::core {

    void GameState::execute_command(std::shared_ptr<dm::engine::game_command::GameCommand> cmd) {
        if (!cmd) return;
        cmd->execute(*this);
        command_history.push_back(cmd);
    }

    void GameState::undo_last_command() {
        if (command_history.empty()) return;
        auto cmd = command_history.back();
        cmd->invert(*this);
        command_history.pop_back();
    }

}
