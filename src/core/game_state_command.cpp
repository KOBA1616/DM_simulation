#include "core/game_state.hpp"
#include "engine/game_command/game_command.hpp"

namespace dm::core {

    void GameState::execute_command(std::shared_ptr<dm::engine::game_command::GameCommand> cmd) {
        if (!cmd) return;
        cmd->execute(*this);
        command_history.push_back(cmd);
    }

}
