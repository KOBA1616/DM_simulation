#pragma once

#include <string>
#include <vector>
#include <map>

// Forward includes from core to match IntentGenerator signature
#include "src/core/action.hpp"
#include "src/core/card_json_types.hpp"
#include "src/core/game_state.hpp"

namespace engine {
namespace commands {

struct CommandDef {
    std::string type;
    std::string uid;
    int instance_id = -1;
    int source_instance_id = -1;
    int target_instance_id = -1;
    int owner_id = -1;
    std::string from_zone;
    std::string to_zone;
    int amount = 0;
    bool optional = false;
    bool up_to = false;
    std::string str_param;
};

class CommandGenerator {
public:
    CommandGenerator() = default;

    std::vector<CommandDef> generate_commands(
        const dm::core::GameState& game_state,
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
    );
};

} // namespace commands
} // namespace engine
#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include <map>

namespace dm::engine {

    class CommandGenerator {
    public:
        // Produce a vector of core::CommandDef from the given game state and card DB.
        // This v1 implementation bridges by calling the legacy IntentGenerator (actions)
        // and mapping Action -> CommandDef. Later versions should implement direct
        // command-first generation.
        static std::vector<dm::core::CommandDef> generate_legal_commands(
            const dm::core::GameState& game_state,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );
    };

}
