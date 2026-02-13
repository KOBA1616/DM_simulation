#ifndef DM_ENGINE_SYSTEMS_DECISION_MAKER_HPP
#define DM_ENGINE_SYSTEMS_DECISION_MAKER_HPP

#include <vector>
#include <map>
#include <string>

namespace dm::core {
    class GameState;
    struct CardInstance;
}

namespace dm::engine::game_command {
    struct CommandDef;
}

namespace dm::engine::systems {

    class DecisionMaker {
    public:
        virtual ~DecisionMaker() = default;

        /**
         * @brief Select targets for a command from a list of candidates.
         * 
         * @param state Current game state
         * @param cmd The command definition requiring selection
         * @param candidates List of candidate card instance IDs
         * @param amount Number of targets to select
         * @return std::vector<int> Selected card instance IDs (must be subset of candidates)
         */
        virtual std::vector<int> select_targets(
            const dm::core::GameState& state, 
            const dm::engine::game_command::CommandDef& cmd, 
            const std::vector<int>& candidates, 
            int amount
        ) = 0;
    };

}

#endif // DM_ENGINE_SYSTEMS_DECISION_MAKER_HPP
