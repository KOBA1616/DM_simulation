#pragma once

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/types.hpp"
#include <map>

namespace dm::engine::systems {

    class RestrictionSystem {
    public:
        static RestrictionSystem& instance() {
            static RestrictionSystem instance;
            return instance;
        }

        // Checks if a card play is forbidden (Gatekeeper for handle_play_card / resolve_play)
        bool is_play_forbidden(const core::GameState& state,
                               const core::CardInstance& card,
                               const core::CardDefinition& def,
                               const std::string& origin_zone,
                               const std::map<core::CardID, core::CardDefinition>& card_db);

        // Checks if an attack is forbidden (Gatekeeper for handle_attack)
        bool is_attack_forbidden(const core::GameState& state,
                                 const core::CardInstance& attacker,
                                 const core::CardDefinition& def,
                                 int target_id,
                                 const std::map<core::CardID, core::CardDefinition>& card_db);

        // Checks if a block is forbidden (Gatekeeper for handle_block)
        bool is_block_forbidden(const core::GameState& state,
                                const core::CardInstance& blocker,
                                const core::CardDefinition& def,
                                const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        RestrictionSystem() = default;
    };

}
