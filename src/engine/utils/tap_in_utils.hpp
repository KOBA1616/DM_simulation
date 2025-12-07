#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"

namespace dm::engine {
    class TapInUtils {
    public:
        static void apply_tap_in_rule(dm::core::CardInstance& card, const dm::core::CardDefinition& def) {
             // Multi-color tap-in logic
             if (def.civilizations.size() > 1) {
                 if (!def.keywords.untap_in) {
                     card.is_tapped = true;
                 }
             }
        }
    };
}
