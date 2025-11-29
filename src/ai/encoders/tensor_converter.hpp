#pragma once
#include "../../core/game_state.hpp"
#include <vector>

namespace dm::ai {

    class TensorConverter {
    public:
        // Returns the size of the input tensor
        static constexpr int INPUT_SIZE = 
            10 + // Global (Turn, Phase, etc)
            (20 + 6 + 20 * 3 + 1 + 20) + // Self: Hand(20), Mana(6 civs), Battle(20*3 props), Shield(1), Grave(20)
            (1 + 6 + 20 * 3 + 1 + 20);   // Opp: Hand(1), Mana(6 civs), Battle(20*3 props), Shield(1), Grave(20)
            // Note: Opponent Hand is masked (count only)
            // Battle props: ID, Tapped, Sickness
            // Mana: Light, Water, Darkness, Fire, Nature, Zero counts

        static std::vector<float> convert_to_tensor(const dm::core::GameState& game_state, int player_view);
    };

}
