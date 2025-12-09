#pragma once
#include <cstdint>

namespace dm::core {

    // 3.1 Constants (constants.hpp)
    constexpr int MAX_HAND_SIZE = 20;
    constexpr int MAX_BATTLE_SIZE = 20;
    constexpr int MAX_MANA_SIZE = 20;
    constexpr int MAX_GRAVE_SEARCH = 20; // Tensor入力用
    constexpr int TURN_LIMIT = 100; // Forced draw-loss limit
    constexpr int POWER_INFINITY = 32000;

    // Action Space Dimensions (approximate based on 5.2)
    constexpr int ACTION_MANA_SIZE = 20;
    constexpr int ACTION_PLAY_SIZE = 20;
    constexpr int ACTION_ATTACK_SIZE = 420;
    constexpr int ACTION_BLOCK_SIZE = 20;
    constexpr int ACTION_SELECT_TARGET_SIZE = 100; // 100+ in spec
    constexpr int ACTION_PASS_SIZE = 1;

}
