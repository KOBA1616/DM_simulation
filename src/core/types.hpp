#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dm::core {

    // 3.2 カードデータ構造
    using CardID = uint16_t;
    using PlayerID = uint8_t; // 0 or 1

    // Bitmask for Civilization
    enum class Civilization : uint8_t {
        NONE = 0,
        LIGHT = 1 << 0,
        WATER = 1 << 1,
        DARKNESS = 1 << 2,
        FIRE = 1 << 3,
        NATURE = 1 << 4,
        ZERO = 1 << 5 // Colorless
    };

    inline Civilization operator|(Civilization a, Civilization b) {
        return static_cast<Civilization>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    inline Civilization operator&(Civilization a, Civilization b) {
        return static_cast<Civilization>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
    }

    enum class CardType : uint8_t {
        CREATURE,
        SPELL,
        EVOLUTION_CREATURE,
        CROSS_GEAR,
        CASTLE,
        PSYCHIC_CREATURE,
        GR_CREATURE
    };

    enum class Zone : uint8_t {
        DECK,
        HAND,
        MANA,
        BATTLE,
        GRAVEYARD,
        SHIELD,
        HYPER_SPATIAL,
        GR_DECK
    };

    enum class Phase : uint8_t {
        START_OF_TURN,
        DRAW,
        MANA,
        MAIN,
        ATTACK,
        BLOCK, // Added for blocking step
        END_OF_TURN
    };

    // Effect Types for Pending Stack
    enum class EffectType : uint8_t {
        NONE,
        CIP,               // Comes Into Play (登場時)
        AT_ATTACK,         // 攻撃時
        AT_BLOCK,          // ブロック時
        AT_START_OF_TURN,  // ターン開始時
        AT_END_OF_TURN,    // ターン終了時
        SHIELD_TRIGGER,    // S・トリガー
        G_STRIKE,          // G・ストライク
        DESTRUCTION        // 破壊時
    };
    
    // Result of a game
    enum class GameResult {
        NONE,
        P1_WIN,
        P2_WIN,
        DRAW
    };

}
