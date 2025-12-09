#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dm::core {

    // 3.2 繧ｫ繝ｼ繝峨ョ繝ｼ繧ｿ讒矩
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
        GR_CREATURE,
        TAMASEED
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

    // SpawnSource: How a card is being put into the Battle Zone
    enum class SpawnSource : uint8_t {
        HAND_SUMMON,    // Normal summon from hand (includes G-Zero, Speed Attacker logic checks)
        EFFECT_SUMMON,  // Summon via effect (S-Trigger, Mekraid, etc.)
        EFFECT_PUT      // Put directly into battle zone (Reanimate, etc.)
    };

    // Effect Types for Pending Stack
    enum class EffectType : uint8_t {
        NONE,
        CIP,               // Comes Into Play (逋ｻ蝣ｴ譎・
        AT_ATTACK,         // 謾ｻ謦・凾
        AT_BLOCK,          // 繝悶Ο繝・け譎・
        AT_START_OF_TURN,  // 繧ｿ繝ｼ繝ｳ髢句ｧ区凾
        AT_END_OF_TURN,    // 繧ｿ繝ｼ繝ｳ邨ゆｺ・凾
        SHIELD_TRIGGER,    // S繝ｻ繝医Μ繧ｬ繝ｼ
        G_STRIKE,          // G繝ｻ繧ｹ繝医Λ繧､繧ｯ
        DESTRUCTION,       // 遐ｴ螢頑凾
        ON_ATTACK_FROM_HAND, // Revolution Change
        INTERNAL_PLAY,     // For stacking play actions (Gatekeeper)
        META_COUNTER,      // For Meta Counter (Counter play at end of turn)
        RESOLVE_BATTLE,    // For pending battle resolution
        BREAK_SHIELD,      // For pending shield break
        REACTION_WINDOW,   // For Ninja Strike / Strike Back reaction windows
        TRIGGER_ABILITY    // Generic queued trigger (new Stack System)
    };
    
    // Result of a game
    enum class GameResult {
        NONE,
        P1_WIN,
        P2_WIN,
        DRAW
    };

    // Resolve Type for Pending Effects
    enum class ResolveType : uint8_t {
        NONE,
        TARGET_SELECT,
        EFFECT_RESOLUTION
    };

    // Forward declaration for CostModifier
    struct FilterDef;

}
