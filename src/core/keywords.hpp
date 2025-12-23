#pragma once
#include <string>
#include <map>
#include <optional>
#include <vector>

namespace dm::core {

#define FOREACH_KEYWORD(KEYWORD) \
    KEYWORD(G_ZERO, "g_zero") \
    KEYWORD(REVOLUTION_CHANGE, "revolution_change") \
    KEYWORD(MACH_FIGHTER, "mach_fighter") \
    KEYWORD(G_STRIKE, "g_strike") \
    KEYWORD(SPEED_ATTACKER, "speed_attacker") \
    KEYWORD(BLOCKER, "blocker") \
    KEYWORD(SLAYER, "slayer") \
    KEYWORD(DOUBLE_BREAKER, "double_breaker") \
    KEYWORD(TRIPLE_BREAKER, "triple_breaker") \
    KEYWORD(WORLD_BREAKER, "world_breaker") \
    KEYWORD(POWER_ATTACKER, "power_attacker") \
    KEYWORD(SHIELD_TRIGGER, "shield_trigger") \
    KEYWORD(EVOLUTION, "evolution") \
    KEYWORD(NEO, "neo") \
    KEYWORD(CIP, "cip") \
    KEYWORD(AT_ATTACK, "at_attack") \
    KEYWORD(AT_BLOCK, "at_block") \
    KEYWORD(AT_START_OF_TURN, "at_start_of_turn") \
    KEYWORD(AT_END_OF_TURN, "at_end_of_turn") \
    KEYWORD(DESTRUCTION, "destruction") \
    KEYWORD(JUST_DIVER, "just_diver") \
    KEYWORD(HYPER_ENERGY, "hyper_energy") \
    KEYWORD(META_COUNTER_PLAY, "meta_counter_play") \
    KEYWORD(SHIELD_BURN, "shield_burn") \
    KEYWORD(UNTAP_IN, "untap_in") \
    KEYWORD(UNBLOCKABLE, "unblockable") \
    KEYWORD(FRIEND_BURST, "friend_burst") \
    KEYWORD(EX_LIFE, "ex_life") \
    KEYWORD(MEGA_LAST_BURST, "mega_last_burst")

    enum class Keyword {
#define GENERATE_ENUM(ENUM, STRING) ENUM,
        FOREACH_KEYWORD(GENERATE_ENUM)
#undef GENERATE_ENUM
        COUNT
    };

    inline std::string keyword_to_string(Keyword k) {
        switch (k) {
#define GENERATE_STRING(ENUM, STRING) case Keyword::ENUM: return STRING;
            FOREACH_KEYWORD(GENERATE_STRING)
#undef GENERATE_STRING
            default: return "UNKNOWN";
        }
    }

    inline std::optional<Keyword> string_to_keyword(const std::string& str) {
        static const std::map<std::string, Keyword> map = {
#define GENERATE_MAP(ENUM, STRING) {STRING, Keyword::ENUM},
            FOREACH_KEYWORD(GENERATE_MAP)
#undef GENERATE_MAP
        };
        auto it = map.find(str);
        if (it != map.end()) return it->second;
        return std::nullopt;
    }

    // Helper to iterate all keywords
    inline std::vector<Keyword> get_all_keywords() {
        std::vector<Keyword> all;
        all.reserve(static_cast<size_t>(Keyword::COUNT));
        for (int i = 0; i < static_cast<int>(Keyword::COUNT); ++i) {
            all.push_back(static_cast<Keyword>(i));
        }
        return all;
    }

}
