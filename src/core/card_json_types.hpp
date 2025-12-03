#pragma once
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "core/constants.hpp"

namespace dm::core {

    // Enums for JSON mapping
    enum class TriggerType {
        ON_PLAY,
        ON_ATTACK,
        ON_DESTROY,
        S_TRIGGER,
        TURN_START,
        PASSIVE_CONST,
        ON_OTHER_ENTER,
        ON_ATTACK_FROM_HAND,
        NONE
    };

    enum class TargetScope {
        SELF,
        PLAYER_SELF,
        PLAYER_OPPONENT,
        ALL_PLAYERS,
        TARGET_SELECT,
        RANDOM,
        ALL_FILTERED,
        NONE
    };

    enum class EffectActionType {
        DRAW_CARD,
        ADD_MANA,
        DESTROY,
        RETURN_TO_HAND,
        SEND_TO_MANA,
        TAP,
        UNTAP,
        MODIFY_POWER,
        BREAK_SHIELD,
        LOOK_AND_ADD,
        SUMMON_TOKEN,
        SEARCH_DECK_BOTTOM,
        MEKRAID,
        DISCARD,
        PLAY_FROM_ZONE,
        COST_REFERENCE,
        LOOK_TO_BUFFER,
        SELECT_FROM_BUFFER,
        PLAY_FROM_BUFFER,
        MOVE_BUFFER_TO_ZONE,
        REVOLUTION_CHANGE,
        NONE
    };

    // JSON Structures
    struct FilterDef {
        std::optional<std::string> owner; // "SELF", "OPPONENT", "BOTH"
        std::vector<std::string> zones;   // "BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE"
        std::vector<std::string> types;   // "CREATURE", "SPELL"
        std::vector<std::string> civilizations;
        std::vector<std::string> races;
        std::optional<int> min_cost;
        std::optional<int> max_cost;
        std::optional<int> min_power;
        std::optional<int> max_power;
        std::optional<bool> is_tapped;
        std::optional<bool> is_blocker;
        std::optional<bool> is_evolution;
        std::optional<int> count; // For selection
    };

    struct ActionDef {
        EffectActionType type = EffectActionType::NONE;
        TargetScope scope = TargetScope::NONE;
        FilterDef filter;
        int value1 = 0;
        int value2 = 0;
        std::string str_val;
        std::string value; // Legacy compat
        bool optional = false;
        std::string target_player;
        std::string source_zone;
        std::string destination_zone;
        std::string target_choice; // Legacy compat
    };

    struct ConditionDef {
        std::string type; // "NONE", "MANA_ARMED", "SHIELD_COUNT"
        int value = 0;
        std::string str_val;
    };

    struct EffectDef {
        TriggerType trigger = TriggerType::NONE;
        ConditionDef condition;
        std::vector<ActionDef> actions;
    };

    struct CardData {
        int id;
        std::string name;
        int cost;
        std::string civilization;
        int power;
        std::string type;
        std::vector<std::string> races;
        std::vector<EffectDef> effects;
    };

} // namespace dm::core

// JSON Serialization Macros & Helpers
namespace nlohmann {
    template <typename T>
    struct adl_serializer<std::optional<T>> {
        static void to_json(json& j, const std::optional<T>& opt) {
            if (opt == std::nullopt) {
                j = nullptr;
            } else {
                j = *opt;
            }
        }

        static void from_json(const json& j, std::optional<T>& opt) {
            if (j.is_null()) {
                opt = std::nullopt;
            } else {
                opt = j.get<T>();
            }
        }
    };
}

namespace dm::core {
    NLOHMANN_JSON_SERIALIZE_ENUM(TriggerType, {
        {TriggerType::NONE, "NONE"},
        {TriggerType::ON_PLAY, "ON_PLAY"},
        {TriggerType::ON_ATTACK, "ON_ATTACK"},
        {TriggerType::ON_DESTROY, "ON_DESTROY"},
        {TriggerType::S_TRIGGER, "S_TRIGGER"},
        {TriggerType::TURN_START, "TURN_START"},
        {TriggerType::PASSIVE_CONST, "PASSIVE_CONST"},
        {TriggerType::ON_OTHER_ENTER, "ON_OTHER_ENTER"},
        {TriggerType::ON_ATTACK_FROM_HAND, "ON_ATTACK_FROM_HAND"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(TargetScope, {
        {TargetScope::NONE, "NONE"},
        {TargetScope::SELF, "SELF"},
        {TargetScope::PLAYER_SELF, "PLAYER_SELF"},
        {TargetScope::PLAYER_OPPONENT, "PLAYER_OPPONENT"},
        {TargetScope::ALL_PLAYERS, "ALL_PLAYERS"},
        {TargetScope::TARGET_SELECT, "TARGET_SELECT"},
        {TargetScope::RANDOM, "RANDOM"},
        {TargetScope::ALL_FILTERED, "ALL_FILTERED"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(EffectActionType, {
        {EffectActionType::NONE, "NONE"},
        {EffectActionType::DRAW_CARD, "DRAW_CARD"},
        {EffectActionType::ADD_MANA, "ADD_MANA"},
        {EffectActionType::DESTROY, "DESTROY"},
        {EffectActionType::RETURN_TO_HAND, "RETURN_TO_HAND"},
        {EffectActionType::SEND_TO_MANA, "SEND_TO_MANA"},
        {EffectActionType::TAP, "TAP"},
        {EffectActionType::UNTAP, "UNTAP"},
        {EffectActionType::MODIFY_POWER, "MODIFY_POWER"},
        {EffectActionType::BREAK_SHIELD, "BREAK_SHIELD"},
        {EffectActionType::LOOK_AND_ADD, "LOOK_AND_ADD"},
        {EffectActionType::SUMMON_TOKEN, "SUMMON_TOKEN"},
        {EffectActionType::SEARCH_DECK_BOTTOM, "SEARCH_DECK_BOTTOM"},
        {EffectActionType::MEKRAID, "MEKRAID"},
        {EffectActionType::DISCARD, "DISCARD"},
        {EffectActionType::PLAY_FROM_ZONE, "PLAY_FROM_ZONE"},
        {EffectActionType::COST_REFERENCE, "COST_REFERENCE"},
        {EffectActionType::LOOK_TO_BUFFER, "LOOK_TO_BUFFER"},
        {EffectActionType::SELECT_FROM_BUFFER, "SELECT_FROM_BUFFER"},
        {EffectActionType::PLAY_FROM_BUFFER, "PLAY_FROM_BUFFER"},
        {EffectActionType::MOVE_BUFFER_TO_ZONE, "MOVE_BUFFER_TO_ZONE"},
        {EffectActionType::REVOLUTION_CHANGE, "REVOLUTION_CHANGE"}
    })

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(FilterDef, owner, zones, types, civilizations, races, min_cost, max_cost, min_power, max_power, is_tapped, is_blocker, is_evolution, count)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ActionDef, type, scope, filter, value1, value2, str_val, value, optional, target_player, source_zone, destination_zone, target_choice)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ConditionDef, type, value, str_val)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EffectDef, trigger, condition, actions)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CardData, id, name, cost, civilization, power, type, races, effects)
}
