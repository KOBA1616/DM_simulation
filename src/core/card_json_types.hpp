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
        ON_BLOCK,
        AT_BREAK_SHIELD,
        NONE
    };

    enum class ReactionType {
        NONE,
        NINJA_STRIKE,
        STRIKE_BACK,
        REVOLUTION_0_TRIGGER
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
        COUNT_CARDS,
        GET_GAME_STAT,
        APPLY_MODIFIER,
        REVEAL_CARDS,
        REGISTER_DELAYED_EFFECT,
        RESET_INSTANCE,
        SEARCH_DECK,
        SHUFFLE_DECK,
        ADD_SHIELD,
        SEND_SHIELD_TO_GRAVE,
        SEND_TO_DECK_BOTTOM,
        MOVE_TO_UNDER_CARD,
        SELECT_NUMBER,
        FRIEND_BURST, // Added: Friend Burst
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

        // Step 3-1: Advanced Selection
        std::optional<std::string> selection_mode; // "MIN", "MAX", "RANDOM"
        std::optional<std::string> selection_sort_key; // "COST", "POWER"

        // Step 3-2: Composite Filters
        std::vector<FilterDef> and_conditions;
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
        // Phase 5: Variable Linking
        std::string input_value_key;
        std::string output_value_key;
        // Step 3-3: Negative Selection
        bool inverse_target = false;
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

    struct ReactionCondition {
        std::string trigger_event; // "ON_BLOCK_OR_ATTACK", "ON_SHIELD_ADD"
        bool civilization_match = false;
        int mana_count_min = 0;
        bool same_civilization_shield = false;
    };

    struct ReactionAbility {
        ReactionType type = ReactionType::NONE;
        int cost = 0;
        std::string zone;
        ReactionCondition condition;
    };

    struct CardData {
        int id;
        std::string name;
        int cost;
        std::vector<std::string> civilizations;
        int power;
        std::string type;
        std::vector<std::string> races;
        std::vector<EffectDef> effects;
        std::optional<FilterDef> revolution_change_condition;
        std::optional<std::map<std::string, bool>> keywords;
        std::vector<ReactionAbility> reaction_abilities;
        std::shared_ptr<CardData> spell_side;

        // AI Metadata
        bool is_key_card = false;
        int ai_importance_score = 0;
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

    // Custom serializer for shared_ptr
    template <typename T>
    struct adl_serializer<std::shared_ptr<T>> {
        static void to_json(json& j, const std::shared_ptr<T>& opt) {
            if (!opt) {
                j = nullptr;
            } else {
                j = *opt;
            }
        }

        static void from_json(const json& j, std::shared_ptr<T>& opt) {
            if (j.is_null()) {
                opt = nullptr;
            } else {
                opt = std::make_shared<T>(j.get<T>());
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
        {TriggerType::ON_ATTACK_FROM_HAND, "ON_ATTACK_FROM_HAND"},
        {TriggerType::ON_BLOCK, "ON_BLOCK"},
        {TriggerType::AT_BREAK_SHIELD, "AT_BREAK_SHIELD"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(ReactionType, {
        {ReactionType::NONE, "NONE"},
        {ReactionType::NINJA_STRIKE, "NINJA_STRIKE"},
        {ReactionType::STRIKE_BACK, "STRIKE_BACK"},
        {ReactionType::REVOLUTION_0_TRIGGER, "REVOLUTION_0_TRIGGER"}
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
        {EffectActionType::REVOLUTION_CHANGE, "REVOLUTION_CHANGE"},
        {EffectActionType::COUNT_CARDS, "COUNT_CARDS"},
        {EffectActionType::GET_GAME_STAT, "GET_GAME_STAT"},
        {EffectActionType::APPLY_MODIFIER, "APPLY_MODIFIER"},
        {EffectActionType::REVEAL_CARDS, "REVEAL_CARDS"},
        {EffectActionType::REGISTER_DELAYED_EFFECT, "REGISTER_DELAYED_EFFECT"},
        {EffectActionType::RESET_INSTANCE, "RESET_INSTANCE"},
        {EffectActionType::SEARCH_DECK, "SEARCH_DECK"},
        {EffectActionType::SHUFFLE_DECK, "SHUFFLE_DECK"},
        {EffectActionType::ADD_SHIELD, "ADD_SHIELD"},
        {EffectActionType::SEND_SHIELD_TO_GRAVE, "SEND_SHIELD_TO_GRAVE"},
        {EffectActionType::SEND_TO_DECK_BOTTOM, "SEND_TO_DECK_BOTTOM"},
        {EffectActionType::MOVE_TO_UNDER_CARD, "MOVE_TO_UNDER_CARD"},
        {EffectActionType::SELECT_NUMBER, "SELECT_NUMBER"},
        {EffectActionType::FRIEND_BURST, "FRIEND_BURST"}
    })

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(FilterDef, owner, zones, types, civilizations, races, min_cost, max_cost, min_power, max_power, is_tapped, is_blocker, is_evolution, count, selection_mode, selection_sort_key, and_conditions)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ActionDef, type, scope, filter, value1, value2, str_val, value, optional, target_player, source_zone, destination_zone, target_choice, input_value_key, output_value_key, inverse_target)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ConditionDef, type, value, str_val)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EffectDef, trigger, condition, actions)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionCondition, trigger_event, civilization_match, mana_count_min, same_civilization_shield)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionAbility, type, cost, zone, condition)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CardData, id, name, cost, civilizations, power, type, races, effects, revolution_change_condition, keywords, reaction_abilities, spell_side, is_key_card, ai_importance_score)
}
