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
        ON_SHIELD_ADD, // Added: ON_SHIELD_ADD
        ON_CAST_SPELL, // Added: ON_CAST_SPELL
        ON_OPPONENT_DRAW, // Added: ON_OPPONENT_DRAW
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

    NLOHMANN_JSON_SERIALIZE_ENUM(Civilization, {
        {Civilization::NONE, "NONE"},
        {Civilization::LIGHT, "LIGHT"},
        {Civilization::WATER, "WATER"},
        {Civilization::DARKNESS, "DARKNESS"},
        {Civilization::FIRE, "FIRE"},
        {Civilization::NATURE, "NATURE"},
        {Civilization::ZERO, "ZERO"}
    })

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
        FRIEND_BURST,
        GRANT_KEYWORD,
        MOVE_CARD,
        CAST_SPELL,     // Added: CAST_SPELL
        PUT_CREATURE,   // Added: PUT_CREATURE
        SELECT_OPTION,  // Added: SELECT_OPTION
        RESOLVE_BATTLE, // Added: RESOLVE_BATTLE for engine compatibility
        NONE
    };

    // Phase 4: Cost System Enums
    enum class CostType {
        MANA,               // Standard Mana Payment
        TAP_CARD,           // Tap cards in specified zone
        SACRIFICE_CARD,     // Send cards to graveyard
        RETURN_CARD,        // Return cards to hand
        SHIELD_BURN,        // Send shields to graveyard
        DISCARD             // Discard cards from hand
    };

    enum class ReductionType {
        PASSIVE,        // Passive reduction (e.g. -1 for each Dragon)
        ACTIVE_PAYMENT  // Active payment reduction (e.g. Tap creature to reduce by 2)
    };

    // JSON Structures
    struct FilterDef {
        std::optional<std::string> owner; // "SELF", "OPPONENT", "BOTH"
        std::vector<std::string> zones;   // "BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE"
        std::vector<std::string> types;   // "CREATURE", "SPELL"
        std::vector<Civilization> civilizations;
        std::vector<std::string> races;
        std::optional<int> min_cost;
        std::optional<int> max_cost;
        std::optional<int> min_power;
        std::optional<int> max_power;
        std::optional<bool> is_tapped;
        std::optional<bool> is_blocker;
        std::optional<bool> is_evolution;
        std::optional<bool> is_card_designation; // New field for card vs element distinction
        std::optional<int> count; // For selection

        // Step 3-1: Advanced Selection
        std::optional<std::string> selection_mode; // "MIN", "MAX", "RANDOM"
        std::optional<std::string> selection_sort_key; // "COST", "POWER"

        // Step 5.2.3: Context Reference
        std::optional<std::string> power_max_ref; // e.g., "EVENT_CONTEXT_POWER"

        // Step 3-2: Composite Filters
        std::vector<FilterDef> and_conditions;
    };

    // Phase 4: Cost System Structs
    struct CostDef {
        CostType type;
        int amount;
        FilterDef filter;
        bool is_optional = false;
        std::string cost_id;
    };

    struct CostReductionDef {
        ReductionType type;

        // ACTIVE_PAYMENT specific
        CostDef unit_cost;
        int reduction_amount;
        int max_units = -1; // -1 for unlimited
        int min_mana_cost = 0;

        std::string name;
    };

    struct ConditionDef {
        std::string type; // "NONE", "MANA_ARMED", "SHIELD_COUNT", "COMPARE_STAT", "EVENT_FILTER_MATCH"
        int value = 0;
        std::string str_val;
        // Condition Generalization
        std::string stat_key; // e.g. "OPPONENT_HAND_COUNT"
        std::string op; // ">", "=", "<"
        // Step 3-1: Trigger Inclusion
        std::optional<FilterDef> filter;
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
        // Step 3-1: Conditional Actions
        std::optional<ConditionDef> condition;
        // Step 3.1: Mode Selection (Nested Actions)
        std::vector<std::vector<ActionDef>> options;
        // Twinpact Support
        bool cast_spell_side = false;
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
        std::vector<Civilization> civilizations;
        int power;
        std::string type;
        std::vector<std::string> races;
        std::vector<EffectDef> effects;
        std::vector<EffectDef> metamorph_abilities; // Ultra Soul Cross abilities
        std::optional<FilterDef> revolution_change_condition;
        std::optional<std::map<std::string, bool>> keywords;
        std::vector<ReactionAbility> reaction_abilities;
        std::vector<CostReductionDef> cost_reductions; // Phase 4
        std::shared_ptr<CardData> spell_side;

        // AI Metadata
        bool is_key_card = false;
        int ai_importance_score = 0;
    };

    // Forward declarations
    void to_json(nlohmann::json& j, const CardData& c);
    void from_json(const nlohmann::json& j, CardData& c);

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
        {TriggerType::ON_ATTACK_FROM_HAND, "ON_ATTACK_FROM_HAND"},
        {TriggerType::ON_BLOCK, "ON_BLOCK"},
        {TriggerType::AT_BREAK_SHIELD, "AT_BREAK_SHIELD"},
        {TriggerType::ON_SHIELD_ADD, "ON_SHIELD_ADD"},
        {TriggerType::ON_CAST_SPELL, "ON_CAST_SPELL"},
        {TriggerType::ON_OPPONENT_DRAW, "ON_OPPONENT_DRAW"}
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
        {EffectActionType::FRIEND_BURST, "FRIEND_BURST"},
        {EffectActionType::GRANT_KEYWORD, "GRANT_KEYWORD"},
        {EffectActionType::MOVE_CARD, "MOVE_CARD"},
        {EffectActionType::CAST_SPELL, "CAST_SPELL"},
        {EffectActionType::PUT_CREATURE, "PUT_CREATURE"},
        {EffectActionType::SELECT_OPTION, "SELECT_OPTION"},
        {EffectActionType::RESOLVE_BATTLE, "RESOLVE_BATTLE"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(CostType, {
        {CostType::MANA, "MANA"},
        {CostType::TAP_CARD, "TAP_CARD"},
        {CostType::SACRIFICE_CARD, "SACRIFICE_CARD"},
        {CostType::RETURN_CARD, "RETURN_CARD"},
        {CostType::SHIELD_BURN, "SHIELD_BURN"},
        {CostType::DISCARD, "DISCARD"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(ReductionType, {
        {ReductionType::PASSIVE, "PASSIVE"},
        {ReductionType::ACTIVE_PAYMENT, "ACTIVE_PAYMENT"}
    })

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(FilterDef, owner, zones, types, civilizations, races, min_cost, max_cost, min_power, max_power, is_tapped, is_blocker, is_evolution, is_card_designation, count, selection_mode, selection_sort_key, power_max_ref, and_conditions)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ConditionDef, type, value, str_val, stat_key, op, filter)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ActionDef, type, scope, filter, value1, value2, str_val, value, optional, target_player, source_zone, destination_zone, target_choice, input_value_key, output_value_key, inverse_target, condition, options, cast_spell_side)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EffectDef, trigger, condition, actions)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionCondition, trigger_event, civilization_match, mana_count_min, same_civilization_shield)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionAbility, type, cost, zone, condition)

    // Phase 4: Cost System Serialization
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CostDef, type, amount, filter, is_optional, cost_id)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CostReductionDef, type, unit_cost, reduction_amount, max_units, min_mana_cost, name)

    inline void to_json(nlohmann::json& j, const CardData& c) {
        j = nlohmann::json{
            {"id", c.id},
            {"name", c.name},
            {"cost", c.cost},
            {"civilizations", c.civilizations},
            {"power", c.power},
            {"type", c.type},
            {"races", c.races},
            {"effects", c.effects},
            {"metamorph_abilities", c.metamorph_abilities},
            {"revolution_change_condition", c.revolution_change_condition},
            {"keywords", c.keywords},
            {"reaction_abilities", c.reaction_abilities},
            {"cost_reductions", c.cost_reductions},
            {"is_key_card", c.is_key_card},
            {"ai_importance_score", c.ai_importance_score}
        };
        if (c.spell_side) {
            j["spell_side"] = *c.spell_side;
        } else {
            j["spell_side"] = nullptr;
        }
    }

    inline void from_json(const nlohmann::json& j, CardData& c) {
        c.id = j.value("id", 0);
        c.name = j.value("name", std::string(""));
        c.cost = j.value("cost", 0);
        if (j.contains("civilizations")) j.at("civilizations").get_to(c.civilizations); else c.civilizations = {};
        c.power = j.value("power", 0);
        c.type = j.value("type", std::string("CREATURE"));
        if (j.contains("races")) j.at("races").get_to(c.races); else c.races = {};

        if (j.contains("effects")) {
            const auto& arr = j.at("effects");
            c.effects.clear();
            if (arr.is_array()) {
                c.effects.reserve(arr.size());
                for (const auto& elem : arr) {
                    EffectDef e;
                    from_json(elem, e); // Using ADL or found via lookup
                    c.effects.push_back(e);
                }
            }
        } else {
            c.effects = {};
        }

        if (j.contains("metamorph_abilities")) {
            const auto& arr = j.at("metamorph_abilities");
            c.metamorph_abilities.clear();
            if (arr.is_array()) {
                c.metamorph_abilities.reserve(arr.size());
                for (const auto& elem : arr) {
                    EffectDef e;
                    from_json(elem, e);
                    c.metamorph_abilities.push_back(e);
                }
            }
        } else {
            c.metamorph_abilities = {};
        }

        if (j.contains("revolution_change_condition")) {
             j.at("revolution_change_condition").get_to(c.revolution_change_condition);
        }

        if (j.contains("keywords")) j.at("keywords").get_to(c.keywords);

        if (j.contains("reaction_abilities")) {
            const auto& arr = j.at("reaction_abilities");
            c.reaction_abilities.clear();
            if (arr.is_array()) {
                c.reaction_abilities.reserve(arr.size());
                for (const auto& elem : arr) {
                    ReactionAbility ra;
                    from_json(elem, ra);
                    c.reaction_abilities.push_back(ra);
                }
            }
        } else {
            c.reaction_abilities = {};
        }

        if (j.contains("cost_reductions")) {
            const auto& arr = j.at("cost_reductions");
            c.cost_reductions.clear();
            if (arr.is_array()) {
                c.cost_reductions.reserve(arr.size());
                for (const auto& elem : arr) {
                    CostReductionDef cr;
                    from_json(elem, cr);
                    c.cost_reductions.push_back(cr);
                }
            }
        } else {
            c.cost_reductions = {};
        }

        c.is_key_card = j.value("is_key_card", false);
        c.ai_importance_score = j.value("ai_importance_score", 0);

        if (j.contains("spell_side") && !j["spell_side"].is_null()) {
            c.spell_side = std::make_shared<CardData>();
            from_json(j.at("spell_side"), *c.spell_side); // Recursive call using ADL
        } else {
            c.spell_side = nullptr;
        }
    }
}
