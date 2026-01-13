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
        BEFORE_BREAK_SHIELD,
        ON_SHIELD_ADD,
        ON_CAST_SPELL,
        ON_OPPONENT_DRAW,
        ON_DRAW,
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

    enum class ModifierType {
        NONE,
        COST_MODIFIER,
        POWER_MODIFIER,
        GRANT_KEYWORD,
        SET_KEYWORD,
        FORCE_ATTACK
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

    NLOHMANN_JSON_SERIALIZE_ENUM(CardType, {
        {CardType::CREATURE, "CREATURE"},
        {CardType::SPELL, "SPELL"},
        {CardType::EVOLUTION_CREATURE, "EVOLUTION_CREATURE"},
        {CardType::CROSS_GEAR, "CROSS_GEAR"},
        {CardType::CASTLE, "CASTLE"},
        {CardType::PSYCHIC_CREATURE, "PSYCHIC_CREATURE"},
        {CardType::GR_CREATURE, "GR_CREATURE"},
        {CardType::TAMASEED, "TAMASEED"}
    })

    enum class EffectPrimitive {
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
        CAST_SPELL,
        PUT_CREATURE,
        SELECT_OPTION,
        RESOLVE_BATTLE,
        IF,
        IF_ELSE,
        ELSE,
        NONE
    };

    // New Engine: Hybrid Command Types
    enum class CommandType {
        // Primitives
        TRANSITION,
        MUTATE,
        FLOW,
        QUERY,

        // Macros
        DRAW_CARD,
        DISCARD,
        DESTROY,
        BOOST_MANA, // Deck -> Mana (formerly MANA_CHARGE)
        TAP,
        UNTAP,
        POWER_MOD,
        ADD_KEYWORD,
        RETURN_TO_HAND,
        BREAK_SHIELD,
        SEARCH_DECK,
        SHIELD_TRIGGER,

        // New Primitives (Phase 2 Strict Enforcement)
        MOVE_CARD,
        ADD_MANA,
        SEND_TO_MANA,
        PLAYER_MANA_CHARGE,
        SEARCH_DECK_BOTTOM,
        ADD_SHIELD,
        SEND_TO_DECK_BOTTOM,

        // Expanded Set
        ATTACK_PLAYER,
        ATTACK_CREATURE,
        BLOCK,
        RESOLVE_BATTLE,
        RESOLVE_PLAY,
        RESOLVE_EFFECT,
        SHUFFLE_DECK,
        LOOK_AND_ADD,
        MEKRAID,
        REVEAL_CARDS,
        PLAY_FROM_ZONE,
        CAST_SPELL,
        SUMMON_TOKEN,
        SHIELD_BURN,
        SELECT_NUMBER,
        CHOICE,
        LOOK_TO_BUFFER,
        SELECT_FROM_BUFFER,
        PLAY_FROM_BUFFER,
        MOVE_BUFFER_TO_ZONE,
        FRIEND_BURST,
        REGISTER_DELAYED_EFFECT,
        IF,
        IF_ELSE,
        ELSE,

        NONE
    };

    // Phase 4: Cost System Enums
    enum class CostType {
        MANA,
        TAP_CARD,
        SACRIFICE_CARD,
        RETURN_CARD,
        SHIELD_BURN,
        DISCARD
    };

    enum class ReductionType {
        PASSIVE,
        ACTIVE_PAYMENT
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
        std::optional<bool> is_card_designation;
        std::optional<int> count;

        std::optional<std::string> selection_mode;
        std::optional<std::string> selection_sort_key;

        std::optional<std::string> power_max_ref;

        std::vector<FilterDef> and_conditions;
    };

    struct CostDef {
        CostType type;
        int amount;
        FilterDef filter;
        bool is_optional = false;
        std::string cost_id;
    };

    struct CostReductionDef {
        ReductionType type;
        CostDef unit_cost;
        int reduction_amount;
        int max_units = -1;
        int min_mana_cost = 0;
        std::string name;
    };

    struct ConditionDef {
        std::string type;
        int value = 0;
        std::string str_val;
        std::string stat_key;
        std::string op;
        std::optional<FilterDef> filter;
    };

    struct ModifierDef {
        ModifierType type = ModifierType::NONE;
        int value = 0;
        std::string str_val;
        ConditionDef condition;
        FilterDef filter;
    };

    struct ActionDef {
        EffectPrimitive type = EffectPrimitive::NONE;
        TargetScope scope = TargetScope::NONE;
        FilterDef filter;
        int value1 = 0;
        int value2 = 0;
        std::string str_val;
        std::string value;
        bool optional = false;
        std::string target_player;
        std::string source_zone;
        std::string destination_zone;
        std::string target_choice;
        std::string input_value_key;
        std::string input_value_usage;
        std::string output_value_key;
        bool inverse_target = false;
        std::optional<ConditionDef> condition;
        std::vector<std::vector<ActionDef>> options;
        bool cast_spell_side = false;
    };

    struct CommandDef {
        CommandType type = CommandType::NONE;
        int instance_id = 0;
        int target_instance = 0;
        int owner_id = 0;
        TargetScope target_group = TargetScope::NONE;
        FilterDef target_filter;
        int amount = 0;
        std::string str_param;
        bool optional = false;
        std::string from_zone;
        std::string to_zone;
        std::string mutation_kind;
        std::optional<ConditionDef> condition;
        std::vector<CommandDef> if_true;
        std::vector<CommandDef> if_false;
        std::string input_value_key;
        std::string input_value_usage;
        std::string output_value_key;
    };

    struct EffectDef {
        TriggerType trigger = TriggerType::NONE;
        TargetScope trigger_scope = TargetScope::NONE;
        FilterDef trigger_filter;
        ConditionDef condition;
        std::vector<ActionDef> actions;
        std::vector<CommandDef> commands;
    };

    struct ReactionCondition {
        std::string trigger_event;
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
        CardType type; // Changed from std::string to CardType
        std::vector<std::string> races;
        std::vector<EffectDef> effects;
        std::vector<ModifierDef> static_abilities; // Added
        std::vector<EffectDef> metamorph_abilities;
        std::optional<FilterDef> evolution_condition;
        std::optional<FilterDef> revolution_change_condition;
        std::optional<std::map<std::string, bool>> keywords;
        std::vector<ReactionAbility> reaction_abilities;
        std::vector<CostReductionDef> cost_reductions;
        std::shared_ptr<CardData> spell_side;

        bool is_key_card = false;
        int ai_importance_score = 0;
    };

    void to_json(nlohmann::json& j, const CardData& c);
    void from_json(const nlohmann::json& j, CardData& c);

} // namespace dm::core

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
        {TriggerType::BEFORE_BREAK_SHIELD, "BEFORE_BREAK_SHIELD"},
        {TriggerType::ON_SHIELD_ADD, "ON_SHIELD_ADD"},
        {TriggerType::ON_CAST_SPELL, "ON_CAST_SPELL"},
        {TriggerType::ON_OPPONENT_DRAW, "ON_OPPONENT_DRAW"},
        {TriggerType::ON_DRAW, "ON_DRAW"}
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

    NLOHMANN_JSON_SERIALIZE_ENUM(ModifierType, {
        {ModifierType::NONE, "NONE"},
        {ModifierType::COST_MODIFIER, "COST_MODIFIER"},
        {ModifierType::POWER_MODIFIER, "POWER_MODIFIER"},
        {ModifierType::GRANT_KEYWORD, "GRANT_KEYWORD"},
        {ModifierType::SET_KEYWORD, "SET_KEYWORD"},
        {ModifierType::FORCE_ATTACK, "FORCE_ATTACK"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(EffectPrimitive, {
        {EffectPrimitive::NONE, "NONE"},
        {EffectPrimitive::DRAW_CARD, "DRAW_CARD"},
        {EffectPrimitive::ADD_MANA, "ADD_MANA"},
        {EffectPrimitive::DESTROY, "DESTROY"},
        {EffectPrimitive::RETURN_TO_HAND, "RETURN_TO_HAND"},
        {EffectPrimitive::SEND_TO_MANA, "SEND_TO_MANA"},
        {EffectPrimitive::TAP, "TAP"},
        {EffectPrimitive::UNTAP, "UNTAP"},
        {EffectPrimitive::MODIFY_POWER, "MODIFY_POWER"},
        {EffectPrimitive::BREAK_SHIELD, "BREAK_SHIELD"},
        {EffectPrimitive::LOOK_AND_ADD, "LOOK_AND_ADD"},
        {EffectPrimitive::SUMMON_TOKEN, "SUMMON_TOKEN"},
        {EffectPrimitive::SEARCH_DECK_BOTTOM, "SEARCH_DECK_BOTTOM"},
        {EffectPrimitive::MEKRAID, "MEKRAID"},
        {EffectPrimitive::DISCARD, "DISCARD"},
        {EffectPrimitive::PLAY_FROM_ZONE, "PLAY_FROM_ZONE"},
        {EffectPrimitive::COST_REFERENCE, "COST_REFERENCE"},
        {EffectPrimitive::LOOK_TO_BUFFER, "LOOK_TO_BUFFER"},
        {EffectPrimitive::SELECT_FROM_BUFFER, "SELECT_FROM_BUFFER"},
        {EffectPrimitive::PLAY_FROM_BUFFER, "PLAY_FROM_BUFFER"},
        {EffectPrimitive::MOVE_BUFFER_TO_ZONE, "MOVE_BUFFER_TO_ZONE"},
        {EffectPrimitive::REVOLUTION_CHANGE, "REVOLUTION_CHANGE"},
        {EffectPrimitive::COUNT_CARDS, "COUNT_CARDS"},
        {EffectPrimitive::GET_GAME_STAT, "GET_GAME_STAT"},
        {EffectPrimitive::APPLY_MODIFIER, "APPLY_MODIFIER"},
        {EffectPrimitive::REVEAL_CARDS, "REVEAL_CARDS"},
        {EffectPrimitive::REGISTER_DELAYED_EFFECT, "REGISTER_DELAYED_EFFECT"},
        {EffectPrimitive::RESET_INSTANCE, "RESET_INSTANCE"},
        {EffectPrimitive::SEARCH_DECK, "SEARCH_DECK"},
        {EffectPrimitive::SHUFFLE_DECK, "SHUFFLE_DECK"},
        {EffectPrimitive::ADD_SHIELD, "ADD_SHIELD"},
        {EffectPrimitive::SEND_SHIELD_TO_GRAVE, "SEND_SHIELD_TO_GRAVE"},
        {EffectPrimitive::SEND_TO_DECK_BOTTOM, "SEND_TO_DECK_BOTTOM"},
        {EffectPrimitive::MOVE_TO_UNDER_CARD, "MOVE_TO_UNDER_CARD"},
        {EffectPrimitive::SELECT_NUMBER, "SELECT_NUMBER"},
        {EffectPrimitive::FRIEND_BURST, "FRIEND_BURST"},
        {EffectPrimitive::GRANT_KEYWORD, "GRANT_KEYWORD"},
        {EffectPrimitive::MOVE_CARD, "MOVE_CARD"},
        {EffectPrimitive::CAST_SPELL, "CAST_SPELL"},
        {EffectPrimitive::PUT_CREATURE, "PUT_CREATURE"},
        {EffectPrimitive::SELECT_OPTION, "SELECT_OPTION"},
        {EffectPrimitive::RESOLVE_BATTLE, "RESOLVE_BATTLE"},
        {EffectPrimitive::IF, "IF"},
        {EffectPrimitive::IF_ELSE, "IF_ELSE"},
        {EffectPrimitive::ELSE, "ELSE"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(CommandType, {
        {CommandType::NONE, "NONE"},
        {CommandType::TRANSITION, "TRANSITION"},
        {CommandType::MUTATE, "MUTATE"},
        {CommandType::FLOW, "FLOW"},
        {CommandType::QUERY, "QUERY"},
        {CommandType::DRAW_CARD, "DRAW_CARD"},
        {CommandType::DISCARD, "DISCARD"},
        {CommandType::DESTROY, "DESTROY"},
        {CommandType::BOOST_MANA, "BOOST_MANA"},
        {CommandType::TAP, "TAP"},
        {CommandType::UNTAP, "UNTAP"},
        {CommandType::POWER_MOD, "POWER_MOD"},
        {CommandType::ADD_KEYWORD, "ADD_KEYWORD"},
        {CommandType::RETURN_TO_HAND, "RETURN_TO_HAND"},
        {CommandType::BREAK_SHIELD, "BREAK_SHIELD"},
        {CommandType::SEARCH_DECK, "SEARCH_DECK"},
        {CommandType::SHIELD_TRIGGER, "SHIELD_TRIGGER"},

        // New Primitives (Phase 2 Strict Enforcement)
        {CommandType::MOVE_CARD, "MOVE_CARD"},
        {CommandType::ADD_MANA, "ADD_MANA"},
        {CommandType::SEND_TO_MANA, "SEND_TO_MANA"},
        {CommandType::PLAYER_MANA_CHARGE, "PLAYER_MANA_CHARGE"},
        {CommandType::SEARCH_DECK_BOTTOM, "SEARCH_DECK_BOTTOM"},
        {CommandType::ADD_SHIELD, "ADD_SHIELD"},
        {CommandType::SEND_TO_DECK_BOTTOM, "SEND_TO_DECK_BOTTOM"},

        // Expanded Set
        {CommandType::ATTACK_PLAYER, "ATTACK_PLAYER"},
        {CommandType::ATTACK_CREATURE, "ATTACK_CREATURE"},
        {CommandType::BLOCK, "BLOCK"},
        {CommandType::RESOLVE_BATTLE, "RESOLVE_BATTLE"},
        {CommandType::RESOLVE_PLAY, "RESOLVE_PLAY"},
        {CommandType::RESOLVE_EFFECT, "RESOLVE_EFFECT"},
        {CommandType::SHUFFLE_DECK, "SHUFFLE_DECK"},
        {CommandType::LOOK_AND_ADD, "LOOK_AND_ADD"},
        {CommandType::MEKRAID, "MEKRAID"},
        {CommandType::REVEAL_CARDS, "REVEAL_CARDS"},
        {CommandType::PLAY_FROM_ZONE, "PLAY_FROM_ZONE"},
        {CommandType::CAST_SPELL, "CAST_SPELL"},
        {CommandType::SUMMON_TOKEN, "SUMMON_TOKEN"},
        {CommandType::SHIELD_BURN, "SHIELD_BURN"},
        {CommandType::SELECT_NUMBER, "SELECT_NUMBER"},
        {CommandType::CHOICE, "CHOICE"},
        {CommandType::LOOK_TO_BUFFER, "LOOK_TO_BUFFER"},
        {CommandType::SELECT_FROM_BUFFER, "SELECT_FROM_BUFFER"},
        {CommandType::PLAY_FROM_BUFFER, "PLAY_FROM_BUFFER"},
        {CommandType::MOVE_BUFFER_TO_ZONE, "MOVE_BUFFER_TO_ZONE"},
        {CommandType::FRIEND_BURST, "FRIEND_BURST"},
        {CommandType::REGISTER_DELAYED_EFFECT, "REGISTER_DELAYED_EFFECT"},
        {CommandType::IF, "IF"},
        {CommandType::IF_ELSE, "IF_ELSE"},
        {CommandType::ELSE, "ELSE"}
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
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ModifierDef, type, value, str_val, condition, filter)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ActionDef, type, scope, filter, value1, value2, str_val, value, optional, target_player, source_zone, destination_zone, target_choice, input_value_key, input_value_usage, output_value_key, inverse_target, condition, options, cast_spell_side)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CommandDef, type, instance_id, target_instance, owner_id, target_group, target_filter, amount, str_param, optional, from_zone, to_zone, mutation_kind, condition, if_true, if_false, input_value_key, input_value_usage, output_value_key)

    // Manual to_json for EffectDef to exclude actions
    inline void to_json(nlohmann::json& j, const EffectDef& e) {
        j = nlohmann::json{
            {"trigger", e.trigger},
            {"trigger_scope", e.trigger_scope},
            {"trigger_filter", e.trigger_filter},
            {"condition", e.condition},
            {"commands", e.commands}
            // Explicitly excluding "actions" from output
        };
    }

    inline void from_json(const nlohmann::json& j, EffectDef& e) {
        if (j.contains("trigger")) j.at("trigger").get_to(e.trigger);
        if (j.contains("trigger_scope")) j.at("trigger_scope").get_to(e.trigger_scope);
        if (j.contains("trigger_filter")) j.at("trigger_filter").get_to(e.trigger_filter);
        if (j.contains("condition")) j.at("condition").get_to(e.condition);
        if (j.contains("commands")) j.at("commands").get_to(e.commands);
        if (j.contains("actions")) j.at("actions").get_to(e.actions);
    }

    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EffectDef, trigger, condition, actions, commands) -- REPLACED BY MANUAL IMPL ABOVE

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionCondition, trigger_event, civilization_match, mana_count_min, same_civilization_shield)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionAbility, type, cost, zone, condition)

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
            {"triggers", c.effects}, // Mapped to triggers in JSON
            {"static_abilities", c.static_abilities}, // Added
            {"metamorph_abilities", c.metamorph_abilities},
            {"evolution_condition", c.evolution_condition},
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
        c.type = j.value("type", CardType::CREATURE); // Default to CREATURE
        if (j.contains("races")) j.at("races").get_to(c.races); else c.races = {};

        // Support both "triggers" and "effects"
        if (j.contains("triggers")) {
            j.at("triggers").get_to(c.effects);
        } else if (j.contains("effects")) {
            j.at("effects").get_to(c.effects);
        } else {
            c.effects = {};
        }

        if (j.contains("static_abilities")) {
            j.at("static_abilities").get_to(c.static_abilities);
        } else {
            c.static_abilities = {};
        }

        if (j.contains("metamorph_abilities")) {
            j.at("metamorph_abilities").get_to(c.metamorph_abilities);
        } else {
            c.metamorph_abilities = {};
        }

        if (j.contains("evolution_condition")) {
             j.at("evolution_condition").get_to(c.evolution_condition);
        }

        if (j.contains("revolution_change_condition")) {
             j.at("revolution_change_condition").get_to(c.revolution_change_condition);
        }

        if (j.contains("keywords")) j.at("keywords").get_to(c.keywords);

        if (j.contains("reaction_abilities")) {
            j.at("reaction_abilities").get_to(c.reaction_abilities);
        } else {
            c.reaction_abilities = {};
        }

        if (j.contains("cost_reductions")) {
            j.at("cost_reductions").get_to(c.cost_reductions);
        } else {
            c.cost_reductions = {};
        }

        c.is_key_card = j.value("is_key_card", false);
        c.ai_importance_score = j.value("ai_importance_score", 0);

        if (j.contains("spell_side") && !j["spell_side"].is_null()) {
            c.spell_side = std::make_shared<CardData>();
            from_json(j.at("spell_side"), *c.spell_side);
        } else {
            c.spell_side = nullptr;
        }
    }
}
