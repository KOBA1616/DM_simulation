#include "bindings/bindings.hpp"
#include "bindings/bindings_helper.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/card_json_types.hpp"
#include "core/card_stats.hpp"
#include "core/instruction.hpp"
#include "core/action.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/command_system.hpp" // Added include for CommandSystem
#include <pybind11/stl.h>

using namespace dm;
using namespace dm::core;

void bind_core(py::module& m) {
    // ... (Previous Enums and Classes) ...
    // GameEvent bindings
    py::enum_<dm::core::EventType>(m, "EventType")
        .value("NONE", dm::core::EventType::NONE)
        .value("ZONE_ENTER", dm::core::EventType::ZONE_ENTER)
        .value("ZONE_LEAVE", dm::core::EventType::ZONE_LEAVE)
        .value("TURN_START", dm::core::EventType::TURN_START)
        .value("TURN_END", dm::core::EventType::TURN_END)
        .value("PHASE_START", dm::core::EventType::PHASE_START)
        .value("PHASE_END", dm::core::EventType::PHASE_END)
        .value("PLAY_CARD", dm::core::EventType::PLAY_CARD)
        .value("ATTACK_INITIATE", dm::core::EventType::ATTACK_INITIATE)
        .value("BLOCK_INITIATE", dm::core::EventType::BLOCK_INITIATE)
        .value("BATTLE_START", dm::core::EventType::BATTLE_START)
        .value("BATTLE_WIN", dm::core::EventType::BATTLE_WIN)
        .value("BATTLE_LOSE", dm::core::EventType::BATTLE_LOSE)
        .value("SHIELD_BREAK", dm::core::EventType::SHIELD_BREAK)
        .value("DIRECT_ATTACK", dm::core::EventType::DIRECT_ATTACK)
        .value("TAP_CARD", dm::core::EventType::TAP_CARD)
        .value("UNTAP_CARD", dm::core::EventType::UNTAP_CARD)
        .value("CUSTOM", dm::core::EventType::CUSTOM)
        .export_values();

    py::enum_<dm::core::GameState::Status>(m, "GameStatus")
        .value("PLAYING", dm::core::GameState::Status::PLAYING)
        .value("WAITING_FOR_REACTION", dm::core::GameState::Status::WAITING_FOR_REACTION)
        .value("GAME_OVER", dm::core::GameState::Status::GAME_OVER)
        .export_values();

    py::class_<dm::core::GameEvent>(m, "GameEvent")
        .def(py::init<>())
        .def(py::init<dm::core::EventType, int, int, PlayerID>(),
             py::arg("type"), py::arg("inst") = -1, py::arg("tgt") = -1, py::arg("pid") = 255)
        .def_readwrite("type", &dm::core::GameEvent::type)
        .def_readwrite("instance_id", &dm::core::GameEvent::instance_id)
        .def_readwrite("card_id", &dm::core::GameEvent::card_id)
        .def_readwrite("player_id", &dm::core::GameEvent::player_id)
        .def_readwrite("target_id", &dm::core::GameEvent::target_id)
        .def_readwrite("context", &dm::core::GameEvent::context);

    // Enums
    py::enum_<Civilization>(m, "Civilization")
        .value("FIRE", Civilization::FIRE)
        .value("WATER", Civilization::WATER)
        .value("NATURE", Civilization::NATURE)
        .value("LIGHT", Civilization::LIGHT)
        .value("DARKNESS", Civilization::DARKNESS)
        .value("ZERO", Civilization::ZERO)
        .value("NONE", Civilization::NONE)
        .export_values();

    py::enum_<CardType>(m, "CardType")
        .value("CREATURE", CardType::CREATURE)
        .value("SPELL", CardType::SPELL)
        .value("EVOLUTION_CREATURE", CardType::EVOLUTION_CREATURE)
        .value("CROSS_GEAR", CardType::CROSS_GEAR)
        .value("TAMASEED", CardType::TAMASEED)
        .value("PSYCHIC_CREATURE", CardType::PSYCHIC_CREATURE)
        .value("GR_CREATURE", CardType::GR_CREATURE)
        .export_values();

    py::enum_<Zone>(m, "Zone")
        .value("DECK", Zone::DECK)
        .value("HAND", Zone::HAND)
        .value("MANA", Zone::MANA)
        .value("GRAVEYARD", Zone::GRAVEYARD)
        .value("BATTLE", Zone::BATTLE)
        .value("SHIELD", Zone::SHIELD)
        .value("HYPER_SPATIAL", Zone::HYPER_SPATIAL)
        .value("GR_ZONE", Zone::GR_DECK)
        .value("BUFFER", Zone::BUFFER)
        .value("STACK", Zone::STACK)
        .export_values();

    py::enum_<Phase>(m, "Phase")
        .value("START_OF_TURN", Phase::START_OF_TURN)
        .value("DRAW", Phase::DRAW)
        .value("MANA", Phase::MANA)
        .value("MAIN", Phase::MAIN)
        .value("ATTACK", Phase::ATTACK)
        .value("BLOCK", Phase::BLOCK)
        .value("END_OF_TURN", Phase::END_OF_TURN)
        .export_values();

    py::enum_<PlayerIntent>(m, "PlayerIntent")
        .value("PASS", PlayerIntent::PASS)
        .value("PLAY_CARD", PlayerIntent::PLAY_CARD)
        .value("MANA_CHARGE", PlayerIntent::MANA_CHARGE)
        .value("ATTACK_CREATURE", PlayerIntent::ATTACK_CREATURE)
        .value("ATTACK_PLAYER", PlayerIntent::ATTACK_PLAYER)
        .value("BLOCK", PlayerIntent::BLOCK)
        .value("USE_SHIELD_TRIGGER", PlayerIntent::USE_SHIELD_TRIGGER)
        .value("RESOLVE_EFFECT", PlayerIntent::RESOLVE_EFFECT)
        .value("SELECT_TARGET", PlayerIntent::SELECT_TARGET)
        .value("USE_ABILITY", PlayerIntent::USE_ABILITY)
        .value("DECLARE_REACTION", PlayerIntent::DECLARE_REACTION)
        .value("PLAY_CARD_INTERNAL", PlayerIntent::PLAY_CARD_INTERNAL)
        .value("RESOLVE_BATTLE", PlayerIntent::RESOLVE_BATTLE)
        .value("BREAK_SHIELD", PlayerIntent::BREAK_SHIELD)
        .value("MOVE_CARD", PlayerIntent::MOVE_CARD)
        .value("DECLARE_PLAY", PlayerIntent::DECLARE_PLAY)
        .value("PAY_COST", PlayerIntent::PAY_COST)
        .value("RESOLVE_PLAY", PlayerIntent::RESOLVE_PLAY)
        .value("SELECT_OPTION", PlayerIntent::SELECT_OPTION)
        .value("SELECT_NUMBER", PlayerIntent::SELECT_NUMBER)
        .export_values();

    // Alias for backward compatibility
    m.attr("ActionType") = m.attr("PlayerIntent");

    py::enum_<GameResult>(m, "GameResult")
        .value("NONE", GameResult::NONE)
        .value("P1_WIN", GameResult::P1_WIN)
        .value("P2_WIN", GameResult::P2_WIN)
        .value("DRAW", GameResult::DRAW)
        .export_values();

    py::enum_<TargetScope>(m, "TargetScope")
        .value("TARGET_SELECT", TargetScope::TARGET_SELECT)
        .value("NONE", TargetScope::NONE)
        .value("SELF", TargetScope::SELF)
        .value("PLAYER_SELF", TargetScope::PLAYER_SELF)
        .value("PLAYER_OPPONENT", TargetScope::PLAYER_OPPONENT)
        .value("ALL_PLAYERS", TargetScope::ALL_PLAYERS)
        .value("RANDOM", TargetScope::RANDOM)
        .value("ALL_FILTERED", TargetScope::ALL_FILTERED)
        .export_values();

    py::enum_<TriggerType>(m, "TriggerType")
        .value("ON_PLAY", TriggerType::ON_PLAY)
        .value("ON_ATTACK", TriggerType::ON_ATTACK)
        .value("ON_DESTROY", TriggerType::ON_DESTROY)
        .value("S_TRIGGER", TriggerType::S_TRIGGER)
        .value("TURN_START", TriggerType::TURN_START)
        .value("PASSIVE_CONST", TriggerType::PASSIVE_CONST)
        .value("BEFORE_BREAK_SHIELD", TriggerType::BEFORE_BREAK_SHIELD)
        .value("ON_BLOCK", TriggerType::ON_BLOCK)
        .value("NONE", TriggerType::NONE)
        .export_values();

    py::enum_<PassiveType>(m, "PassiveType")
        .value("POWER_MODIFIER", PassiveType::POWER_MODIFIER)
        .value("KEYWORD_GRANT", PassiveType::KEYWORD_GRANT)
        .value("COST_REDUCTION", PassiveType::COST_REDUCTION)
        .value("BLOCKER_GRANT", PassiveType::BLOCKER_GRANT)
        .value("SPEED_ATTACKER_GRANT", PassiveType::SPEED_ATTACKER_GRANT)
        .value("SLAYER_GRANT", PassiveType::SLAYER_GRANT)
        .value("CANNOT_ATTACK", PassiveType::CANNOT_ATTACK)
        .value("CANNOT_BLOCK", PassiveType::CANNOT_BLOCK)
        .value("CANNOT_USE_SPELLS", PassiveType::CANNOT_USE_SPELLS)
        .value("LOCK_SPELL_BY_COST", PassiveType::LOCK_SPELL_BY_COST)
        .value("CANNOT_SUMMON", PassiveType::CANNOT_SUMMON)
        .export_values();

    py::class_<PassiveEffect>(m, "PassiveEffect")
        .def(py::init<>())
        .def_readwrite("type", &PassiveEffect::type)
        .def_readwrite("value", &PassiveEffect::value)
        .def_readwrite("str_value", &PassiveEffect::str_value)
        .def_readwrite("target_filter", &PassiveEffect::target_filter)
        .def_readwrite("specific_targets", &PassiveEffect::specific_targets)
        .def_readwrite("condition", &PassiveEffect::condition)
        .def_readwrite("source_instance_id", &PassiveEffect::source_instance_id)
        .def_readwrite("controller", &PassiveEffect::controller)
        .def_readwrite("turns_remaining", &PassiveEffect::turns_remaining)
        .def_readwrite("is_source_static", &PassiveEffect::is_source_static);

    py::enum_<EffectType>(m, "EffectType")
        .value("NONE", EffectType::NONE)
        .value("CIP", EffectType::CIP)
        .value("AT_ATTACK", EffectType::AT_ATTACK)
        .value("AT_BLOCK", EffectType::AT_BLOCK)
        .value("AT_START_OF_TURN", EffectType::AT_START_OF_TURN)
        .value("AT_END_OF_TURN", EffectType::AT_END_OF_TURN)
        .value("SHIELD_TRIGGER", EffectType::SHIELD_TRIGGER)
        .value("G_STRIKE", EffectType::G_STRIKE)
        .value("DESTRUCTION", EffectType::DESTRUCTION)
        .value("ON_ATTACK_FROM_HAND", EffectType::ON_ATTACK_FROM_HAND)
        .value("INTERNAL_PLAY", EffectType::INTERNAL_PLAY)
        .value("META_COUNTER", EffectType::META_COUNTER)
        .value("RESOLVE_BATTLE", EffectType::RESOLVE_BATTLE)
        .value("BREAK_SHIELD", EffectType::BREAK_SHIELD)
        .value("REACTION_WINDOW", EffectType::REACTION_WINDOW)
        .value("TRIGGER_ABILITY", EffectType::TRIGGER_ABILITY)
        .value("SELECT_OPTION", EffectType::SELECT_OPTION)
        .value("SELECT_NUMBER", EffectType::SELECT_NUMBER)
        .export_values();

    py::enum_<ResolveType>(m, "ResolveType")
        .value("NONE", ResolveType::NONE)
        .value("TARGET_SELECT", ResolveType::TARGET_SELECT)
        .value("EFFECT_RESOLUTION", ResolveType::EFFECT_RESOLUTION)
        .export_values();

    // Re-assert TARGET_SELECT to be TargetScope::TARGET_SELECT (resolving collision with ResolveType)
    m.attr("TARGET_SELECT") = TargetScope::TARGET_SELECT;

    py::enum_<ModifierType>(m, "ModifierType")
        .value("NONE", ModifierType::NONE)
        .value("COST_MODIFIER", ModifierType::COST_MODIFIER)
        .value("POWER_MODIFIER", ModifierType::POWER_MODIFIER)
        .value("GRANT_KEYWORD", ModifierType::GRANT_KEYWORD)
        .value("SET_KEYWORD", ModifierType::SET_KEYWORD)
        .export_values();

    py::enum_<EffectPrimitive>(m, "EffectPrimitive")
        .value("DRAW_CARD", EffectPrimitive::DRAW_CARD)
        .value("ADD_MANA", EffectPrimitive::ADD_MANA)
        .value("SEARCH_DECK_BOTTOM", EffectPrimitive::SEARCH_DECK_BOTTOM)
        .value("SEND_TO_DECK_BOTTOM", EffectPrimitive::SEND_TO_DECK_BOTTOM)
        .value("SEARCH_DECK", EffectPrimitive::SEARCH_DECK)
        .value("SHUFFLE_DECK", EffectPrimitive::SHUFFLE_DECK)
        .value("DESTROY", EffectPrimitive::DESTROY)
        .value("RETURN_TO_HAND", EffectPrimitive::RETURN_TO_HAND)
        .value("SEND_TO_MANA", EffectPrimitive::SEND_TO_MANA)
        .value("TAP", EffectPrimitive::TAP)
        .value("UNTAP", EffectPrimitive::UNTAP)
        .value("MODIFY_POWER", EffectPrimitive::MODIFY_POWER)
        .value("BREAK_SHIELD", EffectPrimitive::BREAK_SHIELD)
        .value("LOOK_AND_ADD", EffectPrimitive::LOOK_AND_ADD)
        .value("SUMMON_TOKEN", EffectPrimitive::SUMMON_TOKEN)
        .value("MEKRAID", EffectPrimitive::MEKRAID)
        .value("DISCARD", EffectPrimitive::DISCARD)
        .value("PLAY_FROM_ZONE", EffectPrimitive::PLAY_FROM_ZONE)
        .value("COST_REFERENCE", EffectPrimitive::COST_REFERENCE)
        .value("LOOK_TO_BUFFER", EffectPrimitive::LOOK_TO_BUFFER)
        .value("SELECT_FROM_BUFFER", EffectPrimitive::SELECT_FROM_BUFFER)
        .value("PLAY_FROM_BUFFER", EffectPrimitive::PLAY_FROM_BUFFER)
        .value("MOVE_BUFFER_TO_ZONE", EffectPrimitive::MOVE_BUFFER_TO_ZONE)
        .value("REVOLUTION_CHANGE", EffectPrimitive::REVOLUTION_CHANGE)
        .value("COUNT_CARDS", EffectPrimitive::COUNT_CARDS)
        .value("GET_GAME_STAT", EffectPrimitive::GET_GAME_STAT)
        .value("APPLY_MODIFIER", EffectPrimitive::APPLY_MODIFIER)
        .value("REVEAL_CARDS", EffectPrimitive::REVEAL_CARDS)
        .value("REGISTER_DELAYED_EFFECT", EffectPrimitive::REGISTER_DELAYED_EFFECT)
        .value("RESET_INSTANCE", EffectPrimitive::RESET_INSTANCE)
        .value("ADD_SHIELD", EffectPrimitive::ADD_SHIELD)
        .value("SEND_SHIELD_TO_GRAVE", EffectPrimitive::SEND_SHIELD_TO_GRAVE)
        .value("MOVE_TO_UNDER_CARD", EffectPrimitive::MOVE_TO_UNDER_CARD)
        .value("SELECT_NUMBER", EffectPrimitive::SELECT_NUMBER)
        .value("FRIEND_BURST", EffectPrimitive::FRIEND_BURST)
        .value("GRANT_KEYWORD", EffectPrimitive::GRANT_KEYWORD)
        .value("MOVE_CARD", EffectPrimitive::MOVE_CARD)
        .value("CAST_SPELL", EffectPrimitive::CAST_SPELL)
        .value("PUT_CREATURE", EffectPrimitive::PUT_CREATURE)
        .value("SELECT_OPTION", EffectPrimitive::SELECT_OPTION)
        .value("RESOLVE_BATTLE", EffectPrimitive::RESOLVE_BATTLE)
        .export_values();

    // Alias for backward compatibility
    m.attr("EffectActionType") = m.attr("EffectPrimitive");

    // CommandType (New Enum from card_json_types.hpp)
    py::enum_<CommandType>(m, "CommandType")
        .value("TRANSITION", CommandType::TRANSITION)
        .value("MUTATE", CommandType::MUTATE)
        .value("FLOW", CommandType::FLOW)
        .value("QUERY", CommandType::QUERY)
        .value("DRAW_CARD", CommandType::DRAW_CARD)
        .value("DISCARD", CommandType::DISCARD)
        .value("DESTROY", CommandType::DESTROY)
        .value("MANA_CHARGE", CommandType::MANA_CHARGE)
        .value("TAP", CommandType::TAP)
        .value("UNTAP", CommandType::UNTAP)
        .value("POWER_MOD", CommandType::POWER_MOD)
        .value("ADD_KEYWORD", CommandType::ADD_KEYWORD)
        .value("RETURN_TO_HAND", CommandType::RETURN_TO_HAND)
        .value("BREAK_SHIELD", CommandType::BREAK_SHIELD)
        .value("SEARCH_DECK", CommandType::SEARCH_DECK)
        .value("SHIELD_TRIGGER", CommandType::SHIELD_TRIGGER)

        // New Primitives (Phase 2 Strict Enforcement)
        .value("MOVE_CARD", CommandType::MOVE_CARD)
        .value("ADD_MANA", CommandType::ADD_MANA)
        .value("SEND_TO_MANA", CommandType::SEND_TO_MANA)
        .value("SEARCH_DECK_BOTTOM", CommandType::SEARCH_DECK_BOTTOM)
        .value("ADD_SHIELD", CommandType::ADD_SHIELD)
        .value("SEND_TO_DECK_BOTTOM", CommandType::SEND_TO_DECK_BOTTOM)

        // Expanded Set
        .value("ATTACK_PLAYER", CommandType::ATTACK_PLAYER)
        .value("ATTACK_CREATURE", CommandType::ATTACK_CREATURE)
        .value("BLOCK", CommandType::BLOCK)
        .value("RESOLVE_BATTLE", CommandType::RESOLVE_BATTLE)
        .value("RESOLVE_PLAY", CommandType::RESOLVE_PLAY)
        .value("RESOLVE_EFFECT", CommandType::RESOLVE_EFFECT)
        .value("SHUFFLE_DECK", CommandType::SHUFFLE_DECK)
        .value("LOOK_AND_ADD", CommandType::LOOK_AND_ADD)
        .value("MEKRAID", CommandType::MEKRAID)
        .value("REVEAL_CARDS", CommandType::REVEAL_CARDS)
        .value("PLAY_FROM_ZONE", CommandType::PLAY_FROM_ZONE)
        .value("CAST_SPELL", CommandType::CAST_SPELL)
        .value("SUMMON_TOKEN", CommandType::SUMMON_TOKEN)
        .value("SHIELD_BURN", CommandType::SHIELD_BURN)
        .value("SELECT_NUMBER", CommandType::SELECT_NUMBER)
        .value("CHOICE", CommandType::CHOICE)
        .value("LOOK_TO_BUFFER", CommandType::LOOK_TO_BUFFER)
        .value("SELECT_FROM_BUFFER", CommandType::SELECT_FROM_BUFFER)
        .value("PLAY_FROM_BUFFER", CommandType::PLAY_FROM_BUFFER)
        .value("MOVE_BUFFER_TO_ZONE", CommandType::MOVE_BUFFER_TO_ZONE)
        .value("FRIEND_BURST", CommandType::FRIEND_BURST)
        .value("REGISTER_DELAYED_EFFECT", CommandType::REGISTER_DELAYED_EFFECT)
        .value("NONE", CommandType::NONE)
        .export_values();

    py::enum_<InstructionOp>(m, "InstructionOp")
        .value("NOOP", InstructionOp::NOOP)
        .value("IF", InstructionOp::IF)
        .value("LOOP", InstructionOp::LOOP)
        .value("REPEAT", InstructionOp::REPEAT)
        .value("SELECT", InstructionOp::SELECT)
        .value("GET_STAT", InstructionOp::GET_STAT)
        .value("MOVE", InstructionOp::MOVE)
        .value("MODIFY", InstructionOp::MODIFY)
        .value("GAME_ACTION", InstructionOp::GAME_ACTION)
        .value("PLAY", InstructionOp::PLAY)
        .value("ATTACK", InstructionOp::ATTACK)
        .value("BLOCK", InstructionOp::BLOCK)
        .value("COUNT", InstructionOp::COUNT)
        .value("MATH", InstructionOp::MATH)
        .value("CALL", InstructionOp::CALL)
        .value("RETURN", InstructionOp::RETURN)
        .value("WAIT_INPUT", InstructionOp::WAIT_INPUT)
        .value("PRINT", InstructionOp::PRINT)
        .export_values();

    // Card Data Structures
    py::class_<CardKeywords>(m, "CardKeywords")
        .def(py::init<>())
        .def_property("g_zero", [](const CardKeywords& k) { return k.g_zero; }, [](CardKeywords& k, bool v) { k.g_zero = v; })
        .def_property("revolution_change", [](const CardKeywords& k) { return k.revolution_change; }, [](CardKeywords& k, bool v) { k.revolution_change = v; })
        .def_property("mach_fighter", [](const CardKeywords& k) { return k.mach_fighter; }, [](CardKeywords& k, bool v) { k.mach_fighter = v; })
        .def_property("speed_attacker", [](const CardKeywords& k) { return k.speed_attacker; }, [](CardKeywords& k, bool v) { k.speed_attacker = v; })
        .def_property("blocker", [](const CardKeywords& k) { return k.blocker; }, [](CardKeywords& k, bool v) { k.blocker = v; })
        .def_property("slayer", [](const CardKeywords& k) { return k.slayer; }, [](CardKeywords& k, bool v) { k.slayer = v; })
        .def_property("double_breaker", [](const CardKeywords& k) { return k.double_breaker; }, [](CardKeywords& k, bool v) { k.double_breaker = v; })
        .def_property("triple_breaker", [](const CardKeywords& k) { return k.triple_breaker; }, [](CardKeywords& k, bool v) { k.triple_breaker = v; })
        .def_property("shield_trigger", [](const CardKeywords& k) { return k.shield_trigger; }, [](CardKeywords& k, bool v) { k.shield_trigger = v; })
        .def_property("evolution", [](const CardKeywords& k) { return k.evolution; }, [](CardKeywords& k, bool v) { k.evolution = v; })
        .def_property("neo", [](const CardKeywords& k) { return k.neo; }, [](CardKeywords& k, bool v) { k.neo = v; })
        .def_property("g_neo", [](const CardKeywords& k) { return k.g_neo; }, [](CardKeywords& k, bool v) { k.g_neo = v; })
        .def_property("cip", [](const CardKeywords& k) { return k.cip; }, [](CardKeywords& k, bool v) { k.cip = v; })
        .def_property("at_attack", [](const CardKeywords& k) { return k.at_attack; }, [](CardKeywords& k, bool v) { k.at_attack = v; })
        .def_property("destruction", [](const CardKeywords& k) { return k.destruction; }, [](CardKeywords& k, bool v) { k.destruction = v; })
        .def_property("before_break_shield", [](const CardKeywords& k) { return k.before_break_shield; }, [](CardKeywords& k, bool v) { k.before_break_shield = v; })
        .def_property("just_diver", [](const CardKeywords& k) { return k.just_diver; }, [](CardKeywords& k, bool v) { k.just_diver = v; })
        .def_property("hyper_energy", [](const CardKeywords& k) { return k.hyper_energy; }, [](CardKeywords& k, bool v) { k.hyper_energy = v; })
        .def_property("at_block", [](const CardKeywords& k) { return k.at_block; }, [](CardKeywords& k, bool v) { k.at_block = v; })
        .def_property("at_start_of_turn", [](const CardKeywords& k) { return k.at_start_of_turn; }, [](CardKeywords& k, bool v) { k.at_start_of_turn = v; })
        .def_property("at_end_of_turn", [](const CardKeywords& k) { return k.at_end_of_turn; }, [](CardKeywords& k, bool v) { k.at_end_of_turn = v; })
        .def_property("g_strike", [](const CardKeywords& k) { return k.g_strike; }, [](CardKeywords& k, bool v) { k.g_strike = v; })
        .def_property("world_breaker", [](const CardKeywords& k) { return k.world_breaker; }, [](CardKeywords& k, bool v) { k.world_breaker = v; })
        .def_property("power_attacker", [](const CardKeywords& k) { return k.power_attacker; }, [](CardKeywords& k, bool v) { k.power_attacker = v; })
        .def_property("shield_burn", [](const CardKeywords& k) { return k.shield_burn; }, [](CardKeywords& k, bool v) { k.shield_burn = v; })
        .def_property("untap_in", [](const CardKeywords& k) { return k.untap_in; }, [](CardKeywords& k, bool v) { k.untap_in = v; })
        .def_property("unblockable", [](const CardKeywords& k) { return k.unblockable; }, [](CardKeywords& k, bool v) { k.unblockable = v; })
        .def_property("friend_burst", [](const CardKeywords& k) { return k.friend_burst; }, [](CardKeywords& k, bool v) { k.friend_burst = v; })
        .def_property("ex_life", [](const CardKeywords& k) { return k.ex_life; }, [](CardKeywords& k, bool v) { k.ex_life = v; })
        .def_property("mega_last_burst", [](const CardKeywords& k) { return k.mega_last_burst; }, [](CardKeywords& k, bool v) { k.mega_last_burst = v; });

    py::class_<FilterDef>(m, "FilterDef")
        .def(py::init<>())
        .def_readwrite("zones", &FilterDef::zones)
        .def_readwrite("types", &FilterDef::types)
        .def_readwrite("civilizations", &FilterDef::civilizations)
        .def_readwrite("races", &FilterDef::races)
        .def_readwrite("min_cost", &FilterDef::min_cost)
        .def_readwrite("max_cost", &FilterDef::max_cost)
        .def_readwrite("min_power", &FilterDef::min_power)
        .def_readwrite("max_power", &FilterDef::max_power)
        .def_readwrite("is_tapped", &FilterDef::is_tapped)
        .def_readwrite("is_blocker", &FilterDef::is_blocker)
        .def_readwrite("is_evolution", &FilterDef::is_evolution)
        .def_readwrite("owner", &FilterDef::owner)
        .def_readwrite("count", &FilterDef::count);

    py::class_<ConditionDef>(m, "ConditionDef")
        .def(py::init<>())
        .def_readwrite("type", &ConditionDef::type)
        .def_readwrite("value", &ConditionDef::value)
        .def_readwrite("str_val", &ConditionDef::str_val)
        .def_readwrite("stat_key", &ConditionDef::stat_key)
        .def_readwrite("op", &ConditionDef::op)
        .def_readwrite("filter", &ConditionDef::filter);

    py::class_<ModifierDef>(m, "ModifierDef")
        .def(py::init<>())
        .def_readwrite("type", &ModifierDef::type)
        .def_readwrite("value", &ModifierDef::value)
        .def_readwrite("str_val", &ModifierDef::str_val)
        .def_readwrite("condition", &ModifierDef::condition)
        .def_readwrite("filter", &ModifierDef::filter);

    py::class_<ActionDef>(m, "ActionDef")
        .def(py::init<>())
        .def_readwrite("type", &ActionDef::type)
        .def_readwrite("value1", &ActionDef::value1)
        .def_readwrite("value2", &ActionDef::value2)
        .def_readwrite("str_val", &ActionDef::str_val)
        .def_readwrite("optional", &ActionDef::optional)
        .def_readwrite("filter", &ActionDef::filter)
        .def_readwrite("target_player", &ActionDef::target_player)
        .def_readwrite("source_zone", &ActionDef::source_zone)
        .def_readwrite("destination_zone", &ActionDef::destination_zone)
        .def_readwrite("target_choice", &ActionDef::target_choice)
        .def_readwrite("input_value_key", &ActionDef::input_value_key)
        .def_readwrite("output_value_key", &ActionDef::output_value_key)
        .def_readwrite("condition", &ActionDef::condition)
        .def_readwrite("options", &ActionDef::options)
        .def_readwrite("scope", &ActionDef::scope)
        .def_readwrite("cast_spell_side", &ActionDef::cast_spell_side);

    // Bind CommandDef
    py::class_<CommandDef>(m, "CommandDef")
        .def(py::init<>())
        .def_readwrite("type", &CommandDef::type)
        .def_readwrite("instance_id", &CommandDef::instance_id) // Exposed to Python
        .def_readwrite("target_instance", &CommandDef::target_instance) // Exposed to Python
        .def_readwrite("owner_id", &CommandDef::owner_id) // Exposed to Python
        .def_readwrite("target_group", &CommandDef::target_group)
        .def_readwrite("target_filter", &CommandDef::target_filter)
        .def_readwrite("amount", &CommandDef::amount)
        .def_readwrite("str_param", &CommandDef::str_param)
        .def_readwrite("optional", &CommandDef::optional)
        .def_readwrite("from_zone", &CommandDef::from_zone)
        .def_readwrite("to_zone", &CommandDef::to_zone)
        .def_readwrite("mutation_kind", &CommandDef::mutation_kind)
        .def_readwrite("condition", &CommandDef::condition)
        .def_readwrite("if_true", &CommandDef::if_true)
        .def_readwrite("if_false", &CommandDef::if_false)
        .def_readwrite("input_value_key", &CommandDef::input_value_key)
        .def_readwrite("output_value_key", &CommandDef::output_value_key);

    py::class_<EffectDef>(m, "EffectDef")
        .def(py::init<>())
        .def_readwrite("trigger", &EffectDef::trigger)
        .def_readwrite("condition", &EffectDef::condition)
        .def_readwrite("actions", &EffectDef::actions)
        .def_readwrite("commands", &EffectDef::commands); // Added commands field

    py::class_<CardDefinition, std::shared_ptr<CardDefinition>>(m, "CardDefinition")
        .def(py::init([](int id, std::string name, std::string civ_str, std::vector<std::string> races, int cost, int power, CardKeywords keywords, std::vector<EffectDef> effects) {
            try {
                auto c = std::make_shared<CardDefinition>();
                c->id = id;
                c->name = name;
                c->cost = cost;
                if (civ_str == "FIRE") c->civilizations.push_back(Civilization::FIRE);
                else if (civ_str == "WATER") c->civilizations.push_back(Civilization::WATER);
                else if (civ_str == "NATURE") c->civilizations.push_back(Civilization::NATURE);
                else if (civ_str == "LIGHT") c->civilizations.push_back(Civilization::LIGHT);
                else if (civ_str == "DARKNESS") c->civilizations.push_back(Civilization::DARKNESS);
                c->power = power;
                c->races = races;
                c->keywords = keywords;
                c->effects = effects;
                // Default type
                c->type = CardType::CREATURE;
                return c;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in CardDefinition constructor: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in CardDefinition constructor");
            }
        }),
             py::arg("id") = 0, py::arg("name") = "", py::arg("civilization") = "NONE",
             py::arg("races") = std::vector<std::string>{}, py::arg("cost") = 0, py::arg("power") = 0,
             py::arg("keywords") = CardKeywords(), py::arg("effects") = std::vector<EffectDef>{})
        .def_readwrite("id", &CardDefinition::id)
        .def_readwrite("name", &CardDefinition::name)
        .def_readwrite("cost", &CardDefinition::cost)
        .def_readwrite("power", &CardDefinition::power)
        .def_readwrite("type", &CardDefinition::type)
        .def_readwrite("races", &CardDefinition::races)
        .def_readwrite("keywords", &CardDefinition::keywords)
        .def_readwrite("effects", &CardDefinition::effects)
        .def_readwrite("static_abilities", &CardDefinition::static_abilities)
        .def_readwrite("evolution_condition", &CardDefinition::evolution_condition)
        .def_readwrite("revolution_change_condition", &CardDefinition::revolution_change_condition)
        .def_readwrite("is_key_card", &CardDefinition::is_key_card)
        .def_readwrite("ai_importance_score", &CardDefinition::ai_importance_score)
        .def_readwrite("spell_side", &CardDefinition::spell_side)
        .def_property("civilization",
            [](const CardDefinition& c) { return c.civilizations.empty() ? Civilization::NONE : c.civilizations[0]; },
            [](CardDefinition& c, Civilization civ) { c.civilizations = {civ}; })
        .def_readwrite("civilizations", &CardDefinition::civilizations);

    py::class_<CardData>(m, "CardData")
        .def(py::init([](CardID id, std::string name, int cost, Civilization civilization, int power, CardType type, std::vector<std::string> races, std::vector<EffectDef> effects) {
            try {
                CardData c;
                c.id = id;
                c.name = name;
                c.cost = cost;
                c.civilizations.push_back(civilization);
                c.power = power;
                c.type = type;
                c.races = races;
                c.effects = effects;
                return c;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in CardData constructor: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in CardData constructor");
            }
        }))
        .def_readwrite("id", &CardData::id)
        .def_readwrite("name", &CardData::name)
        .def_readwrite("cost", &CardData::cost)
        .def_readwrite("power", &CardData::power)
        .def_readwrite("type", &CardData::type)
        .def_readwrite("races", &CardData::races)
        .def_readwrite("effects", &CardData::effects)
        .def_readwrite("static_abilities", &CardData::static_abilities)
        .def_readwrite("evolution_condition", &CardData::evolution_condition)
        .def_readwrite("keywords", &CardData::keywords);

    py::class_<CardInstance>(m, "CardInstance")
        .def(py::init<>())
        .def_readwrite("instance_id", &CardInstance::instance_id)
        .def_readwrite("card_id", &CardInstance::card_id)
        .def_readwrite("owner", &CardInstance::owner)
        .def_readwrite("is_tapped", &CardInstance::is_tapped)
        .def_readwrite("summoning_sickness", &CardInstance::summoning_sickness)
        .def_readwrite("turn_played", &CardInstance::turn_played)
        .def_readwrite("is_face_down", &CardInstance::is_face_down);

    // Instruction
    py::class_<Instruction>(m, "Instruction")
        .def(py::init<>())
        .def(py::init<InstructionOp>())
        .def_readwrite("op", &Instruction::op)
        .def("set_args", [](Instruction& i, py::dict args) {
             i.args = py_to_json(args);
        })
        .def("get_arg_str", [](const Instruction& i, const std::string& key) {
            if (i.args.contains(key) && i.args[key].is_string()) return i.args[key].get<std::string>();
            return std::string("");
        })
        .def("get_arg_int", [](const Instruction& i, const std::string& key) {
            if (i.args.contains(key) && i.args[key].is_number()) return i.args[key].get<int>();
            return 0;
        })
        .def("get_then_block_size", [](const Instruction& i) { return (int)i.then_block.size(); })
        .def("get_then_instruction", [](const Instruction& i, int index) { return i.then_block[index]; })
        .def("get_else_block_size", [](const Instruction& i) { return (int)i.else_block.size(); })
        .def("get_else_instruction", [](const Instruction& i, int index) { return i.else_block[index]; });

    py::class_<GameState::QueryContext>(m, "QueryContext")
        .def_readwrite("query_id", &GameState::QueryContext::query_id)
        .def_readwrite("query_type", &GameState::QueryContext::query_type)
        .def_readwrite("params", &GameState::QueryContext::params)
        .def_readwrite("valid_targets", &GameState::QueryContext::valid_targets)
        .def_readwrite("options", &GameState::QueryContext::options);

    py::class_<TurnStats>(m, "TurnStats")
         .def_readwrite("cards_drawn_this_turn", &TurnStats::cards_drawn_this_turn);

    // Added CardStats binding
    py::class_<dm::core::CardStats>(m, "CardStats")
        .def(py::init<>())
        .def_readwrite("play_count", &dm::core::CardStats::play_count)
        .def_readwrite("win_count", &dm::core::CardStats::win_count)
        .def_readwrite("sum_cost_discount", &dm::core::CardStats::sum_cost_discount)
        .def_readwrite("sum_early_usage", &dm::core::CardStats::sum_early_usage)
        .def_readwrite("sum_late_usage", &dm::core::CardStats::sum_late_usage)
        .def_readwrite("mana_usage_count", &dm::core::CardStats::mana_usage_count)
        .def_readwrite("shield_trigger_count", &dm::core::CardStats::shield_trigger_count)
        .def_readwrite("hand_play_count", &dm::core::CardStats::hand_play_count)
        .def_readwrite("sum_win_contribution", &dm::core::CardStats::sum_win_contribution);

    py::class_<Player>(m, "Player")
        .def_readwrite("hand", &Player::hand)
        .def_readwrite("mana_zone", &Player::mana_zone)
        .def_readwrite("battle_zone", &Player::battle_zone)
        .def_readwrite("shield_zone", &Player::shield_zone)
        .def_readwrite("graveyard", &Player::graveyard)
        .def_readwrite("deck", &Player::deck)
        .def_readwrite("stack", &Player::stack)
        .def_readwrite("effect_buffer", &Player::effect_buffer);

    // Bind CommandSystem helper struct to execute commands from Python
    struct CommandSystemWrapper {
        static void execute_command(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, py::dict py_ctx) {
            std::map<std::string, int> ctx;
            // Convert python dict to map
            for (auto item : py_ctx) {
                if (py::isinstance<py::str>(item.first) && py::isinstance<py::int_>(item.second)) {
                    ctx[item.first.cast<std::string>()] = item.second.cast<int>();
                }
            }
            dm::engine::systems::CommandSystem::execute_command(state, cmd, source_instance_id, player_id, ctx);

            // Update python dict back? (pybind11 dict is reference, so modifying it in place might work if we wrapped map)
            // But we created a temporary map.
            // If the user wants output, they should pass a dict and we update it.
            for (const auto& pair : ctx) {
                py_ctx[py::str(pair.first)] = pair.second;
            }
        }
    };

    py::class_<CommandSystemWrapper>(m, "CommandSystem")
        .def_static("execute_command", &CommandSystemWrapper::execute_command);

    py::class_<GameState, std::shared_ptr<GameState>>(m, "GameState")
        .def(py::init<int>())
        .def("setup_test_duel", &GameState::setup_test_duel)
        .def("execute_command", &GameState::execute_command)
        .def("add_card_to_zone", &GameState::add_card_to_zone)
        .def("register_card_instance", &GameState::register_card_instance)
        .def("undo", &GameState::undo)
        .def_readonly("command_history", &GameState::command_history)
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("current_phase", &GameState::current_phase)
        .def_readwrite("players", &GameState::players)
        .def_readwrite("game_over", &GameState::game_over)
        .def_readwrite("winner", &GameState::winner)
        .def_readwrite("active_modifiers", &GameState::active_modifiers)
        .def_readwrite("passive_effects", &GameState::passive_effects)
        .def_readwrite("turn_stats", &GameState::turn_stats)
        .def_readwrite("waiting_for_user_input", &GameState::waiting_for_user_input)
        .def_readwrite("pending_query", &GameState::pending_query)
        .def_readwrite("status", &GameState::status)
        .def_property_readonly("command_system", [](GameState& s) { return CommandSystemWrapper(); }) // Expose helper
        .def("get_pending_effect_count", [](const GameState& s) { return s.pending_effects.size(); })
        .def("create_observer_view", &GameState::create_observer_view)
        .def("clone", &GameState::clone)
        .def("get_card_instance", [](GameState& s, int id) {
            try {
                return s.get_card_instance(id);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in get_card_instance: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in get_card_instance");
            }
        }, py::return_value_policy::reference)
        .def("get_zone", &GameState::get_zone)
        .def("set_deck", [](GameState& s, PlayerID pid, std::vector<int> ids) {
            try {
                 s.players[pid].deck.clear();
                 int counter = 0;
                 for (int id : ids) s.players[pid].deck.push_back(CardInstance(id, counter++, pid));
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in set_deck: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in set_deck");
            }
        })
        .def("add_test_card_to_battle", [](GameState& s, PlayerID pid, int cid, int iid, bool tapped, bool sick) {
            try {
                 CardInstance c(cid, iid, pid);
                 c.is_tapped = tapped;
                 c.summoning_sickness = sick;
                 s.players[pid].battle_zone.push_back(c);
                 // Update owner map
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 100, 0);
                 s.card_owner_map[iid] = pid;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in add_test_card_to_battle: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in add_test_card_to_battle");
            }
        })
        .def("add_card_to_hand", [](GameState& s, PlayerID pid, int cid, int iid) {
            try {
                 CardInstance c(cid, iid, pid);
                 s.players[pid].hand.push_back(c);
                 // Update owner map to allow lookup
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 100, 0);
                 s.card_owner_map[iid] = pid;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in add_card_to_hand: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in add_card_to_hand");
            }
        })
        .def("add_card_to_mana", [](GameState& s, PlayerID pid, int cid, int iid) {
            try {
                 CardInstance c(cid, iid, pid);
                 s.add_card_to_zone(c, Zone::MANA, pid);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in add_card_to_mana: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in add_card_to_mana");
            }
        })
        .def("add_card_to_deck", [](GameState& s, PlayerID pid, int cid, int iid) {
            try {
                 CardInstance c(cid, iid, pid);
                 s.add_card_to_zone(c, Zone::DECK, pid);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in add_card_to_deck: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in add_card_to_deck");
            }
        })
        .def("add_card_to_shield", [](GameState& s, PlayerID pid, int cid, int iid) {
            try {
                 CardInstance c(cid, iid, pid);
                 s.add_card_to_zone(c, Zone::SHIELD, pid);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in add_card_to_shield: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in add_card_to_shield");
            }
        })
        .def("add_card_to_graveyard", [](GameState& s, PlayerID pid, int cid, int iid) {
            try {
                 CardInstance c(cid, iid, pid);
                 s.add_card_to_zone(c, Zone::GRAVEYARD, pid);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in add_card_to_graveyard: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in add_card_to_graveyard");
            }
        })
        .def("initialize_card_stats", &GameState::initialize_card_stats)
        .def("on_card_play", &GameState::on_card_play)
        .def("vectorize_card_stats", &GameState::vectorize_card_stats)
        .def("calculate_hash", &GameState::calculate_hash)
        .def("get_pending_effects_info", [](const GameState& s) {
            try {
                py::list list;
                for (const auto& pe : s.pending_effects) {
                    py::dict d;
                    d["type"] = pe.type;
                    d["source_instance_id"] = pe.source_instance_id;
                    d["controller"] = pe.controller;
                    d["resolve_type"] = pe.resolve_type;
                    list.append(d);
                }
                return list;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in get_pending_effects_info: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in get_pending_effects_info");
            }
        })
        .def("add_passive_effect", [](GameState& s, const PassiveEffect& p) {
            s.passive_effects.push_back(p);
        });

    m.def("initialize_card_stats", [](GameState& state, const std::map<dm::core::CardID, CardDefinition>& db, int count) {
         state.initialize_card_stats(db, count);
    });

    m.def("get_card_stats", [](const GameState& state) {
        try {
            py::dict result;
            for (const auto& [cid, stats] : state.global_card_stats) {
                py::dict s;
                s["play_count"] = stats.play_count;
                s["win_count"] = stats.win_count;
                s["sum_cost_discount"] = stats.sum_cost_discount;
                s["sum_early_usage"] = stats.sum_early_usage;
                s["sum_win_contribution"] = stats.sum_win_contribution;
                result[py::int_(cid)] = s;
            }
            return result;
        } catch (const py::error_already_set& e) {
            throw;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error in get_card_stats: " + std::string(e.what()));
        } catch (...) {
            throw std::runtime_error("Unknown error in get_card_stats");
        }
    });

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("card_id", &Action::card_id)
        .def_readwrite("source_instance_id", &Action::source_instance_id)
        .def_readwrite("target_instance_id", &Action::target_instance_id)
        .def_readwrite("target_player", &Action::target_player)
        .def_readwrite("slot_index", &Action::slot_index)
        .def_readwrite("target_slot_index", &Action::target_slot_index)
        .def("to_string", &Action::to_string);
}
