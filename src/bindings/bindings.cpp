#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../core/game_state.hpp"
#include "../core/card_def.hpp"
#include "../engine/systems/mana/mana_system.hpp"
#include "../engine/actions/action_generator.hpp"
#include "../engine/effects/effect_resolver.hpp"
#include "../engine/game_instance.hpp"
#include "../engine/systems/card/json_loader.hpp"
#include "../core/scenario_config.hpp"
#include "../engine/systems/card/generic_card_system.hpp"
#include "../engine/utils/zone_utils.hpp"
#include "../engine/systems/card/target_utils.hpp"
#include "../engine/systems/flow/phase_manager.hpp"
#include "../engine/systems/card/card_registry.hpp"
#include "../ai/scenario/scenario_executor.hpp"
#include "../ai/self_play/parallel_runner.hpp"
#include "../ai/evaluator/beam_search_evaluator.hpp"
#include "../ai/encoders/action_encoder.hpp"
#include "../ai/encoders/tensor_converter.hpp"
#include "../ai/pomdp/pomdp.hpp"
#include "../ai/pomdp/parametric_belief.hpp"
#include "../ai/evaluator/neural_evaluator.hpp"
#include "../core/card_stats.hpp"
#include "../core/game_state_tracking.cpp"
#include "../engine/utils/dev_tools.hpp"
#include "../utils/csv_loader.hpp"
#include "python_batch_inference.hpp"
#include "../ai/solver/lethal_solver.hpp"

namespace py = pybind11;
using namespace dm::core;
using namespace dm::engine;
using namespace dm::ai;

// Helpers for Pybind
std::string civilization_to_string(Civilization c) {
    if (c == Civilization::LIGHT) return "LIGHT";
    if (c == Civilization::WATER) return "WATER";
    if (c == Civilization::DARKNESS) return "DARKNESS";
    if (c == Civilization::FIRE) return "FIRE";
    if (c == Civilization::NATURE) return "NATURE";
    if (c == Civilization::ZERO) return "ZERO";
    return "NONE";
}

Civilization string_to_civilization(const std::string& s) {
    if (s == "LIGHT") return Civilization::LIGHT;
    if (s == "WATER") return Civilization::WATER;
    if (s == "DARKNESS") return Civilization::DARKNESS;
    if (s == "FIRE") return Civilization::FIRE;
    if (s == "NATURE") return Civilization::NATURE;
    if (s == "ZERO") return Civilization::ZERO;
    return Civilization::NONE;
}

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Engine Module";

    // Enums
    py::enum_<Civilization>(m, "Civilization")
        .value("NONE", Civilization::NONE)
        .value("LIGHT", Civilization::LIGHT)
        .value("WATER", Civilization::WATER)
        .value("DARKNESS", Civilization::DARKNESS)
        .value("FIRE", Civilization::FIRE)
        .value("NATURE", Civilization::NATURE)
        .value("ZERO", Civilization::ZERO)
        .export_values();

    py::enum_<CardType>(m, "CardType")
        .value("CREATURE", CardType::CREATURE)
        .value("SPELL", CardType::SPELL)
        .export_values();

    py::enum_<GameResult>(m, "GameResult")
        .value("NONE", GameResult::NONE)
        .value("P1_WIN", GameResult::P1_WIN)
        .value("P2_WIN", GameResult::P2_WIN)
        .value("DRAW", GameResult::DRAW)
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

    py::enum_<SpawnSource>(m, "SpawnSource")
        .value("HAND_SUMMON", SpawnSource::HAND_SUMMON)
        .value("EFFECT_SUMMON", SpawnSource::EFFECT_SUMMON)
        .value("EFFECT_PUT", SpawnSource::EFFECT_PUT)
        .export_values();

    py::enum_<Zone>(m, "Zone")
        .value("DECK", Zone::DECK)
        .value("HAND", Zone::HAND)
        .value("MANA", Zone::MANA)
        .value("BATTLE", Zone::BATTLE)
        .value("GRAVEYARD", Zone::GRAVEYARD)
        .value("SHIELD", Zone::SHIELD)
        .export_values();

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
        .export_values();

    py::enum_<EffectActionType>(m, "EffectActionType")
        .value("COST_REFERENCE", EffectActionType::COST_REFERENCE)
        .value("DRAW_CARD", EffectActionType::DRAW_CARD)
        .value("ADD_MANA", EffectActionType::ADD_MANA)
        .value("DESTROY", EffectActionType::DESTROY)
        .value("RETURN_TO_HAND", EffectActionType::RETURN_TO_HAND)
        .value("SEND_TO_MANA", EffectActionType::SEND_TO_MANA)
        .value("TAP", EffectActionType::TAP)
        .value("UNTAP", EffectActionType::UNTAP)
        .value("MODIFY_POWER", EffectActionType::MODIFY_POWER)
        .value("BREAK_SHIELD", EffectActionType::BREAK_SHIELD)
        .value("LOOK_AND_ADD", EffectActionType::LOOK_AND_ADD)
        .value("SEARCH_DECK_BOTTOM", EffectActionType::SEARCH_DECK_BOTTOM)
        .value("MEKRAID", EffectActionType::MEKRAID)
        .value("REVOLUTION_CHANGE", EffectActionType::REVOLUTION_CHANGE)
        .value("COUNT_CARDS", EffectActionType::COUNT_CARDS)
        .value("GET_GAME_STAT", EffectActionType::GET_GAME_STAT)
        .value("APPLY_MODIFIER", EffectActionType::APPLY_MODIFIER)
        .value("REVEAL_CARDS", EffectActionType::REVEAL_CARDS)
        .value("RESET_INSTANCE", EffectActionType::RESET_INSTANCE)
        .value("REGISTER_DELAYED_EFFECT", EffectActionType::REGISTER_DELAYED_EFFECT)
        .value("SEARCH_DECK", EffectActionType::SEARCH_DECK)
        .value("SHUFFLE_DECK", EffectActionType::SHUFFLE_DECK)
        .value("ADD_SHIELD", EffectActionType::ADD_SHIELD)
        .value("SEND_SHIELD_TO_GRAVE", EffectActionType::SEND_SHIELD_TO_GRAVE)
        .value("SEND_TO_DECK_BOTTOM", EffectActionType::SEND_TO_DECK_BOTTOM)
        .value("MOVE_TO_UNDER_CARD", EffectActionType::MOVE_TO_UNDER_CARD)
        .value("NONE", EffectActionType::NONE)
        .export_values();

    py::enum_<TriggerType>(m, "TriggerType")
        .value("ON_PLAY", TriggerType::ON_PLAY)
        .value("ON_ATTACK", TriggerType::ON_ATTACK)
        .value("ON_DESTROY", TriggerType::ON_DESTROY)
        .value("S_TRIGGER", TriggerType::S_TRIGGER)
        .value("TURN_START", TriggerType::TURN_START)
        .value("PASSIVE_CONST", TriggerType::PASSIVE_CONST)
        .value("ON_BLOCK", TriggerType::ON_BLOCK)
        .value("NONE", TriggerType::NONE)
        .export_values();

    py::enum_<ReactionType>(m, "ReactionType")
        .value("NONE", ReactionType::NONE)
        .value("NINJA_STRIKE", ReactionType::NINJA_STRIKE)
        .value("STRIKE_BACK", ReactionType::STRIKE_BACK)
        .value("REVOLUTION_0_TRIGGER", ReactionType::REVOLUTION_0_TRIGGER)
        .export_values();

    py::enum_<TargetScope>(m, "TargetScope")
        .value("SELF", TargetScope::SELF)
        .value("PLAYER_SELF", TargetScope::PLAYER_SELF)
        .value("PLAYER_OPPONENT", TargetScope::PLAYER_OPPONENT)
        .value("ALL_PLAYERS", TargetScope::ALL_PLAYERS)
        .value("TARGET_SELECT", TargetScope::TARGET_SELECT)
        .value("RANDOM", TargetScope::RANDOM)
        .value("ALL_FILTERED", TargetScope::ALL_FILTERED)
        .value("NONE", TargetScope::NONE)
        .export_values();

    py::enum_<ActionType>(m, "ActionType")
        .value("PASS", ActionType::PASS)
        .value("PLAY_CARD", ActionType::PLAY_CARD)
        .value("USE_SHIELD_TRIGGER", ActionType::USE_SHIELD_TRIGGER)
        .value("ACTIVATE_SHIELD_TRIGGER", ActionType::USE_SHIELD_TRIGGER) // Alias
        .value("ATTACK_CREATURE", ActionType::ATTACK_CREATURE)
        .value("ATTACK_PLAYER", ActionType::ATTACK_PLAYER)
        .value("BLOCK", ActionType::BLOCK)
        .value("MANA_CHARGE", ActionType::MANA_CHARGE)
        .value("SELECT_TARGET", ActionType::SELECT_TARGET)
        .value("RESOLVE_EFFECT", ActionType::RESOLVE_EFFECT)
        .value("USE_ABILITY", ActionType::USE_ABILITY) // Revolution Change etc
        .value("PLAY_CARD_INTERNAL", ActionType::PLAY_CARD_INTERNAL)
        .value("RESOLVE_BATTLE", ActionType::RESOLVE_BATTLE)
        .value("BREAK_SHIELD", ActionType::BREAK_SHIELD)
        .value("DECLARE_REACTION", ActionType::DECLARE_REACTION)
        .value("DECLARE_PLAY", ActionType::DECLARE_PLAY)
        .value("PAY_COST", ActionType::PAY_COST)
        .value("RESOLVE_PLAY", ActionType::RESOLVE_PLAY)
        .value("MOVE_CARD", ActionType::MOVE_CARD)
        .export_values();

    // Structs
    py::class_<GameResultInfo>(m, "GameResultInfo")
        .def(py::init<>())
        .def_readwrite("result", &GameResultInfo::result)
        .def_readwrite("turn_count", &GameResultInfo::turn_count)
        .def_readwrite("states", &GameResultInfo::states)
        .def_readwrite("policies", &GameResultInfo::policies)
        .def_readwrite("active_players", &GameResultInfo::active_players);

    py::class_<PendingEffect>(m, "PendingEffect")
        .def(py::init<EffectType, int, int>())
        .def_readwrite("type", &PendingEffect::type)
        .def_readwrite("source_instance_id", &PendingEffect::source_instance_id)
        .def_readwrite("controller", &PendingEffect::controller);

    py::class_<CardKeywords>(m, "CardKeywords")
        .def(py::init<>())
        .def_property("g_zero", [](CardKeywords& k){ return k.g_zero; }, [](CardKeywords& k, bool v){ k.g_zero = v; })
        .def_property("revolution_change", [](CardKeywords& k){ return k.revolution_change; }, [](CardKeywords& k, bool v){ k.revolution_change = v; })
        .def_property("mach_fighter", [](CardKeywords& k){ return k.mach_fighter; }, [](CardKeywords& k, bool v){ k.mach_fighter = v; })
        .def_property("g_strike", [](CardKeywords& k){ return k.g_strike; }, [](CardKeywords& k, bool v){ k.g_strike = v; })
        .def_property("speed_attacker", [](CardKeywords& k){ return k.speed_attacker; }, [](CardKeywords& k, bool v){ k.speed_attacker = v; })
        .def_property("blocker", [](CardKeywords& k){ return k.blocker; }, [](CardKeywords& k, bool v){ k.blocker = v; })
        .def_property("slayer", [](CardKeywords& k){ return k.slayer; }, [](CardKeywords& k, bool v){ k.slayer = v; })
        .def_property("double_breaker", [](CardKeywords& k){ return k.double_breaker; }, [](CardKeywords& k, bool v){ k.double_breaker = v; })
        .def_property("triple_breaker", [](CardKeywords& k){ return k.triple_breaker; }, [](CardKeywords& k, bool v){ k.triple_breaker = v; })
        .def_property("shield_trigger", [](CardKeywords& k){ return k.shield_trigger; }, [](CardKeywords& k, bool v){ k.shield_trigger = v; })
        .def_property("evolution", [](CardKeywords& k){ return k.evolution; }, [](CardKeywords& k, bool v){ k.evolution = v; })
        .def_property("cip", [](CardKeywords& k){ return k.cip; }, [](CardKeywords& k, bool v){ k.cip = v; })
        .def_property("at_attack", [](CardKeywords& k){ return k.at_attack; }, [](CardKeywords& k, bool v){ k.at_attack = v; })
        .def_property("destruction", [](CardKeywords& k){ return k.destruction; }, [](CardKeywords& k, bool v){ k.destruction = v; })
        .def_property("just_diver", [](CardKeywords& k){ return k.just_diver; }, [](CardKeywords& k, bool v){ k.just_diver = v; })
        .def_property("hyper_energy", [](CardKeywords& k){ return k.hyper_energy; }, [](CardKeywords& k, bool v){ k.hyper_energy = v; })
        .def_property("shield_burn", [](CardKeywords& k){ return k.shield_burn; }, [](CardKeywords& k, bool v){ k.shield_burn = v; })
        .def_property("untap_in", [](CardKeywords& k){ return k.untap_in; }, [](CardKeywords& k, bool v){ k.untap_in = v; })
        .def_property("meta_counter_play", [](CardKeywords& k){ return k.meta_counter_play; }, [](CardKeywords& k, bool v){ k.meta_counter_play = v; })
        .def_property("power_attacker", [](CardKeywords& k){ return k.power_attacker; }, [](CardKeywords& k, bool v){ k.power_attacker = v; });

    py::class_<FilterDef>(m, "FilterDef")
        .def(py::init<>())
        // Constructor that takes keyword arguments
        .def(py::init([](std::optional<std::string> owner, std::vector<std::string> zones, std::vector<std::string> types, std::vector<std::string> civilizations, std::vector<std::string> races, std::optional<int> min_cost, std::optional<int> max_cost, std::optional<int> min_power, std::optional<int> max_power, std::optional<bool> is_tapped, std::optional<bool> is_blocker, std::optional<bool> is_evolution, std::optional<int> count) {
            FilterDef f;
            f.owner = owner; f.zones = zones; f.types = types; f.civilizations = civilizations; f.races = races;
            f.min_cost = min_cost; f.max_cost = max_cost; f.min_power = min_power; f.max_power = max_power;
            f.is_tapped = is_tapped; f.is_blocker = is_blocker; f.is_evolution = is_evolution; f.count = count;
            return f;
        }),
        py::arg("owner") = std::nullopt,
        py::arg("zones") = std::vector<std::string>{},
        py::arg("types") = std::vector<std::string>{},
        py::arg("civilizations") = std::vector<std::string>{},
        py::arg("races") = std::vector<std::string>{},
        py::arg("min_cost") = std::nullopt,
        py::arg("max_cost") = std::nullopt,
        py::arg("min_power") = std::nullopt,
        py::arg("max_power") = std::nullopt,
        py::arg("is_tapped") = std::nullopt,
        py::arg("is_blocker") = std::nullopt,
        py::arg("is_evolution") = std::nullopt,
        py::arg("count") = std::nullopt
        )
        .def_readwrite("zones", &FilterDef::zones)
        .def_readwrite("civilizations", &FilterDef::civilizations)
        .def_readwrite("races", &FilterDef::races)
        .def_readwrite("types", &FilterDef::types)
        .def_readwrite("min_cost", &FilterDef::min_cost)
        .def_readwrite("max_cost", &FilterDef::max_cost)
        .def_readwrite("is_tapped", &FilterDef::is_tapped)
        .def_readwrite("is_blocker", &FilterDef::is_blocker)
        .def_readwrite("is_evolution", &FilterDef::is_evolution)
        .def_readwrite("owner", &FilterDef::owner)
        .def_readwrite("count", &FilterDef::count);

    py::class_<ActionDef>(m, "ActionDef")
        .def(py::init<>())
        .def(py::init([](EffectActionType type, TargetScope scope, FilterDef filter) {
            ActionDef a;
            a.type = type;
            a.scope = scope;
            a.filter = filter;
            return a;
        }), py::arg("type"), py::arg("scope") = TargetScope::NONE, py::arg("filter") = FilterDef())
        .def_readwrite("type", &ActionDef::type)
        .def_readwrite("scope", &ActionDef::scope)
        .def_readwrite("filter", &ActionDef::filter)
        .def_readwrite("value1", &ActionDef::value1)
        .def_readwrite("value2", &ActionDef::value2)
        .def_readwrite("str_val", &ActionDef::str_val)
        .def_readwrite("target_choice", &ActionDef::target_choice)
        .def_readwrite("optional", &ActionDef::optional)
        .def_readwrite("input_value_key", &ActionDef::input_value_key)
        .def_readwrite("output_value_key", &ActionDef::output_value_key);

    py::class_<ConditionDef>(m, "ConditionDef")
        .def(py::init<>())
        .def_readwrite("type", &ConditionDef::type)
        .def_readwrite("value", &ConditionDef::value)
        .def_readwrite("str_val", &ConditionDef::str_val);

    py::class_<ReactionCondition>(m, "ReactionCondition")
        .def(py::init<>())
        .def_readwrite("trigger_event", &ReactionCondition::trigger_event)
        .def_readwrite("civilization_match", &ReactionCondition::civilization_match)
        .def_readwrite("mana_count_min", &ReactionCondition::mana_count_min)
        .def_readwrite("same_civilization_shield", &ReactionCondition::same_civilization_shield);

    py::class_<EffectDef>(m, "EffectDef")
        .def(py::init<>())
        .def(py::init([](TriggerType trigger, ConditionDef condition, std::vector<ActionDef> actions) {
            EffectDef e;
            e.trigger = trigger;
            e.condition = condition;
            e.actions = actions;
            return e;
        }), py::arg("trigger"), py::arg("condition") = ConditionDef(), py::arg("actions") = std::vector<ActionDef>{})
        .def_readwrite("trigger", &EffectDef::trigger)
        .def_readwrite("condition", &EffectDef::condition)
        .def_readwrite("actions", &EffectDef::actions);

    py::class_<ReactionAbility>(m, "ReactionAbility")
        .def(py::init<>())
        .def_readwrite("type", &ReactionAbility::type)
        .def_readwrite("cost", &ReactionAbility::cost)
        .def_readwrite("zone", &ReactionAbility::zone)
        .def_readwrite("condition", &ReactionAbility::condition);

    py::class_<CardDefinition>(m, "CardDefinition")
        .def(py::init<>())
        // Add constructor matching test expectations: (id, name, civ, races, cost, power, keywords, effects)
        .def(py::init([](int id, std::string name, std::string civ, std::vector<std::string> races, int cost, int power, CardKeywords keywords, std::vector<EffectDef> effects) {
             CardDefinition d;
             d.id = id; d.name = name; d.civilizations = {string_to_civilization(civ)};
             d.races = races; d.cost = cost; d.power = power; d.keywords = keywords;
             return d;
        }))
        // Overload for Civilization enum
        .def(py::init([](int id, std::string name, Civilization civ, std::vector<std::string> races, int cost, int power, CardKeywords keywords, std::vector<EffectDef> effects) {
             CardDefinition d;
             d.id = id; d.name = name; d.civilizations = {civ};
             d.races = races; d.cost = cost; d.power = power; d.keywords = keywords;
             return d;
        }))
        .def_readwrite("id", &CardDefinition::id)
        .def_readwrite("name", &CardDefinition::name)
        .def_readwrite("cost", &CardDefinition::cost)
        .def_readwrite("power", &CardDefinition::power)
        .def_readwrite("type", &CardDefinition::type)
        .def_readwrite("races", &CardDefinition::races)
        .def_readwrite("keywords", &CardDefinition::keywords)
        .def_readwrite("metamorph_abilities", &CardDefinition::metamorph_abilities)
        .def_readwrite("revolution_change_condition", &CardDefinition::revolution_change_condition)
        .def_readwrite("is_key_card", &CardDefinition::is_key_card)
        .def_readwrite("ai_importance_score", &CardDefinition::ai_importance_score)
        .def_readwrite("civilizations", &CardDefinition::civilizations)
        .def_readwrite("reaction_abilities", &CardDefinition::reaction_abilities)
        .def_property("civilization",
             [](const CardDefinition& c) {
                 return c.civilizations.empty() ? Civilization::NONE : c.civilizations[0];
             },
             [](CardDefinition& c, Civilization civ) {
                 c.civilizations = {civ};
             }
        )
        .def_readwrite("power_attacker_bonus", &CardDefinition::power_attacker_bonus);

    py::class_<CardData>(m, "CardData")
        .def(py::init([](int id, std::string name, int cost, std::string civ, int power, std::string type, std::vector<std::string> races, std::vector<EffectDef> effects) {
             CardData d;
             d.id = id; d.name = name; d.cost = cost;
             d.civilizations = {civ};
             d.power = power; d.type = type; d.races = races; d.effects = effects;
             return d;
        }))
        .def(py::init([](int id, std::string name, int cost, std::vector<std::string> civs, int power, std::string type, std::vector<std::string> races, std::vector<EffectDef> effects) {
             CardData d;
             d.id = id; d.name = name; d.cost = cost; d.civilizations = civs;
             d.power = power; d.type = type; d.races = races; d.effects = effects;
             return d;
        }))
        .def_readwrite("id", &CardData::id)
        .def_readwrite("effects", &CardData::effects)
        .def_readwrite("metamorph_abilities", &CardData::metamorph_abilities)
        .def_readwrite("revolution_change_condition", &CardData::revolution_change_condition)
        .def_readwrite("keywords", &CardData::keywords)
        .def_readwrite("is_key_card", &CardData::is_key_card)
        .def_readwrite("ai_importance_score", &CardData::ai_importance_score)
        .def_readwrite("reaction_abilities", &CardData::reaction_abilities);
    py::class_<CardInstance>(m, "CardInstance")
        .def(py::init<>())
        .def(py::init<CardID, int>()) // Added constructor
        .def_readwrite("id", &CardInstance::instance_id)
        .def_readwrite("instance_id", &CardInstance::instance_id)
        .def_readwrite("card_id", &CardInstance::card_id)
        .def_readwrite("is_tapped", &CardInstance::is_tapped)
        .def_readwrite("summoning_sickness", &CardInstance::summoning_sickness)
        .def_readwrite("is_face_down", &CardInstance::is_face_down)
        .def_readwrite("turn_played", &CardInstance::turn_played)
        .def_readwrite("underlying_cards", &CardInstance::underlying_cards);

    py::class_<Player>(m, "Player")
        .def(py::init<>()) // Added default constructor
        .def_readwrite("id", &Player::id)
        .def_readwrite("hand", &Player::hand)
        .def_readwrite("mana_zone", &Player::mana_zone)
        .def_readwrite("battle_zone", &Player::battle_zone)
        .def_readwrite("shield_zone", &Player::shield_zone)
        .def_readwrite("graveyard", &Player::graveyard)
        .def_readwrite("deck", &Player::deck);

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("card_id", &Action::card_id)
        .def_readwrite("source_instance_id", &Action::source_instance_id)
        .def_readwrite("target_instance_id", &Action::target_instance_id)
        .def_readwrite("target_player", &Action::target_player)
        .def_readwrite("slot_index", &Action::slot_index)
        .def_readwrite("target_slot_index", &Action::target_slot_index);

    py::class_<GameState>(m, "GameState")
        .def(py::init<int>())
        .def("initialize_card_stats", &GameState::initialize_card_stats) // Bind as member
        .def_readwrite("players", &GameState::players)
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("current_phase", &GameState::current_phase)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("stack_zone", &GameState::stack_zone)
        .def_readwrite("pending_effects", &GameState::pending_effects)
        .def_readwrite("effect_buffer", &GameState::effect_buffer)
        .def_readwrite("turn_stats", &GameState::turn_stats)
        .def_readwrite("winner", &GameState::winner)
        .def_readwrite("loop_proven", &GameState::loop_proven)
        .def("get_card_def", [](GameState& s, CardID id, const std::map<CardID, CardDefinition>& db) {
            return db.at(id);
        })
        .def("add_card_to_hand", [](GameState& s, PlayerID pid, CardID cid, int iid) {
             CardInstance c; c.card_id = cid; c.instance_id = iid;
             s.players[pid].hand.push_back(c);
             if (iid >= 0) {
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 1, 255);
                 s.card_owner_map[iid] = pid;
             }
        })
        .def("clear_zone", [](GameState& s, PlayerID pid, Zone zone) {
            if (pid < 0 || pid >= 2) return;
            if (zone == Zone::SHIELD) s.players[pid].shield_zone.clear();
            else if (zone == Zone::BATTLE) s.players[pid].battle_zone.clear();
            else if (zone == Zone::HAND) s.players[pid].hand.clear();
            else if (zone == Zone::MANA) s.players[pid].mana_zone.clear();
            else if (zone == Zone::GRAVEYARD) s.players[pid].graveyard.clear();
            else if (zone == Zone::DECK) s.players[pid].deck.clear();
        })
        .def("add_card_to_mana", [](GameState& s, PlayerID pid, CardID cid, int iid) {
             CardInstance c; c.card_id = cid; c.instance_id = iid;
             s.players[pid].mana_zone.push_back(c);
             if (iid >= 0) {
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 1, 255);
                 s.card_owner_map[iid] = pid;
             }
        })
        .def("add_card_to_deck", [](GameState& s, PlayerID pid, CardID cid, int iid) {
             CardInstance c; c.card_id = cid; c.instance_id = iid;
             s.players[pid].deck.push_back(c);
             if (iid >= 0) {
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 1, 255);
                 s.card_owner_map[iid] = pid;
             }
        })
        .def("add_test_card_to_battle", [](GameState& s, PlayerID pid, CardID cid, int iid, bool tapped, bool sick) {
             CardInstance c;
             c.card_id = cid;
             c.instance_id = iid;
             c.is_tapped = tapped;
             c.summoning_sickness = sick;
             c.turn_played = s.turn_number;
             s.players[pid].battle_zone.push_back(c);
             if (iid >= 0) {
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 1, 255);
                 s.card_owner_map[iid] = pid;
             }
        })
        .def("add_test_card_to_shield", [](GameState& s, PlayerID pid, CardID cid, int iid) {
             CardInstance c;
             c.card_id = cid;
             c.instance_id = iid;
             s.players[pid].shield_zone.push_back(c);
             if (iid >= 0) {
                 if (s.card_owner_map.size() <= (size_t)iid) s.card_owner_map.resize(iid + 1, 255);
                 s.card_owner_map[iid] = pid;
             }
        })
        .def("setup_test_duel", [](GameState& s) {
            // Minimal test setup: clear zones, reset turn, give basic shields.
            for (auto& p : s.players) {
                p.hand.clear();
                p.mana_zone.clear();
                p.battle_zone.clear();
                p.shield_zone.clear();
                p.graveyard.clear();
                p.deck.clear();
            }
            s.card_owner_map.clear();
            s.turn_number = 1;
            s.active_player_id = 0;
            s.current_phase = Phase::MAIN;
            s.winner = GameResult::NONE;
            // Add five placeholder shields per player
            for (PlayerID pid = 0; pid < 2; ++pid) {
                for (int i = 0; i < 5; ++i) {
                    CardInstance c;
                    c.card_id = 0;
                    c.instance_id = 50000 + pid * 100 + i;
                    s.players[pid].shield_zone.push_back(c);
                    if (s.card_owner_map.size() <= (size_t)c.instance_id) s.card_owner_map.resize(c.instance_id + 1, 255);
                    s.card_owner_map[c.instance_id] = pid;
                }
            }
        })
        .def("set_deck", [](GameState& s, PlayerID pid, const std::vector<int>& card_ids) {
            s.players[pid].deck.clear();
            int instance_id_counter = 1000 * (pid + 1);
            for (int cid : card_ids) {
                CardInstance c;
                c.card_id = cid;
                c.instance_id = instance_id_counter++;
                s.players[pid].deck.push_back(c);
            }
        })
        .def("calculate_hash", &GameState::calculate_hash)
        .def("initialize_card_stats", &GameState::initialize_card_stats)
        .def("vectorize_card_stats", &GameState::vectorize_card_stats);

    py::class_<GameInstance>(m, "GameInstance")
        .def(py::init<uint32_t, const std::map<CardID, CardDefinition>&>())
        .def("reset_with_scenario", &GameInstance::reset_with_scenario)
        .def_readonly("state", &GameInstance::state);
        // .def_readonly("card_db", &GameInstance::card_db); // Cannot bind reference member directly easily

    py::class_<JsonLoader>(m, "JsonLoader")
        .def_static("load_cards", &JsonLoader::load_cards);

    py::class_<dm::utils::CsvLoader>(m, "CsvLoader")
        .def_static("load_cards", &dm::utils::CsvLoader::load_cards);

    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("start_turn", &PhaseManager::start_turn)
        .def_static("next_phase", &PhaseManager::next_phase);

    py::class_<ActionGenerator>(m, "ActionGenerator")
        .def_static("generate_legal_actions", &ActionGenerator::generate_legal_actions);

    py::class_<EffectResolver>(m, "EffectResolver")
        .def_static("resolve_action", &EffectResolver::resolve_action);

    py::class_<ManaSystem>(m, "ManaSystem")
        .def_static("can_pay_cost",
             static_cast<bool (*)(const GameState&, const Player&, const CardDefinition&, const std::map<CardID, CardDefinition>&)>(&ManaSystem::can_pay_cost))
        // Bind legacy overload if needed, or remove tests using it.
        // Or bind as overload.
        .def_static("can_pay_cost",
             static_cast<bool (*)(const Player&, const CardDefinition&, const std::map<CardID, CardDefinition>&)>(&ManaSystem::can_pay_cost))
        .def_static("auto_tap_mana",
             static_cast<bool (*)(GameState&, Player&, const CardDefinition&, const std::map<CardID, CardDefinition>&)>(&ManaSystem::auto_tap_mana))
        .def_static("auto_tap_mana",
             static_cast<bool (*)(Player&, const CardDefinition&, const std::map<CardID, CardDefinition>&)>(&ManaSystem::auto_tap_mana))
        .def_static("get_adjusted_cost", &ManaSystem::get_adjusted_cost);

    py::class_<GenericCardSystem>(m, "GenericCardSystem")
        .def_static("resolve_trigger", [](GameState& state, TriggerType trigger, int source_id) {
            GenericCardSystem::resolve_trigger(state, trigger, source_id);
        })
        .def_static("resolve_effect", &GenericCardSystem::resolve_effect)
        // Overload resolve_action for backward compatibility (no context)
        .def_static("resolve_action", [](GameState& state, const ActionDef& action, int source_id) {
             GenericCardSystem::resolve_action(state, action, source_id);
        })
        .def_static("resolve_action_with_context", [](GameState& state, const ActionDef& action, int source_id, std::map<std::string, int> ctx) {
             GenericCardSystem::resolve_action(state, action, source_id, ctx);
             return ctx;
        })
        .def_static("resolve_effect_with_targets", [](GameState& state, EffectDef& effect, const std::vector<int>& targets, int source_id, const std::map<CardID, CardDefinition>& db, std::map<std::string, int> ctx) {
             GenericCardSystem::resolve_effect_with_targets(state, effect, targets, source_id, db, ctx);
             return ctx;
        });

    py::class_<CardRegistry>(m, "CardRegistry")
        .def_static("get_card_data", &CardRegistry::get_card_data, py::return_value_policy::reference)
        .def_static("load_from_json", &CardRegistry::load_from_json)
        .def_static("get_all_cards", &CardRegistry::get_all_cards);

    // Helper for Python tests to register card data easily without raw JSON string construction if desired,
    // though load_from_json is the primary way.
    // Wait, the error said `cannot import name 'register_card_data'`.
    // It seems tests expect a function `register_card_data` in the module scope or CardRegistry?
    // Looking at the error: `ImportError: cannot import name 'register_card_data' from 'dm_ai_module'`
    // This implies it should be a module-level function.

    m.def("register_card_data", [](const CardData& data) {
         // Serialize back to JSON and load? Or expose a direct add method?
         // CardRegistry only has load_from_json.
         // Let's use nlohmann json to serialize.
         nlohmann::json j = data;
         // Wrap in a list as load_from_json likely expects a list or single object?
         // JsonLoader::load_cards expects a list of objects.
         // CardRegistry::load_from_json expects what?
         // Let's check CardRegistry::load_from_json implementation.
         // Assuming it handles what JsonLoader handles.
         // For safety, let's just make it a list of one.
         nlohmann::json list_j = nlohmann::json::array({j});
         CardRegistry::load_from_json(list_j.dump());
    });

    m.def("card_registry_load_from_json", &CardRegistry::load_from_json);

    py::class_<ActionEncoder>(m, "ActionEncoder")
        .def_readonly_static("TOTAL_ACTION_SIZE", &ActionEncoder::TOTAL_ACTION_SIZE)
        .def_static("action_to_index", &ActionEncoder::action_to_index);

    py::class_<TensorConverter>(m, "TensorConverter")
        .def_readonly_static("INPUT_SIZE", &TensorConverter::INPUT_SIZE)
        .def_static("convert_to_tensor", &TensorConverter::convert_to_tensor,
                    py::arg("game_state"), py::arg("player_view"), py::arg("card_db"), py::arg("mask_opponent_hand") = true)
        .def_static("convert_batch_flat", &TensorConverter::convert_batch_flat);

    py::class_<ScenarioConfig>(m, "ScenarioConfig")
        .def(py::init<>())
        .def_readwrite("my_mana", &ScenarioConfig::my_mana)
        .def_readwrite("my_hand_cards", &ScenarioConfig::my_hand_cards)
        .def_readwrite("my_battle_zone", &ScenarioConfig::my_battle_zone)
        .def_readwrite("my_mana_zone", &ScenarioConfig::my_mana_zone)
        .def_readwrite("my_grave_yard", &ScenarioConfig::my_grave_yard)
        .def_readwrite("my_shields", &ScenarioConfig::my_shields)
        .def_readwrite("enemy_shield_count", &ScenarioConfig::enemy_shield_count)
        .def_readwrite("enemy_battle_zone", &ScenarioConfig::enemy_battle_zone)
        .def_readwrite("enemy_can_use_trigger", &ScenarioConfig::enemy_can_use_trigger)
        .def_readwrite("loop_proof_mode", &ScenarioConfig::loop_proof_mode);

    py::class_<ScenarioExecutor>(m, "ScenarioExecutor")
        .def(py::init<const std::map<dm::core::CardID, dm::core::CardDefinition>&>())
        .def("run_scenario", &ScenarioExecutor::run_scenario);

    py::class_<ParallelRunner>(m, "ParallelRunner")
         .def(py::init<const std::map<CardID, CardDefinition>&, int, int>())
         .def("play_games", &ParallelRunner::play_games,
              py::arg("initial_states"),
              py::arg("evaluator"),
              py::arg("temperature") = 1.0f,
              py::arg("add_noise") = true,
              py::arg("num_threads") = 4,
              py::arg("alpha") = 0.0f)
         .def("play_scenario_match", &ParallelRunner::play_scenario_match)
         .def("play_deck_matchup", &ParallelRunner::play_deck_matchup);

    py::class_<BeamSearchEvaluator>(m, "BeamSearchEvaluator")
        .def(py::init<const std::map<dm::core::CardID, dm::core::CardDefinition>&, int, int>(),
             py::arg("card_db"), py::arg("beam_width") = 7, py::arg("max_depth") = 3)
        .def("evaluate", &BeamSearchEvaluator::evaluate);

    py::class_<POMDPInference>(m, "POMDPInference")
        .def(py::init<>())
        .def("initialize", &POMDPInference::initialize)
        .def("update_belief", &POMDPInference::update_belief)
        .def("infer_action", &POMDPInference::infer_action)
        .def("get_belief_vector", &POMDPInference::get_belief_vector);

    py::class_<ParametricBelief>(m, "ParametricBelief")
        .def(py::init<>())
        .def("initialize", &ParametricBelief::initialize)
        .def("initialize_ids", &ParametricBelief::initialize_ids)
        .def("update", &ParametricBelief::update)
        .def("update_with_prev", &ParametricBelief::update_with_prev)
        .def("get_vector", &ParametricBelief::get_vector)
        .def("set_weights", &ParametricBelief::set_weights)
        .def("get_weights", &ParametricBelief::get_weights)
        .def("set_reveal_weight", &ParametricBelief::set_reveal_weight)
        .def("get_reveal_weight", &ParametricBelief::get_reveal_weight);

    py::class_<CardStats>(m, "CardStats")
        .def_readwrite("play_count", &CardStats::play_count)
        .def_readwrite("win_count", &CardStats::win_count)
        .def_readwrite("sum_cost_discount", &CardStats::sum_cost_discount)
        .def_readwrite("sum_early_usage", &CardStats::sum_early_usage)
        .def_readwrite("sum_win_contribution", &CardStats::sum_win_contribution)
        .def("__getitem__", [](const CardStats& cs, const std::string& key) {
            if (key == "play_count") return py::cast(cs.play_count);
            if (key == "win_count") return py::cast(cs.win_count);
            if (key == "sum_cost_discount") return py::cast(cs.sum_cost_discount);
            if (key == "sum_early_usage") return py::cast(cs.sum_early_usage);
            if (key == "sum_win_contribution") return py::cast(cs.sum_win_contribution);
            if (key == "sum_late_usage") return py::cast(cs.sum_late_usage);
            if (key == "sum_trigger_rate") return py::cast(cs.sum_trigger_rate);
            if (key == "sum_hand_adv") return py::cast(cs.sum_hand_adv);
            if (key == "sum_board_adv") return py::cast(cs.sum_board_adv);
            if (key == "sum_mana_adv") return py::cast(cs.sum_mana_adv);
            if (key == "sum_shield_dmg") return py::cast(cs.sum_shield_dmg);
            if (key == "sum_hand_var") return py::cast(cs.sum_hand_var);
            if (key == "sum_board_var") return py::cast(cs.sum_board_var);
            if (key == "sum_survival_rate") return py::cast(cs.sum_survival_rate);
            if (key == "sum_effect_death") return py::cast(cs.sum_effect_death);
            if (key == "sum_comeback_win") return py::cast(cs.sum_comeback_win);
            if (key == "sum_finish_blow") return py::cast(cs.sum_finish_blow);
            if (key == "sum_deck_consumption") return py::cast(cs.sum_deck_consumption);
            throw py::key_error("Unknown CardStats key: " + key);
        });

    py::class_<TurnStats>(m, "TurnStats")
        .def(py::init<>())
        .def_readwrite("played_without_mana", &TurnStats::played_without_mana)
        .def_readwrite("cards_drawn_this_turn", &TurnStats::cards_drawn_this_turn)
        .def_readwrite("cards_discarded_this_turn", &TurnStats::cards_discarded_this_turn)
        .def_readwrite("creatures_played_this_turn", &TurnStats::creatures_played_this_turn)
        .def_readwrite("spells_cast_this_turn", &TurnStats::spells_cast_this_turn)
        .def_readwrite("attacks_declared_this_turn", &TurnStats::attacks_declared_this_turn);

    py::class_<NeuralEvaluator>(m, "NeuralEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("evaluate", &NeuralEvaluator::evaluate);

    py::class_<LethalSolver>(m, "LethalSolver")
        .def_static("is_lethal", &LethalSolver::is_lethal);

    m.def("initialize_card_stats", [](GameState& state, const std::map<CardID, CardDefinition>& db, int deck_size) {
        state.initialize_card_stats(db, deck_size);
    });

    m.def("get_card_stats", [](const GameState& state) {
        return state.global_card_stats;
    });

    m.def("vectorize_card_stats", [](const GameState& state, CardID cid) {
        return state.vectorize_card_stats(cid);
    });

    m.def("get_library_potential", [](const GameState& state) {
        return state.get_library_potential();
    });

    m.def("get_pending_effects_info", [](const GameState& state) {
        std::vector<std::tuple<EffectType, int, int>> info;
        for (const auto& eff : state.pending_effects) {
             info.emplace_back(eff.type, eff.source_instance_id, eff.controller);
        }
        return info;
    });

    m.def("trigger_loop_detection", [](GameState& state) {
        state.update_loop_check();
    });

    m.def("add_test_passive_buff", [](GameState& s, PlayerID pid, int value) {
         PassiveEffect pe;
         pe.type = PassiveType::POWER_MODIFIER;
         pe.value = value;
         pe.controller = pid;
         pe.condition.type = "NONE";
         pe.target_filter.owner = "SELF";
         pe.target_filter.zones = {"BATTLE_ZONE"};
         s.passive_effects.push_back(pe);
    });

    py::class_<DevTools>(m, "DevTools")
        .def_static("move_cards", &DevTools::move_cards)
        .def_static("trigger_loop_detection", &DevTools::trigger_loop_detection);

    m.def("register_batch_inference", &dm::python::set_batch_callback);
    m.def("register_batch_inference_numpy", &dm::python::set_flat_batch_callback);
    m.def("has_batch_inference_registered", &dm::python::has_batch_callback);
    m.def("has_flat_batch_inference_registered", &dm::python::has_flat_batch_callback);
}
