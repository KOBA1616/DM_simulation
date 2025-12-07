#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../core/game_state.hpp"
#include "../core/card_def.hpp"
#include "../engine/mana/mana_system.hpp"
#include "../engine/action_gen/action_generator.hpp"
#include "../engine/effect_resolver.hpp"
#include "../engine/game_instance.hpp"
#include "../engine/json_loader.hpp"
#include "../core/scenario_config.hpp"
#include "../engine/card_system/generic_card_system.hpp"
#include "../engine/utils/game_utils.hpp"
#include "../engine/card_system/target_utils.hpp"
#include "../core/game_state_tracking.cpp"

namespace py = pybind11;
using namespace dm::core;
using namespace dm::engine;

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

    py::enum_<Phase>(m, "Phase")
        .value("START_OF_TURN", Phase::START_OF_TURN)
        .value("DRAW", Phase::DRAW)
        .value("MANA", Phase::MANA)
        .value("MAIN", Phase::MAIN)
        .value("ATTACK", Phase::ATTACK)
        .value("BLOCK", Phase::BLOCK)
        .value("END_OF_TURN", Phase::END_OF_TURN)
        .export_values();

    py::enum_<Zone>(m, "Zone")
        .value("DECK", Zone::DECK)
        .value("HAND", Zone::HAND)
        .value("MANA", Zone::MANA_ZONE) // Mapped to MANA_ZONE in Py if needed, checking existing code use
        .value("BATTLE", Zone::BATTLE_ZONE)
        .value("GRAVEYARD", Zone::GRAVEYARD)
        .value("SHIELD", Zone::SHIELD_ZONE)
        .export_values();

    py::enum_<EffectActionType>(m, "EffectActionType")
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
        .export_values();

    py::enum_<TriggerType>(m, "TriggerType")
        .value("ON_PLAY", TriggerType::ON_PLAY)
        .value("ON_ATTACK", TriggerType::ON_ATTACK)
        .value("ON_DESTROY", TriggerType::ON_DESTROY)
        .value("S_TRIGGER", TriggerType::S_TRIGGER)
        .value("TURN_START", TriggerType::TURN_START)
        .value("PASSIVE_CONST", TriggerType::PASSIVE_CONST)
        .value("ON_BLOCK", TriggerType::ON_BLOCK)
        .export_values();

    py::enum_<TargetScope>(m, "TargetScope")
        .value("SELF", TargetScope::SELF)
        .value("PLAYER_SELF", TargetScope::PLAYER_SELF)
        .value("PLAYER_OPPONENT", TargetScope::PLAYER_OPPONENT)
        .value("ALL_PLAYERS", TargetScope::ALL_PLAYERS)
        .value("TARGET_SELECT", TargetScope::TARGET_SELECT)
        .value("RANDOM", TargetScope::RANDOM)
        .value("ALL_FILTERED", TargetScope::ALL_FILTERED)
        .export_values();

    py::enum_<ActionType>(m, "ActionType")
        .value("PASS", ActionType::PASS)
        .value("PLAY_CARD", ActionType::PLAY_CARD)
        .value("ACTIVATE_SHIELD_TRIGGER", ActionType::USE_SHIELD_TRIGGER)
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
        .export_values();

    // Structs
    py::class_<CardKeywords>(m, "CardKeywords")
        .def(py::init<>())
        .def_readwrite("g_zero", &CardKeywords::g_zero)
        .def_readwrite("revolution_change", &CardKeywords::revolution_change)
        .def_readwrite("mach_fighter", &CardKeywords::mach_fighter)
        .def_readwrite("g_strike", &CardKeywords::g_strike)
        .def_readwrite("speed_attacker", &CardKeywords::speed_attacker)
        .def_readwrite("blocker", &CardKeywords::blocker)
        .def_readwrite("slayer", &CardKeywords::slayer)
        .def_readwrite("double_breaker", &CardKeywords::double_breaker)
        .def_readwrite("triple_breaker", &CardKeywords::triple_breaker)
        .def_readwrite("shield_trigger", &CardKeywords::shield_trigger)
        .def_readwrite("evolution", &CardKeywords::evolution)
        .def_readwrite("cip", &CardKeywords::cip)
        .def_readwrite("at_attack", &CardKeywords::at_attack)
        .def_readwrite("destruction", &CardKeywords::destruction)
        .def_readwrite("just_diver", &CardKeywords::just_diver)
        .def_readwrite("hyper_energy", &CardKeywords::hyper_energy)
        .def_readwrite("shield_burn", &CardKeywords::shield_burn)
        .def_readwrite("untap_in", &CardKeywords::untap_in);

    py::class_<FilterDef>(m, "FilterDef")
        .def(py::init<>())
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
        .def_readwrite("type", &ActionDef::type)
        .def_readwrite("filter", &ActionDef::filter)
        .def_readwrite("value1", &ActionDef::value1)
        .def_readwrite("value2", &ActionDef::value2)
        .def_readwrite("str_val", &ActionDef::str_val)
        .def_readwrite("optional", &ActionDef::optional)
        .def_readwrite("input_value_key", &ActionDef::input_value_key)
        .def_readwrite("output_value_key", &ActionDef::output_value_key);

    py::class_<ConditionDef>(m, "ConditionDef")
        .def(py::init<>())
        .def_readwrite("type", &ConditionDef::type)
        .def_readwrite("value", &ConditionDef::value)
        .def_readwrite("str_val", &ConditionDef::str_val);

    py::class_<EffectDef>(m, "EffectDef")
        .def(py::init<>())
        .def_readwrite("trigger", &EffectDef::trigger)
        .def_readwrite("condition", &EffectDef::condition)
        .def_readwrite("actions", &EffectDef::actions);

    py::class_<CardDefinition>(m, "CardDefinition")
        .def(py::init<>())
        .def_readwrite("id", &CardDefinition::id)
        .def_readwrite("name", &CardDefinition::name)
        .def_readwrite("cost", &CardDefinition::cost)
        .def_readwrite("power", &CardDefinition::power)
        .def_readwrite("type", &CardDefinition::type)
        .def_readwrite("races", &CardDefinition::races)
        .def_readwrite("keywords", &CardDefinition::keywords)
        .def_readwrite("civilizations", &CardDefinition::civilizations) // Changed to list
        .def_property("civilization",
             [](const CardDefinition& c) {
                 return c.civilizations.empty() ? Civilization::NONE : c.civilizations[0];
             },
             [](CardDefinition& c, Civilization civ) {
                 c.civilizations = {civ};
             }
        ); // Backward compatibility property for existing scripts

    py::class_<CardData>(m, "CardData")
        .def(py::init([](int id, std::string name, int cost, std::string civ, int power, std::string type, std::vector<std::string> races, std::vector<EffectDef> effects) {
             CardData d;
             d.id = id; d.name = name; d.cost = cost;
             d.civilizations = {civ}; // Constructor shim for legacy single-civ strings
             d.power = power; d.type = type; d.races = races; d.effects = effects;
             return d;
        }))
        // Overload for list of civs
        .def(py::init([](int id, std::string name, int cost, std::vector<std::string> civs, int power, std::string type, std::vector<std::string> races, std::vector<EffectDef> effects) {
             CardData d;
             d.id = id; d.name = name; d.cost = cost; d.civilizations = civs;
             d.power = power; d.type = type; d.races = races; d.effects = effects;
             return d;
        }))
        .def_readwrite("id", &CardData::id)
        .def_readwrite("effects", &CardData::effects)
        .def_readwrite("revolution_change_condition", &CardData::revolution_change_condition)
        .def_readwrite("keywords", &CardData::keywords);

    py::class_<CardInstance>(m, "CardInstance")
        .def(py::init<>())
        .def_readwrite("id", &CardInstance::instance_id) // Renamed for clarity, map to instance_id
        .def_readwrite("card_id", &CardInstance::card_id)
        .def_readwrite("is_tapped", &CardInstance::is_tapped)
        .def_readwrite("summoning_sickness", &CardInstance::summoning_sickness)
        .def_readwrite("is_face_down", &CardInstance::is_face_down)
        .def_readwrite("turn_played", &CardInstance::turn_played);

    py::class_<Player>(m, "Player")
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
        .def_readwrite("target_slot_index", &Action::target_slot_index) // Used for mana payment
        .def_readwrite("mana_payment", &Action::mana_payment); // std::vector<int>

    py::class_<GameState>(m, "GameState")
        .def(py::init<int>())
        .def_readwrite("players", &GameState::players)
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("effect_buffer", &GameState::effect_buffer)
        .def("get_card_def", [](GameState& s, CardID id, const std::map<CardID, CardDefinition>& db) {
            return db.at(id);
        })
        .def("add_card_to_hand", &GameState::add_card_to_hand)
        .def("add_card_to_mana", &GameState::add_card_to_mana)
        .def("add_card_to_deck", &GameState::add_card_to_deck)
        .def("add_test_card_to_battle", [](GameState& s, PlayerID pid, CardID cid, int iid, bool tapped, bool sick) {
             CardInstance c;
             c.card_id = cid;
             c.instance_id = iid;
             c.is_tapped = tapped;
             c.summoning_sickness = sick;
             c.turn_played = s.turn_number; // Default to current turn
             s.players[pid].battle_zone.push_back(c);
        })
        .def("calculate_hash", &GameState::calculate_hash);

    // Classes / Systems
    py::class_<JsonLoader>(m, "JsonLoader")
        .def_static("load_cards", &JsonLoader::load_cards);

    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("start_turn", &PhaseManager::start_turn)
        .def_static("next_phase", &PhaseManager::next_phase);

    py::class_<ActionGenerator>(m, "ActionGenerator")
        .def_static("generate_legal_actions", &ActionGenerator::generate_legal_actions);

    py::class_<EffectResolver>(m, "EffectResolver")
        .def_static("resolve_action", &EffectResolver::resolve_action)
        .def_static("resolve_trigger", &EffectResolver::resolve_trigger);

    py::class_<ManaSystem>(m, "ManaSystem")
        .def_static("can_pay_cost",
             static_cast<bool (*)(const GameState&, const Player&, const CardDefinition&, const std::map<CardID, CardDefinition>&)>(&ManaSystem::can_pay_cost))
        .def_static("auto_tap_mana",
             static_cast<bool (*)(GameState&, Player&, const CardDefinition&, const std::map<CardID, CardDefinition>&)>(&ManaSystem::auto_tap_mana))
        .def_static("get_adjusted_cost", &ManaSystem::get_adjusted_cost);

    py::class_<GenericCardSystem>(m, "GenericCardSystem")
        .def_static("resolve_action_with_context", [](GameState& state, PlayerID pid, const Action& action, const std::map<CardID, CardDefinition>& db, std::map<std::string, int> ctx) {
             GenericCardSystem::resolve_action_with_context(state, pid, action, db, ctx);
             return ctx;
        })
        .def_static("resolve_effect_with_targets", [](GameState& state, PlayerID pid, EffectDef& effect, const std::map<CardID, CardDefinition>& db, std::map<std::string, int> ctx) {
             GenericCardSystem::resolve_effect_with_targets(state, pid, effect, db, ctx);
             return ctx; // Return context as it might be modified
        });

    py::class_<CardRegistry>(m, "CardRegistry")
        .def_static("get_instance", &CardRegistry::get_instance, py::return_value_policy::reference)
        .def("register", &CardRegistry::register_card)
        .def("load_from_json", &CardRegistry::load_from_json)
        .def("get_all", &CardRegistry::get_all);

    // AI & Training
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
        .def(py::init<>())
        .def("run_scenario", &ScenarioExecutor::run_scenario);

    py::class_<ParallelRunner>(m, "ParallelRunner")
         .def(py::init<const std::map<CardID, CardDefinition>&, const std::string&>())
         .def("play_games", &ParallelRunner::play_games)
         .def("play_scenario_match", &ParallelRunner::play_scenario_match)
         .def("play_deck_matchup", &ParallelRunner::play_deck_matchup);

    // Stats
    py::class_<CardStats>(m, "CardStats")
        .def_readwrite("play_count", &CardStats::play_count)
        .def_readwrite("win_count", &CardStats::win_count)
        .def_readwrite("sum_cost_discount", &CardStats::sum_cost_discount)
        .def_readwrite("sum_early_usage", &CardStats::sum_early_usage)
        .def_readwrite("sum_win_contribution", &CardStats::sum_win_contribution);

    m.def("initialize_card_stats", &initialize_card_stats);
    m.def("get_card_stats", &get_card_stats);

    // Debug
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
}
