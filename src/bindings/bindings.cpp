#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/game_instance.hpp"
#include "engine/actions/action_generator.hpp"
// #include "engine/game/effect_resolver.hpp" // Removed
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "ai/mcts/mcts.hpp"
#include "ai/evaluator/heuristic_evaluator.hpp" // Typo fix evaluator/evaluators
#include "engine/utils/determinizer.hpp" // Typo fix ai/utils -> engine/utils
#include "core/card_json_types.hpp"
#include "engine/systems/card/json_loader.hpp"
#include "ai/self_play/parallel_runner.hpp" // Typo fix mcts/parallel_runner
#include "engine/systems/flow/phase_manager.hpp" // Typo fix game/phase_manager
#include "core/card_stats.hpp"
#include "ai/solver/lethal_solver.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/cost_payment_system.hpp" // Typo fix systems/cost... -> engine/cost...
#include "ai/self_play/self_play.hpp" // Added to include GameResultInfo definition
#include "ai/scenario/scenario_executor.hpp" // Missing include
#include "core/instruction.hpp" // For Instruction and InstructionOp
#include "engine/game_command/commands.hpp" // Added for GameCommand bindings
#include "ai/encoders/token_converter.hpp" // Phase 8: Feature Tokenization
#include "ai/encoders/tensor_converter.hpp" // Tensor Converter
#include "ai/encoders/action_encoder.hpp" // Action Encoder
#include "ai/inference/deck_inference.hpp" // Phase 2: Deck Inference
#include "ai/pomdp/parametric_belief.hpp" // Added for ParametricBelief

namespace py = pybind11;
using namespace dm;
using namespace dm::core;
using namespace dm::engine;
using namespace dm::ai;

// Helper to access pipeline
std::shared_ptr<dm::engine::systems::PipelineExecutor> get_active_pipeline(GameState& state) {
    return std::static_pointer_cast<dm::engine::systems::PipelineExecutor>(state.active_pipeline);
}

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Module";

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

    // GameCommand bindings
    // Do NOT use using namespace dm::engine::game_command; locally to avoid CommandType conflict with core::CommandType

    py::class_<dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::GameCommand>>(m, "GameCommand")
        .def("get_type", &dm::engine::game_command::GameCommand::get_type);

    py::enum_<dm::engine::game_command::CommandType>(m, "CommandType")
        .value("TRANSITION", dm::engine::game_command::CommandType::TRANSITION)
        .value("MUTATE", dm::engine::game_command::CommandType::MUTATE)
        .value("ATTACH", dm::engine::game_command::CommandType::ATTACH)
        .value("FLOW", dm::engine::game_command::CommandType::FLOW)
        .value("QUERY", dm::engine::game_command::CommandType::QUERY)
        .value("DECIDE", dm::engine::game_command::CommandType::DECIDE)
        .value("DECLARE_REACTION", dm::engine::game_command::CommandType::DECLARE_REACTION)
        .value("STAT", dm::engine::game_command::CommandType::STAT)
        .value("GAME_RESULT", dm::engine::game_command::CommandType::GAME_RESULT)
        .export_values();

    py::class_<dm::engine::game_command::TransitionCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::TransitionCommand>>(m, "TransitionCommand")
        .def(py::init<int, Zone, Zone, PlayerID, int>(),
             py::arg("instance_id"), py::arg("from"), py::arg("to"), py::arg("owner"), py::arg("dest_idx") = -1)
        .def_readwrite("card_instance_id", &dm::engine::game_command::TransitionCommand::card_instance_id)
        .def_readwrite("from_zone", &dm::engine::game_command::TransitionCommand::from_zone)
        .def_readwrite("to_zone", &dm::engine::game_command::TransitionCommand::to_zone)
        .def_readwrite("owner_id", &dm::engine::game_command::TransitionCommand::owner_id)
        .def_readwrite("destination_index", &dm::engine::game_command::TransitionCommand::destination_index);

    py::class_<dm::engine::game_command::MutateCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::MutateCommand>>(m, "MutateCommand")
        .def(py::init<int, dm::engine::game_command::MutateCommand::MutationType, int, std::string>(),
             py::arg("instance_id"), py::arg("type"), py::arg("val") = 0, py::arg("str") = "")
        .def_readwrite("target_instance_id", &dm::engine::game_command::MutateCommand::target_instance_id)
        .def_readwrite("mutation_type", &dm::engine::game_command::MutateCommand::mutation_type)
        .def_readwrite("int_value", &dm::engine::game_command::MutateCommand::int_value)
        .def_readwrite("str_value", &dm::engine::game_command::MutateCommand::str_value);

    py::enum_<dm::engine::game_command::MutateCommand::MutationType>(m, "MutationType")
        .value("TAP", dm::engine::game_command::MutateCommand::MutationType::TAP)
        .value("UNTAP", dm::engine::game_command::MutateCommand::MutationType::UNTAP)
        .value("POWER_MOD", dm::engine::game_command::MutateCommand::MutationType::POWER_MOD)
        .value("ADD_KEYWORD", dm::engine::game_command::MutateCommand::MutationType::ADD_KEYWORD)
        .value("REMOVE_KEYWORD", dm::engine::game_command::MutateCommand::MutationType::REMOVE_KEYWORD)
        .value("ADD_PASSIVE_EFFECT", dm::engine::game_command::MutateCommand::MutationType::ADD_PASSIVE_EFFECT)
        .value("ADD_COST_MODIFIER", dm::engine::game_command::MutateCommand::MutationType::ADD_COST_MODIFIER)
        .value("ADD_PENDING_EFFECT", dm::engine::game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT)
        .export_values();

    py::class_<dm::engine::game_command::AttachCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::AttachCommand>>(m, "AttachCommand")
        .def(py::init<int, int, Zone>())
        .def_readwrite("card_to_attach_id", &dm::engine::game_command::AttachCommand::card_to_attach_id)
        .def_readwrite("target_base_card_id", &dm::engine::game_command::AttachCommand::target_base_card_id)
        .def_readwrite("source_zone", &dm::engine::game_command::AttachCommand::source_zone);

    py::class_<dm::engine::game_command::FlowCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::FlowCommand>>(m, "FlowCommand")
        .def(py::init<dm::engine::game_command::FlowCommand::FlowType, int>())
        .def_readwrite("flow_type", &dm::engine::game_command::FlowCommand::flow_type)
        .def_readwrite("new_value", &dm::engine::game_command::FlowCommand::new_value);

    py::enum_<dm::engine::game_command::FlowCommand::FlowType>(m, "FlowType")
        .value("PHASE_CHANGE", dm::engine::game_command::FlowCommand::FlowType::PHASE_CHANGE)
        .value("TURN_CHANGE", dm::engine::game_command::FlowCommand::FlowType::TURN_CHANGE)
        .value("STEP_CHANGE", dm::engine::game_command::FlowCommand::FlowType::STEP_CHANGE)
        .value("SET_ATTACK_SOURCE", dm::engine::game_command::FlowCommand::FlowType::SET_ATTACK_SOURCE)
        .value("SET_ATTACK_TARGET", dm::engine::game_command::FlowCommand::FlowType::SET_ATTACK_TARGET)
        .value("SET_ATTACK_PLAYER", dm::engine::game_command::FlowCommand::FlowType::SET_ATTACK_PLAYER)
        .value("SET_ACTIVE_PLAYER", dm::engine::game_command::FlowCommand::FlowType::SET_ACTIVE_PLAYER)
        .export_values();

    py::class_<dm::engine::game_command::QueryCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::QueryCommand>>(m, "QueryCommand")
        .def(py::init<std::string, std::vector<int>, std::map<std::string, int>>())
        .def_readwrite("query_type", &dm::engine::game_command::QueryCommand::query_type)
        .def_readwrite("valid_targets", &dm::engine::game_command::QueryCommand::valid_targets)
        .def_readwrite("params", &dm::engine::game_command::QueryCommand::params);

    py::class_<dm::engine::game_command::DecideCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::DecideCommand>>(m, "DecideCommand")
        .def(py::init<int, std::vector<int>, int>())
        .def_readwrite("query_id", &dm::engine::game_command::DecideCommand::query_id)
        .def_readwrite("selected_indices", &dm::engine::game_command::DecideCommand::selected_indices)
        .def_readwrite("selected_option_index", &dm::engine::game_command::DecideCommand::selected_option_index);

    py::class_<dm::engine::game_command::DeclareReactionCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::DeclareReactionCommand>>(m, "DeclareReactionCommand")
        .def(py::init<PlayerID, bool, int>())
        .def_readwrite("pass", &dm::engine::game_command::DeclareReactionCommand::pass)
        .def_readwrite("reaction_index", &dm::engine::game_command::DeclareReactionCommand::reaction_index)
        .def_readwrite("player_id", &dm::engine::game_command::DeclareReactionCommand::player_id);

    py::class_<dm::engine::game_command::StatCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::StatCommand>>(m, "StatCommand")
        .def(py::init<dm::engine::game_command::StatCommand::StatType, int>())
        .def_readwrite("stat", &dm::engine::game_command::StatCommand::stat)
        .def_readwrite("amount", &dm::engine::game_command::StatCommand::amount);

    py::enum_<dm::engine::game_command::StatCommand::StatType>(m, "StatType")
        .value("CARDS_DRAWN", dm::engine::game_command::StatCommand::StatType::CARDS_DRAWN)
        .value("CARDS_DISCARDED", dm::engine::game_command::StatCommand::StatType::CARDS_DISCARDED)
        .value("CREATURES_PLAYED", dm::engine::game_command::StatCommand::StatType::CREATURES_PLAYED)
        .value("SPELLS_CAST", dm::engine::game_command::StatCommand::StatType::SPELLS_CAST)
        .export_values();

    py::class_<dm::engine::game_command::GameResultCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::GameResultCommand>>(m, "GameResultCommand")
        .def(py::init<GameResult>())
        .def_readwrite("result", &dm::engine::game_command::GameResultCommand::result);

    // Phase 8: TokenConverter
    py::class_<dm::ai::encoders::TokenConverter>(m, "TokenConverter")
        .def_static("encode_state", &dm::ai::encoders::TokenConverter::encode_state,
            py::arg("state"), py::arg("perspective") = 0, py::arg("max_len") = 0)
        .def_static("get_vocab_size", &dm::ai::encoders::TokenConverter::get_vocab_size);

    py::class_<dm::ai::TensorConverter>(m, "TensorConverter")
        .def_readonly_static("INPUT_SIZE", &dm::ai::TensorConverter::INPUT_SIZE)
        .def_static("convert_to_tensor", &dm::ai::TensorConverter::convert_to_tensor,
             py::arg("game_state"), py::arg("player_view"), py::arg("card_db"), py::arg("mask_opponent_hand") = true)
        .def_static("convert_batch_flat", static_cast<std::vector<float> (*)(const std::vector<std::shared_ptr<dm::core::GameState>>&, const std::map<dm::core::CardID, dm::core::CardDefinition>&, bool)>(&dm::ai::TensorConverter::convert_batch_flat),
             py::arg("states"), py::arg("card_db"), py::arg("mask_opponent_hand") = true);

    py::class_<ActionEncoder>(m, "ActionEncoder")
        .def_readonly_static("TOTAL_ACTION_SIZE", &ActionEncoder::TOTAL_ACTION_SIZE)
        .def_static("action_to_index", &ActionEncoder::action_to_index);

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

    py::enum_<ActionType>(m, "ActionType")
        .value("PASS", ActionType::PASS)
        .value("PLAY_CARD", ActionType::PLAY_CARD)
        .value("MANA_CHARGE", ActionType::MANA_CHARGE)
        .value("ATTACK_CREATURE", ActionType::ATTACK_CREATURE)
        .value("ATTACK_PLAYER", ActionType::ATTACK_PLAYER)
        .value("BLOCK", ActionType::BLOCK)
        .value("USE_SHIELD_TRIGGER", ActionType::USE_SHIELD_TRIGGER)
        .value("RESOLVE_EFFECT", ActionType::RESOLVE_EFFECT)
        .value("SELECT_TARGET", ActionType::SELECT_TARGET)
        .value("USE_ABILITY", ActionType::USE_ABILITY)
        .value("DECLARE_REACTION", ActionType::DECLARE_REACTION)
        .value("PLAY_CARD_INTERNAL", ActionType::PLAY_CARD_INTERNAL)
        .value("RESOLVE_BATTLE", ActionType::RESOLVE_BATTLE)
        .value("BREAK_SHIELD", ActionType::BREAK_SHIELD)
        .export_values();

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
        .export_values();

    py::enum_<TriggerType>(m, "TriggerType")
        .value("ON_PLAY", TriggerType::ON_PLAY)
        .value("ON_ATTACK", TriggerType::ON_ATTACK)
        .value("ON_DESTROY", TriggerType::ON_DESTROY)
        .value("S_TRIGGER", TriggerType::S_TRIGGER)
        .value("TURN_START", TriggerType::TURN_START)
        .value("PASSIVE_CONST", TriggerType::PASSIVE_CONST)
        .value("NONE", TriggerType::NONE)
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
        .value("SELECT_OPTION", EffectType::SELECT_OPTION)
        .value("SELECT_NUMBER", EffectType::SELECT_NUMBER)
        .export_values();

    py::enum_<ResolveType>(m, "ResolveType")
        .value("NONE", ResolveType::NONE)
        .value("TARGET_SELECT", ResolveType::TARGET_SELECT)
        .value("EFFECT_RESOLUTION", ResolveType::EFFECT_RESOLUTION)
        .export_values();

    py::enum_<ModifierType>(m, "ModifierType")
        .value("NONE", ModifierType::NONE)
        .value("COST_MODIFIER", ModifierType::COST_MODIFIER)
        .value("POWER_MODIFIER", ModifierType::POWER_MODIFIER)
        .value("GRANT_KEYWORD", ModifierType::GRANT_KEYWORD)
        .value("SET_KEYWORD", ModifierType::SET_KEYWORD)
        .export_values();

    py::enum_<EffectActionType>(m, "EffectActionType")
        .value("DRAW_CARD", EffectActionType::DRAW_CARD)
        .value("ADD_MANA", EffectActionType::ADD_MANA)
        .value("SEARCH_DECK_BOTTOM", EffectActionType::SEARCH_DECK_BOTTOM)
        .value("SEND_TO_DECK_BOTTOM", EffectActionType::SEND_TO_DECK_BOTTOM)
        .value("SEARCH_DECK", EffectActionType::SEARCH_DECK)
        .value("SHUFFLE_DECK", EffectActionType::SHUFFLE_DECK)
        .value("DESTROY", EffectActionType::DESTROY)
        .value("RETURN_TO_HAND", EffectActionType::RETURN_TO_HAND)
        .value("SEND_TO_MANA", EffectActionType::SEND_TO_MANA)
        .value("TAP", EffectActionType::TAP)
        .value("UNTAP", EffectActionType::UNTAP)
        .value("MODIFY_POWER", EffectActionType::MODIFY_POWER)
        .value("BREAK_SHIELD", EffectActionType::BREAK_SHIELD)
        .value("LOOK_AND_ADD", EffectActionType::LOOK_AND_ADD)
        .value("SUMMON_TOKEN", EffectActionType::SUMMON_TOKEN)
        .value("MEKRAID", EffectActionType::MEKRAID)
        .value("DISCARD", EffectActionType::DISCARD)
        .value("PLAY_FROM_ZONE", EffectActionType::PLAY_FROM_ZONE)
        .value("COST_REFERENCE", EffectActionType::COST_REFERENCE)
        .value("LOOK_TO_BUFFER", EffectActionType::LOOK_TO_BUFFER)
        .value("SELECT_FROM_BUFFER", EffectActionType::SELECT_FROM_BUFFER)
        .value("PLAY_FROM_BUFFER", EffectActionType::PLAY_FROM_BUFFER)
        .value("MOVE_BUFFER_TO_ZONE", EffectActionType::MOVE_BUFFER_TO_ZONE)
        .value("REVOLUTION_CHANGE", EffectActionType::REVOLUTION_CHANGE)
        .value("COUNT_CARDS", EffectActionType::COUNT_CARDS)
        .value("GET_GAME_STAT", EffectActionType::GET_GAME_STAT)
        .value("APPLY_MODIFIER", EffectActionType::APPLY_MODIFIER)
        .value("REVEAL_CARDS", EffectActionType::REVEAL_CARDS)
        .value("REGISTER_DELAYED_EFFECT", EffectActionType::REGISTER_DELAYED_EFFECT)
        .value("RESET_INSTANCE", EffectActionType::RESET_INSTANCE)
        .value("ADD_SHIELD", EffectActionType::ADD_SHIELD)
        .value("SEND_SHIELD_TO_GRAVE", EffectActionType::SEND_SHIELD_TO_GRAVE)
        .value("MOVE_TO_UNDER_CARD", EffectActionType::MOVE_TO_UNDER_CARD)
        .value("SELECT_NUMBER", EffectActionType::SELECT_NUMBER)
        .value("FRIEND_BURST", EffectActionType::FRIEND_BURST)
        .value("GRANT_KEYWORD", EffectActionType::GRANT_KEYWORD)
        .value("MOVE_CARD", EffectActionType::MOVE_CARD)
        .value("CAST_SPELL", EffectActionType::CAST_SPELL)
        .value("PUT_CREATURE", EffectActionType::PUT_CREATURE)
        .value("SELECT_OPTION", EffectActionType::SELECT_OPTION)
        .value("RESOLVE_BATTLE", EffectActionType::RESOLVE_BATTLE)
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
        .def_property("cip", [](const CardKeywords& k) { return k.cip; }, [](CardKeywords& k, bool v) { k.cip = v; })
        .def_property("at_attack", [](const CardKeywords& k) { return k.at_attack; }, [](CardKeywords& k, bool v) { k.at_attack = v; })
        .def_property("destruction", [](const CardKeywords& k) { return k.destruction; }, [](CardKeywords& k, bool v) { k.destruction = v; })
        .def_property("just_diver", [](const CardKeywords& k) { return k.just_diver; }, [](CardKeywords& k, bool v) { k.just_diver = v; })
        .def_property("hyper_energy", [](const CardKeywords& k) { return k.hyper_energy; }, [](CardKeywords& k, bool v) { k.hyper_energy = v; });

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
        .def_readwrite("input_value_key", &ActionDef::input_value_key)
        .def_readwrite("output_value_key", &ActionDef::output_value_key)
        .def_readwrite("condition", &ActionDef::condition)
        .def_readwrite("options", &ActionDef::options)
        .def_readwrite("scope", &ActionDef::scope);

    py::class_<EffectDef>(m, "EffectDef")
        .def(py::init<>())
        .def_readwrite("trigger", &EffectDef::trigger)
        .def_readwrite("condition", &EffectDef::condition)
        .def_readwrite("actions", &EffectDef::actions);

    py::class_<CardDefinition, std::shared_ptr<CardDefinition>>(m, "CardDefinition")
        .def(py::init([](int id, std::string name, std::string civ_str, std::vector<std::string> races, int cost, int power, CardKeywords keywords, std::vector<EffectDef> effects) {
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
        .def_readwrite("revolution_change_condition", &CardDefinition::revolution_change_condition)
        .def_readwrite("is_key_card", &CardDefinition::is_key_card)
        .def_readwrite("ai_importance_score", &CardDefinition::ai_importance_score)
        .def_readwrite("spell_side", &CardDefinition::spell_side)
        .def_property("civilization",
            [](const CardDefinition& c) { return c.civilizations.empty() ? Civilization::NONE : c.civilizations[0]; },
            [](CardDefinition& c, Civilization civ) { c.civilizations = {civ}; })
        .def_readwrite("civilizations", &CardDefinition::civilizations); // Added property for civilizations vector

    py::class_<CardData>(m, "CardData")
        .def(py::init([](CardID id, std::string name, int cost, std::string civilization, int power, std::string type, std::vector<std::string> races, std::vector<EffectDef> effects) {
            CardData c;
            c.id = id;
            c.name = name;
            c.cost = cost;
            if (civilization == "FIRE") c.civilizations.push_back(Civilization::FIRE);
            else if (civilization == "WATER") c.civilizations.push_back(Civilization::WATER);
            else if (civilization == "NATURE") c.civilizations.push_back(Civilization::NATURE);
            else if (civilization == "LIGHT") c.civilizations.push_back(Civilization::LIGHT);
            else if (civilization == "DARKNESS") c.civilizations.push_back(Civilization::DARKNESS);
            c.power = power;
            c.type = type;
            c.races = races;
            c.effects = effects;
            return c;
        }))
        .def_readwrite("static_abilities", &CardData::static_abilities);

    py::class_<CardInstance>(m, "CardInstance")
        .def(py::init<>())
        .def_readwrite("instance_id", &CardInstance::instance_id)
        .def_readwrite("card_id", &CardInstance::card_id)
        .def_readwrite("owner", &CardInstance::owner)
        .def_readwrite("is_tapped", &CardInstance::is_tapped)
        .def_readwrite("summoning_sickness", &CardInstance::summoning_sickness)
        .def_readwrite("turn_played", &CardInstance::turn_played)
        .def_readwrite("is_face_down", &CardInstance::is_face_down);

    // Instruction and Pipeline
    py::class_<Instruction>(m, "Instruction")
        .def(py::init<>())
        .def(py::init<InstructionOp>())
        .def_readwrite("op", &Instruction::op)
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

    py::class_<dm::engine::systems::PipelineExecutor, std::shared_ptr<dm::engine::systems::PipelineExecutor>>(m, "PipelineExecutor")
        .def(py::init<>())
        .def("set_context_var", &dm::engine::systems::PipelineExecutor::set_context_var)
        .def("execute", static_cast<void (dm::engine::systems::PipelineExecutor::*)(const std::vector<dm::core::Instruction>&, core::GameState&, const std::map<core::CardID, core::CardDefinition>&)>(&dm::engine::systems::PipelineExecutor::execute));

    py::class_<GameState::QueryContext>(m, "QueryContext")
        .def_readwrite("query_id", &GameState::QueryContext::query_id)
        .def_readwrite("query_type", &GameState::QueryContext::query_type)
        .def_readwrite("params", &GameState::QueryContext::params)
        .def_readwrite("valid_targets", &GameState::QueryContext::valid_targets)
        .def_readwrite("options", &GameState::QueryContext::options);

    py::class_<TurnStats>(m, "TurnStats")
         .def_readwrite("cards_drawn_this_turn", &TurnStats::cards_drawn_this_turn);

    py::class_<Player>(m, "Player")
        .def_readwrite("hand", &Player::hand)
        .def_readwrite("mana_zone", &Player::mana_zone)
        .def_readwrite("battle_zone", &Player::battle_zone)
        .def_readwrite("shield_zone", &Player::shield_zone)
        .def_readwrite("graveyard", &Player::graveyard)
        .def_readwrite("deck", &Player::deck)
        .def_readwrite("effect_buffer", &Player::effect_buffer);

    py::class_<GameState>(m, "GameState")
        .def(py::init<int>())
        .def("setup_test_duel", &GameState::setup_test_duel)
        .def("execute_command", &GameState::execute_command)
        .def_readonly("command_history", &GameState::command_history)
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("current_phase", &GameState::current_phase)
        .def_readwrite("players", &GameState::players)
        .def_readwrite("game_over", &GameState::game_over)
        .def_readwrite("winner", &GameState::winner)
        .def_readwrite("turn_stats", &GameState::turn_stats)
        .def_readwrite("waiting_for_user_input", &GameState::waiting_for_user_input)
        .def_readwrite("pending_query", &GameState::pending_query)
        .def_readwrite("status", &GameState::status)
        .def("get_pending_effect_count", [](const GameState& s) { return s.pending_effects.size(); })
        .def("clone", &GameState::clone)
        .def("get_card_instance", [](GameState& s, int id) { return s.get_card_instance(id); }, py::return_value_policy::reference)
        .def("get_zone", &GameState::get_zone)
        .def("set_deck", [](GameState& s, PlayerID pid, std::vector<int> ids) {
             s.players[pid].deck.clear();
             for (int id : ids) s.players[pid].deck.push_back(CardInstance(id, pid));
        })
        .def("add_test_card_to_battle", [](GameState& s, PlayerID pid, int cid, int iid, bool tapped, bool sick) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             c.is_tapped = tapped;
             c.summoning_sickness = sick;
             s.players[pid].battle_zone.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("add_card_to_hand", [](GameState& s, PlayerID pid, int cid, int iid) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             s.players[pid].hand.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("add_card_to_mana", [](GameState& s, PlayerID pid, int cid, int iid) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             s.players[pid].mana_zone.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("add_card_to_deck", [](GameState& s, PlayerID pid, int cid, int iid) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             s.players[pid].deck.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("initialize_card_stats", &GameState::initialize_card_stats)
        .def("calculate_hash", &GameState::calculate_hash)
        .def("get_pending_effects_info", [](const GameState& s) {
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
        });

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("source_instance_id", &Action::source_instance_id)
        .def_readwrite("target_instance_id", &Action::target_instance_id)
        .def_readwrite("slot_index", &Action::slot_index)
        .def("to_string", &Action::to_string);

    // Systems
    py::class_<ActionGenerator>(m, "ActionGenerator")
        .def_static("generate_legal_actions", &ActionGenerator::generate_legal_actions);

    py::class_<dm::engine::EffectSystem, std::unique_ptr<dm::engine::EffectSystem, py::nodelete>>(m, "EffectSystem")
        .def_static("instance", [](){ return &dm::engine::EffectSystem::instance(); }, py::return_value_policy::reference)
        .def_static("compile_action", [](GameState& state, const ActionDef& action, int source_id, std::map<CardID, CardDefinition>& db, py::object py_ctx) {
            std::vector<Instruction> instructions;
            std::map<std::string, int> execution_context;
            // Simplified: Context from python dict not fully supported for input, mostly output
            // But we can try to load it
            if (py::isinstance<py::dict>(py_ctx)) {
                // Ignore for now or implement conversion
            }

            dm::engine::EffectSystem::instance().compile_action(state, action, source_id, execution_context, db, instructions);
            return instructions;
        });

    // Bind GameLogicSystem instead of EffectResolver
    // Also exposing as GenericCardSystem for test compatibility
    auto effect_resolver = py::class_<dm::engine::systems::GameLogicSystem>(m, "EffectResolver"); // Keep name for compatibility
    effect_resolver
        .def_static("resolve_action", [](GameState& state, const Action& action, const std::map<CardID, CardDefinition>& db){
            dm::engine::systems::GameLogicSystem::resolve_action(state, action, db);
        })
        .def_static("resume", [](GameState& state, const std::map<CardID, CardDefinition>& db, py::object input_val) {
             if (!state.active_pipeline) return;

             dm::engine::systems::ContextValue val;
             if (py::isinstance<py::int_>(input_val)) {
                 val = input_val.cast<int>();
             } else if (py::isinstance<py::str>(input_val)) {
                 val = input_val.cast<std::string>();
             } else if (py::isinstance<py::list>(input_val)) {
                 std::vector<int> vec;
                 for (auto item : input_val.cast<py::list>()) {
                     vec.push_back(item.cast<int>());
                 }
                 val = vec;
             }

             auto pipeline = get_active_pipeline(state);
             if (pipeline) {
                 pipeline->resume(state, db, val);
                 if (pipeline->call_stack.empty()) {
                     state.active_pipeline.reset();
                 }
             }
             });

    // Alias GenericCardSystem to EffectResolver or expose EffectSystem wrapper
    // The test calls dm_ai_module.GenericCardSystem.resolve_effect(self.state, ed, -1)
    // We map this to EffectSystem::compile_effect -> PipelineExecutor::execute

    struct GenericCardSystemWrapper {};
    py::class_<GenericCardSystemWrapper>(m, "GenericCardSystem")
        .def_static("resolve_effect", [](GameState& state, const EffectDef& eff, int source_id) {
             auto db = CardRegistry::get_all_definitions();
             dm::engine::EffectSystem::instance().resolve_effect(state, eff, source_id, db);
        })
        .def_static("resolve_effect_with_db", [](GameState& state, const EffectDef& eff, int source_id, const std::map<CardID, CardDefinition>& db) {
             dm::engine::EffectSystem::instance().resolve_effect(state, eff, source_id, db);
        })
        .def_static("resolve_action_with_context", [](GameState& state, int source_id, const ActionDef& action, const std::map<CardID, CardDefinition>& db, std::map<std::string, int> ctx) {
            std::vector<Instruction> instructions;
            dm::engine::EffectSystem::instance().compile_action(state, action, source_id, ctx, db, instructions);
            if (!instructions.empty()) {
                dm::engine::systems::PipelineExecutor pipeline;
                // Load context
                for (const auto& [k, v] : ctx) {
                    pipeline.set_context_var("$" + k, v);
                }

                // Add default source/controller if missing
                 pipeline.set_context_var("$source", source_id);
                 int controller = 0;
                 if ((size_t)source_id < state.card_owner_map.size()) controller = state.card_owner_map[source_id];
                 pipeline.set_context_var("$controller", controller);

                pipeline.execute(instructions, state, db);

                // Extract context back
                for (const auto& [k, v] : pipeline.context) {
                    if (std::holds_alternative<int>(v)) {
                         std::string clean_key = k;
                         if (clean_key.rfind("$", 0) == 0) clean_key = clean_key.substr(1);
                         ctx[clean_key] = std::get<int>(v);
                    }
                }
            }
            return ctx;
        });

    py::class_<JsonLoader>(m, "JsonLoader")
        .def_static("load_cards", &JsonLoader::load_cards);

    // Expose register_card_data
    m.def("register_card_data", [](const CardData& data) {
         // Create a temporary JSON or directly insert into private map?
         // Since CardRegistry members are private and we don't have direct setter,
         // We can serialize to JSON and load it.
         nlohmann::json j;
         dm::core::to_json(j, data);
         std::string json_str = j.dump();
         CardRegistry::load_from_json(json_str);
    });

    py::class_<GameInstance>(m, "GameInstance")
        .def(py::init<uint32_t, const std::map<core::CardID, core::CardDefinition>&>())
        .def_readonly("state", &GameInstance::state)
        .def("reset_with_scenario", &GameInstance::reset_with_scenario);

    py::class_<CardRegistry>(m, "CardRegistry")
        .def_static("register_card_data", [](const CardData& data) {
             nlohmann::json j;
             dm::core::to_json(j, data);
             std::string json_str = j.dump();
             CardRegistry::load_from_json(json_str);
        });

    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("setup_scenario", &PhaseManager::setup_scenario)
        .def_static("next_phase", &PhaseManager::next_phase)
        .def_static("check_game_over", &PhaseManager::check_game_over);

    // AI Components
    // Register shared_ptr holder for GameState if not already implicit (usually good practice for pybind11)
    // But py::class_<GameState> is already defined.
    // If we want to return vector<shared_ptr<GameState>>, we rely on pybind11 smart pointer handling.

    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def_readonly("visit_count", &MCTSNode::visit_count)
        .def_readonly("value", &MCTSNode::value_sum)
        .def_readonly("action", &MCTSNode::action_from_parent)
        .def_property_readonly("children", [](const MCTSNode& node) {
            std::vector<MCTSNode*> children_ptrs;
            for (const auto& child : node.children) {
                children_ptrs.push_back(child.get());
            }
            return children_ptrs;
        }, py::return_value_policy::reference);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::map<CardID, CardDefinition>&, float, float, float, int, float>())
        .def("search", &MCTS::search)
        .def("get_last_root", &MCTS::get_last_root);

    py::class_<HeuristicEvaluator>(m, "HeuristicEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("evaluate", &HeuristicEvaluator::evaluate);

    py::class_<Determinizer>(m, "Determinizer")
        .def_static("determinize", &Determinizer::determinize);

    py::class_<LethalSolver>(m, "LethalSolver")
        .def_static("is_lethal", &LethalSolver::is_lethal);

    // Phase 2: Deck Inference
    py::class_<dm::ai::inference::DeckInference>(m, "DeckInference")
        .def(py::init<>())
        .def("load_decks", &dm::ai::inference::DeckInference::load_decks)
        .def("infer_probabilities", &dm::ai::inference::DeckInference::infer_probabilities)
        .def("sample_hidden_cards", &dm::ai::inference::DeckInference::sample_hidden_cards);

    py::class_<GameResultInfo>(m, "GameResultInfo")
        .def_readwrite("result", &GameResultInfo::result)
        .def_readwrite("turn_count", &GameResultInfo::turn_count)
        .def_readwrite("states", &GameResultInfo::states)
        .def_readwrite("policies", &GameResultInfo::policies)
        .def_readwrite("active_players", &GameResultInfo::active_players);

    py::class_<ParallelRunner>(m, "ParallelRunner")
        .def(py::init<const std::map<CardID, CardDefinition>&, int, int>())
        .def("play_games", &ParallelRunner::play_games, py::return_value_policy::move) // Use move policy
        .def("play_scenario_match", &ParallelRunner::play_scenario_match)
        .def("play_deck_matchup", &ParallelRunner::play_deck_matchup);

    // Scenarios
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
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("run_scenario", &ScenarioExecutor::run_scenario);

    py::class_<ParametricBelief>(m, "ParametricBelief")
        .def(py::init<>())
        .def("set_weights", &ParametricBelief::set_weights)
        .def("initialize", &ParametricBelief::initialize)
        .def("update", &ParametricBelief::update)
        .def("get_vector", &ParametricBelief::get_vector);
}
