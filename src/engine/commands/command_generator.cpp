#include "src/engine/commands/command_generator.hpp"
#include "src/engine/actions/intent_generator.hpp"

#include <atomic>
#include <sstream>

namespace engine {
namespace commands {

static std::atomic<uint64_t> g_cmd_counter{0};

static std::string intent_to_type_name(const dm::core::PlayerIntent intent) {
    using PI = dm::core::PlayerIntent;
    switch (intent) {
        case PI::PASS: return "PASS_TURN";
        case PI::MANA_CHARGE: return "MANA_CHARGE";
        case PI::PLAY_CARD: return "PLAY_CARD";
        case PI::PLAY_FROM_ZONE: return "PLAY_FROM_ZONE";
        case PI::ATTACK_PLAYER: return "ATTACK";
        case PI::ATTACK_CREATURE: return "ATTACK";
        case PI::BLOCK: return "BLOCK";
        case PI::SELECT_TARGET: return "SELECT_TARGET";
        case PI::SELECT_OPTION: return "SELECT_OPTION";
        case PI::SELECT_NUMBER: return "SELECT_NUMBER";
        default: return "UNKNOWN";
    }
}

std::vector<CommandDef> CommandGenerator::generate_commands(
    const dm::core::GameState& game_state,
    const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
) {
    std::vector<CommandDef> out;

    // Use existing IntentGenerator as a bridge for Phase 1.
    auto actions = dm::engine::IntentGenerator::generate_legal_actions(game_state, card_db);
    out.reserve(actions.size());

    for (const auto &a : actions) {
        CommandDef cmd;
        cmd.type = intent_to_type_name(a.type);
        uint64_t id = ++g_cmd_counter;
        cmd.uid = "cmd-" + std::to_string(id);

        if (a.card_id != 0) cmd.instance_id = static_cast<int>(a.card_id);
        if (a.source_instance_id >= 0) cmd.source_instance_id = a.source_instance_id;
        if (a.target_instance_id >= 0) cmd.target_instance_id = a.target_instance_id;
        cmd.owner_id = static_cast<int>(game_state.active_player_id);
        if (a.slot_index >= 0) cmd.str_param = "slot:" + std::to_string(a.slot_index);

        out.push_back(std::move(cmd));
    }

    return out;
}

} // namespace commands
} // namespace engine

