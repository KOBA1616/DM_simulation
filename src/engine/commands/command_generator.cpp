#include "engine/commands/command_generator.hpp"
#include "engine/actions/intent_generator.hpp"
#include "core/action.hpp"
#include "core/card_json_types.hpp"

using namespace dm::engine;

static dm::core::CommandType map_intent_to_command(dm::core::PlayerIntent intent) {
    using PI = dm::core::PlayerIntent;
    switch (intent) {
        case PI::PASS: return dm::core::CommandType::FLOW; // represent as phase/flow pass
        case PI::MANA_CHARGE: return dm::core::CommandType::BOOST_MANA;
        case PI::PLAY_CARD:
        case PI::DECLARE_PLAY: return dm::core::CommandType::PLAY_FROM_ZONE;
        case PI::ATTACK_PLAYER: return dm::core::CommandType::ATTACK_PLAYER;
        case PI::ATTACK_CREATURE: return dm::core::CommandType::ATTACK_CREATURE;
        case PI::BLOCK: return dm::core::CommandType::BLOCK;
        case PI::SELECT_TARGET: return dm::core::CommandType::QUERY;
        case PI::SELECT_OPTION: return dm::core::CommandType::CHOICE;
        case PI::SELECT_NUMBER: return dm::core::CommandType::SELECT_NUMBER;
        default: return dm::core::CommandType::NONE;
    }
}

std::vector<dm::core::CommandDef> CommandGenerator::generate_legal_commands(
    const dm::core::GameState& game_state,
    const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
) {
    std::vector<dm::core::CommandDef> out;

    // Call existing IntentGenerator action generator as a temporary bridge.
    std::vector<dm::core::Action> actions = IntentGenerator::generate_legal_actions(game_state, card_db);
    out.reserve(actions.size());

    for (const auto& a : actions) {
        dm::core::CommandDef cmd;
        cmd.type = map_intent_to_command(a.type);
        // Map common fields
        if (a.card_id != 0) cmd.instance_id = static_cast<int>(a.card_id);
        if (a.source_instance_id >= 0) cmd.instance_id = a.source_instance_id;
        if (a.target_instance_id >= 0) cmd.target_instance = a.target_instance_id;
        cmd.owner_id = 0; // unknown at generation time for generic intents
        // Minimal string parameters for logging/debug
        if (a.slot_index >= 0) cmd.str_param = "slot:" + std::to_string(a.slot_index);

        out.push_back(std::move(cmd));
    }

    return out;
}
