#include "action_commands.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "commands.hpp" // For TransitionCommand

namespace dm::engine::game_command {

    using namespace dm::engine::systems;

    void PlayCardCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::Action action;
        action.type = core::PlayerIntent::PLAY_CARD;
        action.source_instance_id = card_instance_id;
        action.spawn_source = spawn_source;
        action.is_spell_side = is_spell_side;

        GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void PlayCardCommand::invert(core::GameState& state) { (void)state; }

    void AttackCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::Action action;
        if (target_id == -1) {
            action.type = core::PlayerIntent::ATTACK_PLAYER;
            action.target_player = target_player_id;
        } else {
            action.type = core::PlayerIntent::ATTACK_CREATURE;
            action.target_instance_id = target_id;
        }
        action.source_instance_id = source_id;

        GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void AttackCommand::invert(core::GameState& state) { (void)state; }

    void BlockCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::Action action;
        action.type = core::PlayerIntent::BLOCK;
        action.source_instance_id = blocker_id;

        GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void BlockCommand::invert(core::GameState& state) { (void)state; }

    void UseAbilityCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::Action action;
        action.type = core::PlayerIntent::USE_ABILITY;
        action.source_instance_id = source_id;
        action.target_instance_id = target_id;

        GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void UseAbilityCommand::invert(core::GameState& state) { (void)state; }

    void ManaChargeCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::Action action;
        action.type = core::PlayerIntent::MANA_CHARGE;
        action.source_instance_id = card_id;

        GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void ManaChargeCommand::invert(core::GameState& state) { (void)state; }

    void PassCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();
        core::Action action;
        action.type = core::PlayerIntent::PASS;

        GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void PassCommand::invert(core::GameState& state) { (void)state; }

    void SearchDeckCommand::execute(core::GameState& state) {
        // Create a query for deck search with the provided filter
        // Note: Filter string parsing logic is assumed to be handled by the UI/AI or QueryCommand consumer
        std::map<std::string, int> params;
        params["amount"] = count;
        // params["filter"] = filter_str; // QueryCommand params are map<string, int>, we need extended params or store filter in a separate way.
        // For now, we assume the query type itself or a separate mechanism handles the filter details,
        // or we use a convention where we rely on the consumer to know the context.
        // But QueryCommand supports `params` which is <string, int>.
        // Ideally QueryCommand should support string args.
        // However, looking at QueryCommand definition: `std::map<std::string, int> params;`
        // We can't pass string filter here easily.
        // We will create a QueryCommand with a specific type that implies the filter, or update QueryCommand.
        // But updating QueryCommand is out of scope for this immediate fix.
        // Let's use the `options` vector for string data if possible, or just issue the command.

        std::vector<std::string> options;
        if (!filter_str.empty()) {
            options.push_back(filter_str);
        }

        auto cmd = std::make_unique<QueryCommand>("SELECT_TARGET_IN_DECK", std::vector<int>{}, params, options);
        state.execute_command(std::move(cmd));
    }

    void SearchDeckCommand::invert(core::GameState& state) { (void)state; }

    void ShieldTriggerCommand::execute(core::GameState& state) {
         const auto& card_db = CardRegistry::get_all_definitions();
         core::Action action;
         action.type = core::PlayerIntent::USE_SHIELD_TRIGGER;
         action.source_instance_id = card_id;
         GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void ShieldTriggerCommand::invert(core::GameState& state) { (void)state; }

    void ResolveBattleCommand::execute(core::GameState& state) {
         const auto& card_db = CardRegistry::get_all_definitions();
         core::Action action;
         action.type = core::PlayerIntent::RESOLVE_BATTLE;
         action.source_instance_id = attacker_id;
         action.target_instance_id = defender_id;
         GameLogicSystem::resolve_action_oneshot(state, action, card_db);
    }

    void ResolveBattleCommand::invert(core::GameState& state) { (void)state; }

}
