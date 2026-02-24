#include "action_commands.hpp"
#include "commands.hpp" // For TransitionCommand
#include "engine/infrastructure/data/card_registry.hpp"
#include "engine/systems/director/game_logic_system.hpp"
#include <fstream> // For debug logging


namespace dm::engine::game_command {

using namespace dm::engine::systems;

void PlayCardCommand::execute(core::GameState &state) {
  // Construct CommandDef to pass to GameLogicSystem
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  core::CommandDef cmd;
  cmd.type = core::CommandType::PLAY_FROM_ZONE;
  cmd.instance_id = card_instance_id;
  // Use amount to signal spell side (1 = spell side, 0 = creature side)
  cmd.amount = is_spell_side ? 1 : 0;

  // Note: spawn_source is currently handled by inference in dispatch_command
  // via get_card_location, or could be added to CommandDef if explicitly needed
  // in future.

  GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
}

void PlayCardCommand::invert(core::GameState &state) {
  // No-op
  (void)state;
}

void AttackCommand::execute(core::GameState &state) {
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  core::CommandDef cmd;
  if (target_id == -1) {
    cmd.type = core::CommandType::ATTACK_PLAYER;
    // target_instance = -1 implies attacking the opponent player (standard 1v1)
    cmd.target_instance = -1;
  } else {
    cmd.type = core::CommandType::ATTACK_CREATURE;
    cmd.target_instance = target_id;
  }
  cmd.instance_id = source_id;

  GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
}

void AttackCommand::invert(core::GameState &state) { (void)state; }

void BlockCommand::execute(core::GameState &state) {
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  core::CommandDef cmd;
  cmd.type = core::CommandType::BLOCK;
  cmd.instance_id = blocker_id;

  GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
}

void BlockCommand::invert(core::GameState &state) { (void)state; }

void UseAbilityCommand::execute(core::GameState &state) {
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  core::CommandDef cmd;
  cmd.type = core::CommandType::USE_ABILITY;
  cmd.instance_id = source_id;
  cmd.target_instance = target_id;

  GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
}

void UseAbilityCommand::invert(core::GameState &state) { (void)state; }

void ManaChargeCommand::execute(core::GameState &state) {
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();

  core::CommandDef cmd;
  cmd.type = core::CommandType::MANA_CHARGE;
  cmd.instance_id = card_id;

  GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
}

void ManaChargeCommand::invert(core::GameState &state) { (void)state; }

void PassCommand::execute(core::GameState &state) {
  const auto &card_db =
      dm::engine::infrastructure::CardRegistry::get_all_definitions();
  core::CommandDef cmd;
  cmd.type = core::CommandType::PASS;

  GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
}

void PassCommand::invert(core::GameState &state) { (void)state; }

} // namespace dm::engine::game_command
