#include "phase_strategies.hpp"
#include "engine/systems/effects/passive_effect_system.hpp"
#include "engine/systems/mechanics/cost_payment_system.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include "engine/utils/target_utils.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>


namespace dm::engine {

using namespace dm::core;

std::vector<CommandDef>
ManaPhaseStrategy::generate(const CommandGenContext &ctx) {
  // IMPORTANT: Mana charge is restricted to once per turn per player.
  // This strategy checks the 'mana_charged_by_player' flag in TurnStats.
  // DO NOT generate MANA_CHARGE commands if the flag is already true.

  std::vector<CommandDef> actions;
  const auto &game_state = ctx.game_state;
  const Player &active_player = game_state.players[game_state.active_player_id];

  try {
    std::filesystem::create_directories("logs");
    std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
    if (ofs) {
      ofs << "[ManaPhaseStrategy] mana_charged_this_turn="
          << (game_state.turn_stats
                      .mana_charged_by_player[game_state.active_player_id]
                  ? "TRUE"
                  : "FALSE")
          << " turn=" << game_state.turn_number
          << " hand_size=" << active_player.hand.size()
          << " active_pid=" << (int)game_state.active_player_id << "\n";
    }
  } catch (...) {
  }

  if (!game_state.turn_stats
           .mana_charged_by_player[game_state.active_player_id]) {
    for (size_t i = 0; i < active_player.hand.size(); ++i) {
      const auto &card = active_player.hand[i];
      CommandDef cmd;
      cmd.type = CommandType::MANA_CHARGE;
      cmd.instance_id = card.instance_id;
      cmd.slot_index = static_cast<int>(i);
      actions.push_back(cmd);
    }

    try {
      std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
      if (ofs) {
        ofs << "[ManaPhaseStrategy] Generated " << actions.size()
            << " MANA_CHARGE actions\n";
      }
    } catch (...) {
    }

    CommandDef pass;
    pass.type = CommandType::PASS;
    actions.push_back(pass);
  } else {
    try {
      std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
      if (ofs) {
        ofs << "[ManaPhaseStrategy] Already charged - returning PASS only\n";
      }
    } catch (...) {
    }

    CommandDef pass;
    pass.type = CommandType::PASS;
    actions.push_back(pass);
  }

  return actions;
}

std::vector<CommandDef>
MainPhaseStrategy::generate(const CommandGenContext &ctx) {
  std::vector<CommandDef> actions;
  const auto &game_state = ctx.game_state;
  const auto &card_db = ctx.card_db;
  const Player &active_player = game_state.players[game_state.active_player_id];

  try {
    std::filesystem::create_directories("logs");
    std::ofstream enter_ofs("logs/main_phase_checks.txt", std::ios::app);
    if (enter_ofs) {
      enter_ofs << "[MainPhaseEnter] player="
                << static_cast<int>(game_state.active_player_id)
                << " turn=" << game_state.turn_number
                << " hand_count=" << active_player.hand.size() << "\n";
    }
  } catch (...) {
  }

  auto log_skip = [&](int hand_index, const CardInstance &card,
                      const CardDefinition &def, bool spell_restricted,
                      int adjusted_cost, int available_mana, bool can_pay) {
    try {
      std::filesystem::create_directories("logs");
      std::ofstream ofs("logs/main_phase_checks.txt", std::ios::app);
      if (!ofs)
        return;
      std::ostringstream ss;
      ss << "[MainPhase] player="
         << static_cast<int>(game_state.active_player_id)
         << " turn=" << game_state.turn_number << " hand_idx=" << hand_index
         << " instance_id=" << card.instance_id << " card_id=" << def.id
         << " adjusted_cost=" << adjusted_cost
         << " available_mana=" << available_mana
         << " spell_restricted=" << (spell_restricted ? 1 : 0)
         << " can_pay=" << (can_pay ? 1 : 0) << "\n";
      ofs << ss.str();
    } catch (...) {
    }
  };

  for (size_t i = 0; i < active_player.hand.size(); ++i) {
    const auto &card = active_player.hand[i];
    if (card_db.count(card.card_id)) {
      const auto &def = card_db.at(card.card_id);

      bool spell_restricted = false;
      if (def.type == CardType::SPELL) {
        if (PassiveEffectSystem::instance().check_restriction(
                game_state, card, PassiveType::CANNOT_USE_SPELLS, card_db)) {
          spell_restricted = true;
        }
        if (PassiveEffectSystem::instance().check_restriction(
                game_state, card, PassiveType::LOCK_SPELL_BY_COST, card_db)) {
          spell_restricted = true;
        }
      }

      // 1. Standard Play
      {
        int adjusted_cost =
            ManaSystem::get_adjusted_cost(game_state, active_player, def);
        int available_mana = ManaSystem::get_usable_mana_count(
            game_state, active_player.id, def.civilizations, card_db);
        bool can_pay =
            ManaSystem::can_pay_cost(game_state, active_player, def, card_db);

        if (!spell_restricted && can_pay) {
          CommandDef cmd;
          cmd.type = CommandType::PLAY_FROM_ZONE;
          cmd.instance_id = card.instance_id;
          cmd.amount = 0;
          cmd.slot_index = static_cast<int>(i);
          actions.push_back(cmd);
        } else {
          log_skip(static_cast<int>(i), card, def, spell_restricted,
                   adjusted_cost, available_mana, can_pay);
        }
      }

      // 2. Twinpact Spell Side Play
      if (def.spell_side) {
        const auto &spell_def = *def.spell_side;
        bool side_restricted = false;
        if (PassiveEffectSystem::instance().check_restriction(
                game_state, card, PassiveType::CANNOT_USE_SPELLS, card_db)) {
          side_restricted = true;
        }
        if (PassiveEffectSystem::instance().check_restriction(
                game_state, card, PassiveType::LOCK_SPELL_BY_COST, card_db)) {
          side_restricted = true;
        }

        {
          int adjusted_cost_side = ManaSystem::get_adjusted_cost(
              game_state, active_player, spell_def);
          int available_mana_side = ManaSystem::get_usable_mana_count(
              game_state, active_player.id, spell_def.civilizations, card_db);
          bool can_pay_side = ManaSystem::can_pay_cost(
              game_state, active_player, spell_def, card_db);
          if (!side_restricted && can_pay_side) {
            CommandDef cmd;
            cmd.type = CommandType::PLAY_FROM_ZONE;
            cmd.instance_id = card.instance_id;
            cmd.amount = 1;
            cmd.slot_index = static_cast<int>(i);
            actions.push_back(cmd);
          } else {
            log_skip(static_cast<int>(i), card, spell_def, side_restricted,
                     adjusted_cost_side, available_mana_side, can_pay_side);
          }
        }
      }

      // 3. Active Cost Reductions
      for (const auto &reduction : def.cost_reductions) {
        if (reduction.type == ReductionType::ACTIVE_PAYMENT) {
          int max_units = CostPaymentSystem::calculate_max_units(
              game_state, active_player.id, reduction, card_db);

          for (int units = 1; units <= max_units; ++units) {
            if (reduction.max_units != -1 && units > reduction.max_units)
              break;

            int reduction_val = units * reduction.reduction_amount;
            int adjusted_cost =
                ManaSystem::get_adjusted_cost(game_state, active_player, def);
            int effective_cost = std::max(reduction.min_mana_cost,
                                          adjusted_cost - reduction_val);

            int available_mana = ManaSystem::get_usable_mana_count(
                game_state, active_player.id, def.civilizations, card_db);

            if (available_mana >= effective_cost) {
              CommandDef cmd;
              cmd.type = CommandType::PLAY_FROM_ZONE;
              cmd.instance_id = card.instance_id;
              cmd.target_instance = units;
              cmd.str_param = "ACTIVE_PAYMENT";
              cmd.slot_index = static_cast<int>(i);
              cmd.target_slot_index =
                  units; // Mirror units to target_slot_index just in case
              actions.push_back(cmd);
            }
          }
        }
      }
    }
  }

  CommandDef pass_action;
  pass_action.type = CommandType::PASS;
  actions.push_back(pass_action);

  return actions;
}

std::vector<CommandDef>
AttackPhaseStrategy::generate(const CommandGenContext &ctx) {
  std::vector<CommandDef> actions;
  const auto &game_state = ctx.game_state;
  const auto &card_db = ctx.card_db;
  const Player &active_player = game_state.players[game_state.active_player_id];
  const Player &opponent = game_state.players[1 - game_state.active_player_id];

  try {
    char buf2[128];
    int nn = snprintf(buf2, sizeof(buf2), "[DBG PassiveCount] passive_cnt=%zu\n", game_state.passive_effects.size());
    if (nn > 0) fwrite(buf2, 1, (size_t)std::min(nn, (int)sizeof(buf2)), stderr);
  } catch(...) {}

  for (size_t i = 0; i < active_player.battle_zone.size(); ++i) {
    const auto &card = active_player.battle_zone[i];

    if (card_db.count(card.card_id)) {
      const auto &def = card_db.at(card.card_id);

      bool can_attack_player =
          dm::engine::utils::TargetUtils::can_attack_player(
              card, def, game_state, card_db);
      bool can_attack_creature =
          dm::engine::utils::TargetUtils::can_attack_creature(
              card, def, game_state, card_db);

      bool passive_restricted = false;
      if (PassiveEffectSystem::instance().check_restriction(
              game_state, card, PassiveType::CANNOT_ATTACK, card_db)) {
        passive_restricted = true;
      }

      if (can_attack_player && !passive_restricted) {
        CommandDef attack_player;
        attack_player.type = CommandType::ATTACK_PLAYER;
        attack_player.instance_id = card.instance_id;
        attack_player.target_instance = -1;
        attack_player.slot_index = static_cast<int>(i);
        actions.push_back(attack_player);
      }

      try {
        std::filesystem::create_directories("logs");
        std::ofstream ofs("logs/attack_phase_debug.txt", std::ios::app);
        if (ofs) {
          ofs << "[AttackPhase] attacker=" << card.instance_id
              << " can_attack_player=" << (can_attack_player?1:0)
              << " can_attack_creature=" << (can_attack_creature?1:0)
              << " passive_restricted=" << (passive_restricted?1:0)
              << "\n";
        }
      } catch(...) {}

      if (can_attack_creature && !passive_restricted) {
        for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
          const auto &opp_card = opponent.battle_zone[j];

          // Allow attacks against tapped creatures as normal, or allow against
          // untapped creatures if a passive effect grants that permission
          bool target_allowed_by_passive = PassiveEffectSystem::instance().allows_attack_untapped(game_state, card, card_db);

          // Debug log for decision
          try {
            std::filesystem::create_directories("logs");
            std::ofstream ofs("logs/attack_phase_debug.txt", std::ios::app);
            if (ofs) {
              ofs << "[AttackPhase] attacker=" << card.instance_id
                  << " opp=" << opp_card.instance_id
                  << " opp_tapped=" << (opp_card.is_tapped?1:0)
                  << " target_allowed_by_passive=" << (target_allowed_by_passive?1:0)
                  << " can_attack_creature=" << (can_attack_creature?1:0)
                  << " passive_restricted=" << (passive_restricted?1:0)
                  << "\n";
            }
          } catch(...) {}

          // Delegate legality to RestrictionSystem for accurate checks
          if (card_db.count(opp_card.card_id)) {
            const auto &opp_def = card_db.at(opp_card.card_id);

            // Quick passive scan: allow attack if there exists a passive
            // ALLOW_ATTACK_UNTAPPED that explicitly lists this attacker.
            bool allowed_via_passive = false;
            for (const auto &eff : game_state.passive_effects) {
              if (eff.type == dm::core::PassiveType::ALLOW_ATTACK_UNTAPPED) {
                if (eff.specific_targets.has_value()) {
                  for (int id : *eff.specific_targets) {
                    if (id == card.instance_id) {
                      allowed_via_passive = true;
                      break;
                    }
                  }
                }
              }
              if (allowed_via_passive) break;
            }

            try {
              char buf3[128];
              int nn2 = snprintf(buf3, sizeof(buf3), "[DBG PassiveScan] attacker=%d allowed_via_passive=%d\n", card.instance_id, (int)allowed_via_passive);
              if (nn2 > 0) fwrite(buf3, 1, (size_t)std::min(nn2, (int)sizeof(buf3)), stderr);
            } catch(...) {}

            bool protected_by_jd =
                dm::engine::utils::TargetUtils::is_protected_by_just_diver(
                    opp_card, opp_def, game_state, active_player.id);
            if (game_state.turn_number > opp_card.turn_played)
              protected_by_jd = false;
            if (protected_by_jd)
              continue;

            // If target is tapped it's allowed; if untapped, allow only when
            // a passive explicitly permits this attacker. Prefer the
            // PassiveEffectSystem check (`target_allowed_by_passive`) which
            // centralizes passive logic; fallback to allowed_via_passive scan
            // only if needed.
            if (opp_card.is_tapped || target_allowed_by_passive || allowed_via_passive) {
              // Emit a short runtime debug line to stderr so pytest capture shows
              // which attacker/target pairs are being considered and added.
              try {
                char buf[256];
                int n = snprintf(buf, sizeof(buf), "[DBG AttackPhase] push ATTACK_CREATURE atk=%d tgt=%d allowed=%d\n",
                                 card.instance_id, opp_card.instance_id, (int)allowed_via_passive);
                if (n > 0) {
                  fwrite(buf, 1, (size_t)std::min(n, (int)sizeof(buf)), stderr);
                }
              } catch(...) {}

              CommandDef attack_creature;
              attack_creature.type = CommandType::ATTACK_CREATURE;
              attack_creature.instance_id = card.instance_id;
              attack_creature.target_instance = opp_card.instance_id;
              attack_creature.slot_index = static_cast<int>(i);
              attack_creature.target_slot_index = static_cast<int>(j);
              actions.push_back(attack_creature);
            }
          }
        }
      }
    }
  }

  CommandDef pass;
  pass.type = CommandType::PASS;
  actions.push_back(pass);

  return actions;
}

std::vector<CommandDef>
BlockPhaseStrategy::generate(const CommandGenContext &ctx) {
  std::vector<CommandDef> actions;
  const auto &game_state = ctx.game_state;
  const auto &card_db = ctx.card_db;

  const Player &defender = game_state.players[1 - game_state.active_player_id];

  for (size_t i = 0; i < defender.battle_zone.size(); ++i) {
    const auto &card = defender.battle_zone[i];
    if (!card.is_tapped) {
      if (card_db.count(card.card_id)) {
        const auto &def = card_db.at(card.card_id);
        if (dm::engine::utils::TargetUtils::has_keyword_simple(
                game_state, card, def, "BLOCKER")) {
          if (!PassiveEffectSystem::instance().check_restriction(
                  game_state, card, PassiveType::CANNOT_BLOCK, card_db)) {
            CommandDef block;
            block.type = CommandType::BLOCK;
            block.instance_id = card.instance_id;
            block.slot_index = static_cast<int>(i);
            actions.push_back(block);
          }
        }
      }
    }
  }
  CommandDef pass;
  pass.type = CommandType::PASS;
  actions.push_back(pass);

  return actions;
}

} // namespace dm::engine
