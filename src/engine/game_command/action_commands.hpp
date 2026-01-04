#pragma once
#include "game_command.hpp"
#include "core/types.hpp"
#include "core/action.hpp" // For conversion if needed, or purely independent

namespace dm::engine::game_command {

    // High-level commands correspond to Player Intent

    class PlayCardCommand : public GameCommand {
    public:
        int card_instance_id;
        int target_slot_index = -1; // For some UI/AI hints
        bool is_spell_side = false; // For Twinpact
        // Additional context for "Internal" plays
        core::SpawnSource spawn_source = core::SpawnSource::HAND_SUMMON;

        PlayCardCommand(int card_id) : card_instance_id(card_id) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override; // No-op (rely on sub-commands)
        CommandType get_type() const override { return CommandType::PLAY_CARD; }
    };

    class AttackCommand : public GameCommand {
    public:
        int source_id;
        int target_id; // -1 for player
        core::PlayerID target_player_id; // For player attack

        AttackCommand(int src, int tgt, core::PlayerID pid = 0)
            : source_id(src), target_id(tgt), target_player_id(pid) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::ATTACK; }
    };

    class BlockCommand : public GameCommand {
    public:
        int blocker_id;
        int attacker_id; // Usually implicit in state, but good to store

        BlockCommand(int blk, int atk = -1) : blocker_id(blk), attacker_id(atk) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::BLOCK; }
    };

    class UseAbilityCommand : public GameCommand {
    public:
        int source_id;
        int target_id; // Optional target (e.g. for Revolution Change return)

        UseAbilityCommand(int src, int tgt = -1) : source_id(src), target_id(tgt) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::USE_ABILITY; }
    };

    class ManaChargeCommand : public GameCommand {
    public:
        int card_id;

        ManaChargeCommand(int cid) : card_id(cid) {}

        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override;
        CommandType get_type() const override { return CommandType::MANA_CHARGE; }
    };

    class PassCommand : public GameCommand {
    public:
        PassCommand() = default;
        void execute(core::GameState& state) override;
        void invert(core::GameState& state) override; // Phase change undo
        CommandType get_type() const override { return CommandType::PASS_TURN; }
    };

}
