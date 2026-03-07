#pragma once
#include "core/game_state.hpp"
#include <memory>

namespace dm::engine::game_command {

    enum class CommandType {
        TRANSITION,
        MUTATE,
        ATTACH,
        FLOW,
        QUERY,
        DECIDE,
        DECLARE_REACTION,
        STAT,
        GAME_RESULT,
        ADD_CARD,
        SHUFFLE,
        // High-level Action Commands
        PLAY_CARD,
        ATTACK,
        BLOCK,
        USE_ABILITY,
        MANA_CHARGE,
        RESOLVE_PENDING_EFFECT,
        PASS_TURN // or just PASS
    };

    class GameCommand {
    public:
        virtual ~GameCommand() = default;

        // Execute the command, modifying the game state
        virtual void execute(core::GameState& state) = 0;

        // Invert the command to support rollback/undo
        // This should reverse the changes made by execute()
        virtual void invert(core::GameState& state) = 0;

        virtual CommandType get_type() const = 0;

        // 再発防止: コマンド履歴から主要カードのインスタンスIDを取得する。
        //   token_converter の append_command_history で AI 状態表現を細粒度化するために使用。
        //   デフォルトは -1 (該当なし)。サブクラスでオーバーライドする。
        virtual int get_subject_id() const { return -1; }
    };

}
