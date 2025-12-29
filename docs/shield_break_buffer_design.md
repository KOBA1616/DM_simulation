設計: シールドブレイク（バッファ→一括移動方式）

目的
- 指定枚数のシールドを同時に“バッファ”へ移動し、各カードについてShield Trigger / Guard Strike / Strike Back の宣言を受け付ける。
- 宣言の集約後、一括で手札へ加え、宣言でプレイを選択したカードは自動プレイ（クリーチャーは召喚、呪文は唱える）まで行う。
- クリーチャーの "出た時" 効果は、その召喚処理完了の後に解決される。

主要データ構造
- Player::effect_buffer (既存) を使用
  - buffer item 構造:
    - origin_instance_id: int (元の shield インスタンス id)
    - card_id: CardID
    - owner: PlayerID
    - declared_flags: bitfield {s_trigger, guard_strike, strike_back}
    - declaration_choice: enum {UNDECIDED, YES, NO}

命令 / イベント
- BREAK_SHIELD (既存) -> 変更: shield_id[] を含む BLOCK_BREAK 命令に置換可能
- CHECK_S_TRIGGER (既存) -> 拡張: buffer_index を受け取り、宣言キー `$sbuf_<pid>_<index>` を作る
- APPLY_BUFFER_MOVE (新規) -> バッファ中の全カードを手札へ移す（atomic 表示）
- AUTO_PLAY (新規) -> 宣言 YES のカードに対して PLAY 命令を自動生成

処理シーケンス（簡潔）
1. BreakShieldHandler が target_shield_ids を決定し、1 命令 BLOCK_BREAK(shield_ids) を生成
2. GameLogicSystem::handle_break_shield:
   a. BEFORE_BREAK_SHIELD トリガーを通常どおり compile
   b. shield_ids をネイティブから削除し、対応する CardInstance を Player::effect_buffer に push
      - この段階でまだ手札には加えない
   c. 各 buffer item に対して、もし該当カードが shield_trigger なら CHECK_S_TRIGGER(buffer_index) 命令を生成
      - 同様に GuardStrike/StrikeBack 用チェック命令を生成
   d. すべてのチェック命令を exec.call_stack に積み、宣言ウィンドウを開く（reaction_stack / waiting_for_key 経由）
   e. 全宣言が確定したら APPLY_BUFFER_MOVE を実行:
      - buffer 内のすべてのカードを native の手札に追加（順序は仕様で決める）
      - 同時に UI へは一括で更新を通知（atomic 見え方）
   f. 宣言で PLAY を選択したカードについて AUTO_PLAY 命令を生成してスタックへ積む
3. AUTO_PLAY の解決によりクリーチャーが場に出た場合、該当する "出た時" トリガーは通常のトリガー検出で追加され、それらは AUTO_PLAY 解決後のスタックに積まれる

宣言順・競合ルール
- 同時宣言時の優先度: 非アクティブプレイヤー→アクティブプレイヤー（既存の reaction_stack の扱いに合わせる）
- 同一プレイヤーが複数枚 S-Trigger を宣言する場合は、プレイヤーが任意の順序を選べる UI を用意し、その順序で AUTO_PLAY を積む

置換効果の処理位置（優先順）
1. BEFORE_BREAK_SHIELD トリガー（既存）
2. Replacement Effects that affect "move" event (例: シールド→墓地 に置換) — バッファ格納前に評価して適用
3. 宣言受付（CHECK_S_TRIGGER）
4. APPLY_BUFFER_MOVE（実際の手札追加）

エッジケース
- 宣言未回答（タイムアウト）: デフォルト NO にする（テストで明示可能）
- 置換効果が宣言後に起動した場合: 置換は move 前に評価されるため、宣言は置換の結果に基づく（設計で明確化）
- ループ防止: 置換が再び BREAK_SHIELD を発生させる等の再帰は depth limit を設ける

テストケース（必須）
- 単一盾: S-Trigger なし/あり
- 複数盾: 混在する S-Trigger, GuardStrike, StrikeBack
- GuardStrike が宣言された場合の相互作用（宣言者/対象の効果）
- 置換効果（シールドを墓地に直接送る）との組合せ
- クリーチャーの出た時効果が AUTO_PLAY 解決後に発火すること

実装ファイル一覧（変更候補）
- src/engine/systems/card/handlers/break_shield_handler.hpp
- src/engine/systems/game_logic_system.cpp
- src/engine/systems/pipeline_executor.cpp/hpp (CHECK_S_TRIGGER, APPLY_BUFFER_MOVE, AUTO_PLAY のハンドラ追加)
- src/core/game_state.hpp (buffer item 構造のコメント/補強)
- tests/integration/* (テスト追加)

マイグレーション・互換性
- 既存の単体 BREAK_SHIELD フローは互換性レイヤを残しつつ新フローへ移行可能（旧動作をラップしてバッファ手続きへ変換）

---
作業を続けて「`BreakShieldHandler` の命令生成を BUFFER モードに実装」しますか？（はい／いいえ）