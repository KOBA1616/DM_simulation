# Design: Native index -> command inverse (index_to_command)

目的
- C++ MCTS がネットワークの出力（action index）を高速に command / ICommand に復元できるようにする。
- Python 側の互換性を保ちつつ、レイテンシとメモリを削減する。

要件
- 決定論的で再現可能: 同一 state と index に対し常に同じ command を返す。
- 低レイテンシ: 単一マッピングは O(1) で解けるか、及びインデックスから直接参照できること。
- 低メモリ: 大きなテーブルよりも領域効率の良い表現を優先。
- デバッグ容易性: 失敗時に Python レイヤーでデコード可能な情報を付与。

基本設計
1. インデックス空間の分割
   - 事前定義したブロックサイズを用いて、各アクションカテゴリの範囲を確保する。例:
     - PASS: 1 slot
     - MANA_CHARGE: M slots (手札上限)
     - PLAY_CARD: P slots (手札上限)
     - ATTACK_PLAYER: A slots (戦闘領域上限)
     - ATTACK_CREATURE: A*A slots
     - ...
   - Python の `index_to_command` と同じオフセット設計を C++ に移植する。

2. インスタンス参照の解決
   - 多くのインデックスは "slot_index" ベースで表される（例: 手札スロット 0..N）。
   - 実行時に `state` を参照し、slot→instance_id を解決して `Command` 構造体を埋める。
   - 可能であれば、`state` に高速に zone 配列を返す API (`get_zone_slots(player, zone)`) を用意。

3. API 提案（C++）

ヘッダ: include/index_to_command.hpp

struct IndexToCommandResult {
    CommandDef cmd;           // pybind/内部で使用可能な Command 構造体
    bool resolved_instance;   // slot→instance の解決に成功したか
};

// 主要関数
IndexToCommandResult index_to_command(int action_index, const GameState &state, int player_id, const CardDB &card_db);

// バッチ版（複数 index を一括で復元）
std::vector<IndexToCommandResult> index_to_command_batch(const std::vector<int>& indices, const GameState &state, int player_id, const CardDB &card_db);

注意: `CommandDef` は既存のエンジン側 Command 形式に合わせる。

4. パフォーマンス設計
   - index -> (category, slot) の計算は整数演算のみで O(1)。
   - slot→instance の解決は主に配列アクセス（state.players[player].hand[slot] 等）で O(1)。
   - バッチ関数は複数 index をまとめて処理し、メモリ・分岐予測を改善する。

5. テストプラン
   - 単体テスト: 固定 state と既知の手札/battle 配置に対して、期待される `CommandDef` を返す。
   - 回帰テスト: Python の `dm_ai_module.index_to_command` と出力一致を確認するテスト。
   - ストレステスト: 大量の index をバッチで復元して、レイテンシ・メモリを測定。

6. 移行ステップ
   1. ドキュメントとテストケースを作成（このファイル）。
   2. C++ ヘッダとスタブ実装を追加（`index_to_command.hpp` / `.cpp`）。
   3. Pybind 経由で `dm_ai_module.index_to_command` を置き換えられるようにエクスポート。
   4. Python の互換 shim を削除し、Python からネイティブを呼ぶように切替える。
   5. パフォーマンス評価とチューニング。

7. 注意点 / リスク
   - ネイティブ実装では Python の柔軟なフォールバックが失われるため、エッジケース検出用に "unsafe fallback" を残す検討。
   - slot→instance の解決は state の整合性に依存する。MCTS の clone/undo 実装が正しくないと不整合が発生する。

8. 参考: Python 側 index_to_command の実装
- docs for testing should include a Python script that compares outputs of the native implementation vs the Python shim across randomized states.


作業見積もり
- ドキュメント + 単体テスト作成: 1-2 日
- C++ 実装（基本）: 1-3 日（依存環境による）
- Pybind export + テスト統合: 1 日
- パフォーマンスチューニング: 1-3 日


次の具体的作業案
- `docs/design_index_to_command.md` をリポジトリに追加（完了）
- 次に C++ スタブヘッダを追加して、CI 上でのビルド準備に移るか、まず Python 側の追加検証テストを作るかを選んでください。
