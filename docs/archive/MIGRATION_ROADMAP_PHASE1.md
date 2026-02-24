# CommandDef移行ロードマップ Phase 1 完了報告

## 概要
Phase 1の「CommandDef移行のための基盤強化」が完了しました。

### 現状の分析と実装完了項目 (Phase 1)
C++エンジンとPythonバインディング、およびPython側のビルダツールの拡張が完了し、`CommandDef` オブジェクトをPython側でネイティブに扱い、辞書への変換もスムーズに行える状態になりました。

1.  **C++ `CommandDef` の拡張**:
    *   `src/core/card_json_types.hpp` に `options` フィールド (`std::vector<std::vector<CommandDef>>`) を追加し、`CHOICE` コマンドのような複雑な入れ子構造をネイティブで表現可能にしました。
2.  **Pythonバインディングの更新**:
    *   `CommandDef` オブジェクトが `to_dict()` メソッドで `options` フィールドを含めて正しく辞書化されるように修正しました。
3.  **CommandBuilder の `native=True` 対応**:
    *   `dm_toolkit/command_builders.py` を拡張し、`build_choice_command` や `build_attack_player_command` などで `native=True` を指定した際に、正しく `dm_ai_module.CommandDef` オブジェクト（およびその入れ子構造）を生成するようにしました。
4.  **互換レイヤーのパススルー実装**:
    *   `dm_toolkit/action_to_command.py` にて、入力が既に `CommandDef` オブジェクトである場合、レガシーな変換ロジックを通さずに即座に `to_dict()` して返す処理を追加しました。

これにより、既存の辞書ベースのパイプラインを壊すことなく、新しいネイティブオブジェクトを徐々に導入できる準備が整いました。

---

### 今後のロードマップと残タスク

現在は **Phase 2** に移行可能な状態です。

#### Phase 2: テストコードの移行 (Next Step)
**目的:** テストコードから「生の辞書リテラル」を排除し、Type-safeなビルダー利用へ切り替える。
*   [ ] **テストのリファクタリング**: `tests/` 以下の主要なテスト（例: `tests/test_game_flow_minimal.py`）で、`{'type': 'DRAW_CARD', ...}` と書かれている箇所を `cb.build_draw_command(..., native=True)` 等に書き換える。
*   [ ] **検証**: 書き換えたテストが既存エンジンで正しく動作することを確認する（Phase 1の成果により、`action_to_command` が辞書化してエンジンに渡すため動くはずです）。

#### Phase 3: 互換層の分離とメインパスの変更
**目的:** `CommandDef` を第一級市民（First-Class Citizen）として扱い、無駄な辞書変換をなくす。
*   [ ] **実行パスの改修**: `unified_execution.py` の `ensure_executable_command` 等で、`CommandDef` が渡された場合に辞書に変換せず、そのままC++側の `execute_command` に渡すフローを確立する（現在は `action_to_command` で一度辞書に戻している）。
*   [ ] **レガシーパスの隔離**: `action_to_command.py` の利用を、古い形式のデータを読み込む箇所のみに限定する。

#### Phase 4: 完全移行とクリーンアップ
**目的:** レガシーコードの削除。
*   [ ] **クライアントコード更新**: AIエージェントやGUIが最初から `CommandDef` を生成するように修正。
*   [ ] **不要コード削除**: `compat.py` や `action_to_command.py` (アーカイブ化) の削除。
