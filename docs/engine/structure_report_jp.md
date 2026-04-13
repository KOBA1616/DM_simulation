# C++エンジン構造・移行状況調査報告書

## 1. ディレクトリ構成 (Current Directory Structure)

C++エンジン (`src/engine`) は機能別に明確に構造化されており、整理された状態です。

*   **`src/engine/systems/`**: ゲームロジックの中核
    *   `flow/`: フェーズ進行 (`PhaseSystem`)
    *   `mechanics/`: マナ、コスト支払いなどのメカニクス
    *   `rules/`: ルール制限 (`RestrictionSystem`, `ConditionSystem`)
    *   `director/`: ゲーム進行管理 (`GameLogicSystem`)
    *   `effects/`: 効果処理 (`TriggerManager`, `EffectSystem`)
    *   `breaker/`: シールドブレイク処理
*   **`src/engine/infrastructure/`**: データとIO
    *   `commands/`: コマンド定義 (`CommandDef`, `CommandSystem`)
    *   `data/`: カードデータ管理 (`CardRegistry`, `JsonLoader`)
    *   `pipeline/`: 処理パイプライン (`PipelineExecutor`)
*   **`src/engine/command_generation/`**: AI/自動化
    *   `intent_generator.cpp/hpp`: 行動生成ロジック (`IntentGenerator`)
*   **`src/bindings/`**: Pythonバインディング
    *   `bind_engine.cpp`: エンジン機能の公開 (`GameInstance`, `PhaseManager`, `CommandSystem` 等)

## 2. 重複実装 (Duplicate Implementations)

調査の結果、以下の主要な重複が見つかりました。特にPython側の「Shim（シム）」実装がC++側のロジックと大きく重複しています。

### A. `dm_ai_module.py` (Python Shim) vs C++ Engine
ルートディレクトリにある `dm_ai_module.py` は、C++拡張モジュール (`.pyd` / `.so`) が利用できない場合のフォールバックとして機能しますが、以下のクラス・ロジックがC++側と二重管理になっています。

*   **`PhaseManager`**: フェーズ遷移ロジックをPythonで簡易実装しており、C++の `PhaseSystem` と重複。
*   **`CommandSystem`**: `execute_command` 内でコマンドの解釈・実行を行っており、C++の `CommandSystem` と重複。
*   **`GameState` / `Player`**: 状態保持クラスが二重定義されています。

**影響**: C++側の変更（新しいコマンドやフェーズルールの追加）がPython Shimに自動反映されないため、乖離が進むリスクがあります。

### B. `dm_toolkit` 内の v1 / v2 ファイル群
`dm_toolkit` 内には移行過渡期のファイルが混在しており、機能的に重複に近い関係にあります。

*   **`commands.py` vs `commands_v2.py`**:
    *   `commands.py`: レガシーな `ActionGenerator` へのフォールバックや、`Action` を `ICommand` にラップするロジックを含みます。
    *   `commands_v2.py`: `dm_ai_module.generate_commands` を優先的に呼び出し、失敗時に `commands.py` にフォールバックする薄いラッパーです。
*   **`unified_execution.py` vs `unified_execution_v2.py`**:
    *   `unified_execution.py`: レガシー辞書を正規化する重厚なロジック (`to_command_dict`, `ensure_executable_command`) を含みます。
    *   `unified_execution_v2.py`: `unified_execution.py` に依存する薄いラッパーです。

## 3. ActionからCommandへの移行状況

### C++側 (Engine): **完了 (Complete)**
*   ソースコード内にレガシーな `class Action` や `struct Action` の定義は見当たりません。
*   内部ロジックは全て `CommandDef` と `Instruction` ベースで動作しています。

### Python側 (Toolkit/Bindings): **過渡期 (Hybrid/Bridging)**
*   **ブリッジの存在**: `dm_toolkit/action_to_command.py` が「Single Source of Truth」として機能し、レガシーな `Action` 辞書（例: `value1`, `str_val` などの古いキー）を新しい `Command` 辞書に変換し続けています。
*   **ラッパーの利用**: `dm_toolkit/commands.py` の `wrap_action` 関数により、生成されたレガシーActionオブジェクトを `ICommand` インターフェースで包み込み、実行時に `EngineCompat` を経由してC++エンジンに渡す仕組みが稼働しています。
*   **完全移行の阻害要因**: GUIや既存のテストコード、あるいはAIモデルの入出力が依然としてレガシーなAction形式（辞書構造）に依存している可能性が高いため、この変換レイヤーを削除できない状態です。

## 4. 削除候補のレガシーコード (Candidates for Deletion)

以下のファイルは、環境の方針（「ネイティブ拡張必須」とするか否か）や整理の進行度に応じて削除・統合が推奨されます。

1.  **`dm_ai_module.py` (Python Shim)**
    *   **理由**: C++エンジンのロジックと重複しており、メンテナンスコストが高い。
    *   **条件**: 開発・テスト環境においてC++拡張モジュール (`dm_ai_module.pyd/.so`) の利用を**必須**とするなら削除可能です。ただし、CI環境などでPythonのみでテストを回す需要がある場合は残す必要があります。

2.  **`scripts/` 内の非推奨スクリプト**
    *   `scripts/check_gui_form_integrity.py`
    *   `scripts/inspect_mcts_signature.py`
    *   `scripts/register_and_run.py`
    *   `scripts/diagnose_game_training.py`
    *   **理由**: ファイル名や内容から、特定の一時的なデバッグや検証に使われた形跡があり、現在はメンテナンスされていない可能性が高いです。

3.  **`dm_toolkit` の重複ファイル整理**
    *   `dm_toolkit/commands_v2.py`: `commands.py` に統合（または `v2` を正として `commands.py` の内容を移管）し、一本化することを推奨します。
    *   `dm_toolkit/unified_execution_v2.py`: 同様に `unified_execution.py` との一本化を推奨します。

4.  **`dm_toolkit/action_to_command.py` (将来的)**
    *   **理由**: クライアントサイド（GUI/AI）が完全にCommandベースになれば不要になりますが、現時点では削除できません。
