# 11. 詳細実装計画書 (Detailed Implementation Steps)

本ドキュメントは、開発リソース（AIモデルの性能やコンテキスト制限）が限られた状態でも開発を継続できるよう、タスクを極小単位（Step-by-Step）に分解したものである。
各ステップは「入力（編集するファイル）」と「出力（期待される動作）」が明確になっており、順次実行することでPhase 3以降の実装が完了する。

---

## Phase 3.1: 汎用カードシステム統合 (Generic Card System Integration)

**目的**: C++エンジンがJSON定義のカードデータを読み込み、正しく動作するようにする。

### Step 1: EffectResolverへのフック実装
*   **対象ファイル**: `src/engine/effects/effect_resolver.cpp`
*   **タスク**:
    *   `resolve_play_card` 内のハードコードされたCIP処理（`if (def.keywords.cip)` ブロック）を削除またはコメントアウトする。
    *   代わりに `GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, card.instance_id)` を呼び出す。
    *   `resolve_pending_effect` 内で `GenericCardSystem::resolve_effect` を呼び出すように変更する。
*   **検証**: `Bronze-Arm Tribe` をプレイした際、マナが増えること（`GenericCardSystem`経由で実行されること）。

### Step 2: 基本アクションの実装
*   **対象ファイル**: `src/engine/card_system/generic_card_system.cpp`
*   **タスク**: `resolve_action` 関数内の `switch` 文を拡張する。
    *   `DESTROY`: 対象（`scope`）が `TARGET_SELECT` の場合、保留中の効果（PendingEffect）として処理するロジックを追加。
    *   `TAP`: 対象をタップ状態にする。
    *   `RETURN_TO_HAND`: 対象を手札に戻す（バウンス）。
    *   `BREAK_SHIELD`: シールドをブレイクする。
*   **検証**: 各アクションを持つテスト用JSONカードを作成し、動作を確認する。

### Step 3: ターゲット選択ロジックの実装
*   **対象ファイル**: `src/engine/card_system/generic_card_system.cpp`
*   **タスク**: `select_targets` 関数を実装する。
    *   `FilterDef` (文明、種族、コスト、パワー、タップ状態) に基づいて、`GameState` から対象候補の `instance_id` リストを抽出するフィルタリング処理を書く。
    *   `TargetScope::PLAYER_OPPONENT` などのスコープ処理を実装する。

### Step 4: 既存カードの完全移行
*   **対象ファイル**: `data/cards.json`, `src/core/constants.hpp` (ID定義)
*   **タスク**:
    *   `Holy Awe` (ID: 3), `Terror Pit` (ID: 5), `Spiral Gate` (ID: 6) を `data/cards.json` に追記する。
    *   C++側のハードコード（IDによる分岐）を全て削除する。
*   **検証**: 既存のテストスクリプト `python/scripts/test_card_creation.py` がエラーなく動作し、かつゲームロジックが正常であることを確認する。

---

## Phase 3.2: カード生成ツール (Card Generator GUI)

**目的**: 自然言語またはGUI操作で `data/cards.json` を更新するツールを作成する。

### Step 1: GUIスケルトンの作成
*   **対象ファイル**: `tools/card_gen/app.py` (新規作成)
*   **ライブラリ**: `customtkinter` (推奨) または `PyQt6`。
*   **タスク**:
    *   画面左側: 入力フォーム（Name, Cost, Civ, Power, Races, Type）。
    *   画面中央: 効果リスト（Trigger, Actionの追加・削除ボタン）。
    *   画面右側: JSONプレビュー（Read-only Textbox）。
    *   下部: "Generate with AI" 入力欄とボタン。

### Step 2: JSON生成ロジック
*   **対象ファイル**: `tools/card_gen/json_builder.py`
*   **タスク**:
    *   GUIの入力値を `src/core/card_json_types.hpp` の構造に合わせたPython辞書に変換する関数を作成する。
    *   `json.dumps` で整形してプレビューに表示する。

### Step 3: Gemini API 連携
*   **対象ファイル**: `tools/card_gen/ai_assistant.py`
*   **タスク**:
    *   `config/secrets.json` からAPIキーを読み込む。
    *   `google.generativeai` ライブラリを使用して、ユーザーの自然言語入力（例: "出た時、相手獣1体を破壊"）をJSONスキーマに変換するプロンプトを投げる。
    *   返ってきたJSONをGUIに反映させる。

---

## Phase 3.3: LibTorch統合 (C++ Inference)

**目的**: Python依存を排除し、C++のみで推論を実行可能にする。

### Step 1: ライブラリの準備
*   **タスク**:
    *   LibTorch (Release, CPU版) をダウンロードし、`lib/libtorch` に配置する。
    *   `CMakeLists.txt` に `find_package(Torch REQUIRED)` とパス設定を追加する。

### Step 2: モデルローダーの実装
*   **対象ファイル**: `src/ai/inference/torch_model.hpp`, `.cpp`
*   **タスク**:
    *   `torch::jit::script::Module` をメンバに持つクラスを作成。
    *   `load(std::string path)` メソッドで `.pt` ファイルをロードする。
    *   `predict(std::vector<float> input)` メソッドで推論を実行し、`std::vector<float>` (Policy, Value) を返す。

### Step 3: 評価関数の差し替え
*   **対象ファイル**: `src/ai/evaluator/neural_evaluator.cpp`
*   **タスク**:
    *   現在の `HeuristicEvaluator` の代わりに、上記 `TorchModel` を使用する `NeuralEvaluator` クラスを実装する。
    *   MCTSのノード展開時にこれを呼び出す。

---

## Phase 4: リーグ学習とコンテンツ拡充

### Step 1: リーグマネージャーの実装 (Python)
*   **対象ファイル**: `python/training/league_manager.py`
*   **タスク**:
    *   過去のモデルチェックポイントをリスト管理するクラス。
    *   対戦相手選択ロジック（80% 最新, 10% 過去, 10% Bot）を実装。

### Step 2: 並列対戦ランナーの拡張
*   **対象ファイル**: `src/ai/self_play/parallel_runner.cpp`
*   **タスク**:
    *   Player 1 と Player 2 で異なるモデル（重み）を使用できるように改修する。
    *   C++側で「対戦相手のモデルパス」を受け取れるようにする。

---

## 決定事項（2025-12-01）

- 採用方針: **選択肢C（pybind11 を継続し、C++ 化を段階的に進める）を採用**します。
    - 理由: 現在の開発環境では LibTorch を即導入するには互換性や CI コスト等の懸念があり、短期的な開発速度と安定性を優先しました。pybind11 経由でのモデル呼び出しを最小化・最適化することで、当面のパフォーマンス要件を満たしつつ将来のネイティブ化（ONNX/LibTorch）への移行余地を残します。

- LibTorch の扱い: **導入を延期**します（オプションとして CMake の `USE_LIBTORCH` は維持）。将来的に以下の条件が整った段階で再検討します。
    - MSVC ベースのビルド環境を確定できたとき
    - CI でのバイナリ配布・キャッシュ戦略を用意できたとき
    - ONNX Runtime による POC を評価した結果、LibTorch に移行するメリットが明確なとき

## 開発計画（優先度順、短中期プラン）

短期（今〜2 週間）
- 1. Evaluator 抽象インターフェースを C++ 側に追加（`INeuralEvaluator` など）。既存 `HeuristicEvaluator` をこのインターフェースへ適合させる。
- 2. Python モデル呼び出しの最適化
    - バッチ推論 API を作成（Python 側で複数状態を一度に投げる）。
    - pybind11 バインディングで GIL を適切に解放するラッパーを追加（`bindings.cpp` にエンドポイント）。
    - 目的: 呼び出しオーバーヘッド削減と容易なベンチマーク。
- 3. 小規模ベンチマークを作成（レイテンシ、スループットを計測）。

中期（2〜6 週間）
- 4. ONNX Runtime POC: Python モデルを ONNX にエクスポートし、C++（ONNX Runtime）で動作させる。問題がなければ本番用 C++ バックエンド候補とする。
- 5. NeuralEvaluator（C++ 側）スタブを作成し、実行時に切替可能とする（Heuristic / Python-backed / ONNX-backed を切替）。

長期（6 週間〜）
- 6. 必要なら LibTorch を導入（`USE_LIBTORCH=ON` 時のビルドを整備）。ただしこれは上記 POC・CI 設計・環境整備が完了した後に実施します。

運用ルール
- - すべての変更はローカルで `python -m pytest -q` を通すこと。CI は `USE_LIBTORCH=OFF` で動作確認を行う。LibTorch を有効にする PR は別途レビュー対象とする。

連絡先/担当
- - 実装・CI のコーディネーションはリポジトリ管理者と協議しながら進めること。必要であれば環境整備（MSVC/LibTorch）を私がサポートします。


