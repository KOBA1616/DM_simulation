# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

## 2. 実装済み機能 (Completed Features)

### 2.1 コアエンジン (C++)
*   **基本ルール**: ターンの進行、マナチャージ、召喚、攻撃、ブロック、シールドブレイク、勝利条件。
*   **ゾーン管理**: 山札、手札、マナゾーン、バトルゾーン、シールドゾーン、墓地。
*   **カード効果**:
    *   **GenericCardSystemのリファクタリング完了 (2025/02/XX)**: `GenericCardSystem` 内の巨大な `switch` 文を廃止し、`Handler/Command Pattern` を導入しました。各アクションタイプ (`DRAW_CARD`, `TAP`, `DESTROY` など) は独立した `IActionHandler` クラスに分割され、`EffectSystem` レジストリによって管理されます。
    *   **スタックシステムの実装 (2025/03/XX)**: トリガー能力 (`ON_PLAY`など) の即時解決を廃止し、`PendingEffect` リスト（スタック）に一度積んでから、プレイヤー（またはAI）が順次解決する仕組みに変更しました。これにより、連鎖的な能力発動や、同時発動時の解決順序選択が可能になりました。
    *   `dm::engine::GenericCardSystem` によるJSONベースの効果処理。
    *   Trigger: ON_PLAY, ON_ATTACK, ON_DESTROY, S_TRIGGER, PASSIVE_CONST (Speed Attacker, Blocker, etc.), ON_ATTACK_FROM_HAND (Revolution Change).
    *   Actions: DRAW, ADD_MANA, DESTROY, TAP, UNTAP, RETURN_TO_HAND, SEARCH_DECK, SHUFFLE_DECK, ADD_SHIELD, SEND_SHIELD_TO_GRAVE, MEKRAID, REVOLUTION_CHANGE.
*   **アクション生成**:
    *   **ActionGeneratorのリファクタリング完了 (2025/02/XX)**: `ActionGenerator` 内の巨大な条件分岐を廃止し、`Strategy Pattern` を導入しました。Pending Effects, Stack, 各フェーズ (Mana, Main, Attack, Block) のロジックは独立した `IActionStrategy` 実装クラスに分割されました。
*   **JSON読み込み**: `dm::engine::JsonLoader` によるカード定義のロード。
*   **シミュレーション**: `dm::ai::MCTS` および `dm::ai::ParallelRunner` による並列モンテカルロ木探索。

### 2.2 AI & 学習 (Python/C++)
*   **モデル**: ResNetベースのニューラルネットワーク (`AlphaZeroNetwork` in PyTorch).
*   **推論**: C++からのバッチ推論コールバック (`register_batch_inference_numpy`)。
*   **データ収集**: 自己対戦による学習データ生成 (`collect_training_data.py`).
*   **学習ループ**: `train_simple.py` によるモデル更新。

### 2.3 GUI (Python/PyQt6)
*   **カードエディタ**: カードデータの作成・編集 (JSON形式)。日本語対応。Ver 2.0 (Logic Tree + Property Inspector) へ改修済み。
*   **シミュレーション対話**: 対戦の観戦やデバッグ。

### 2.4 テスト状況と課題 (Testing Status & Identified Issues) (2025/03/XX 更新)

#### テスト実行結果
Pythonバインディング (`bindings.cpp`) の不整合を解消し、テストスイートの大部分が通過することを確認しました。
*   **バインディング修正**: `GameInstance`, `GameState` (current_phase, stack_zone, turn_stats), `CardInstance`, `ActionDef`, `FilterDef` の欠落していたメソッドやプロパティを追加。
*   **Enum定義**: `SpawnSource`, `DECLARE_PLAY`, `PAY_COST`, `RESOLVE_PLAY` などを追加。
*   **AI関連**: `POMDPInference`, `ParametricBelief` のスタブを追加。
*   **Numpy依存**: テストに必要な `numpy`, `torch` をインストール。

#### 修復済みのテスト (Resolved Issues)
*   **`python/tests/test_meta_counter.py` の修正**: `ActionGenerator` (PendingEffectStrategy) が生成する `PLAY_CARD_INTERNAL` アクションに `card_id` が正しく設定されておらず、テスト側のフィルタリングで除外されていました。C++側の `PendingEffectStrategy::generate` を修正し、`bindings.cpp` の `add_card_to_hand` 等で `card_owner_map` を正しく更新するようにしたことで、`game_state.get_card_instance` が正常に機能し、`card_id` が設定されるようになりました。
*   **`python/tests/test_engine_basics.py` のリファクタリング**: 非推奨の `GameInstance` 依存コードを最新の `GameState` と `PhaseManager` を使用する形に書き換え、最新のエンジンAPIとの互換性を確保しました。
*   **`python/tests/test_atomic_actions.py` の修正**: `DevTools.move_cards` の引数不足およびダミーカードの追加漏れによるテスト失敗を修正しました。

#### 特定された課題 (Remaining Issues)
*   `python/tests/test_just_diver.py` および `test_json_loader.py` の一部アサーション失敗は、テストデータまたはテストシナリオの修正が必要と考えられます。
    *   特に `test_just_diver_attack.py` は、ターン進行 (`turn_number`) と ジャストダイバーの解除タイミングの解釈に齟齬がある可能性があります。

これらの問題は次のフェーズで対処予定ですが、コアエンジンの主要機能（マナ、召喚、攻撃、シールドブレイク、効果解決）は安定しています。

## 3. 実装完了機能 (Phase 6 & 7 Implementation Summary)

### Phase 6: 多色・進化・高度な選択 (Multi-color, Evolution, Advanced Selection)

1.  **多色システム (Multi-color System)**
    *   **データ構造**: `CardDefinition.civilizations` を単一から配列へ変更し、複数文明に対応。
    *   **厳格なマナ支払い**: `ManaSystem` にバックトラック法を用いた厳密なコスト支払いロジック (`solve_payment`) を実装し、多色カードが1枚につき1文明のみを提供することを保証。
    *   **タップイン処理**: `TapInUtils` を導入し、多色カードがマナゾーンに置かれる際にタップインされるルール（および `untap_in` キーワードによる例外）を実装。

2.  **進化・階層構造 (Evolution & Hierarchy)**
    *   **階層構造**: `CardInstance` に `underlying_cards` を追加し、進化元などのカードスタックをサポート。
    *   **進化クリーチャー**: `ActionGenerator` を拡張し、進化クリーチャーの使用時に進化元を選択するアクションを生成可能に。
    *   **NEOクリーチャー**: 通常召喚と進化召喚の両方を選択可能にするロジックを追加。
    *   **クリーンアップ処理**: `ZoneUtils::on_leave_battle_zone` を実装し、一番上のカードがバトルゾーンを離れた際、下のカードが墓地に置かれるルールを統一的に適用。

3.  **高度な選択 (Advanced Selection)**
    *   **フィルタリング**: `FilterDef` に `selection_mode` (MIN/MAX/RANDOM) と `selection_sort_key` (COST/POWER) を追加。
    *   **ロジック**: `PendingEffectStrategy` にて、選択候補をソートおよびフィルタリングし、「パワーが最小のクリーチャーを選ぶ」等の処理を自動化。
    *   **複合条件**: `FilterDef` に `and_conditions` を追加し、再帰的なAND条件フィルタをサポート。
    *   **逆選択 (Inverse Selection)**: `ActionDef` に `inverse_target` フラグを追加。「選ばなかったものを破壊」するロジックを `DestroyHandler` 等に実装。

### Phase 7: 複雑な領域移動・パッシブ効果 (Allocation & Passives)

4.  **複雑な領域移動 (Complex Allocation)**
    *   **共通基盤**: `ZoneUtils::find_and_remove` を実装し、バトルゾーン、手札、マナ、シールド、墓地、および **Effect Buffer** から安全にカードを検索・移動させるロジックを共通化。
    *   **ハンドラの統合**: `DestroyHandler`, `ReturnToHandHandler`, `ManaChargeHandler` が `ZoneUtils` を使用するように改修し、一時領域 (Buffer) からの移動をサポート。「3枚見て、1枚手札、1枚マナ、1枚墓地」のような振り分け処理が可能に。
    *   **カードの下に置く**: `EffectActionType::MOVE_TO_UNDER_CARD` およびハンドラを実装。

5.  **パッシブ効果・情報開示 (Passive Effects & Reveal)**
    *   **パッシブ効果システム**: `PassiveEffectSystem` を実装。`EffectResolver::get_creature_power` をフックし、常在型能力（全体パワー修正など）を動的に適用。
    *   **公開アクション**: `EffectActionType::REVEAL_CARDS` およびハンドラを実装。

### Phase 8 (New): スタックシステムとビルド安定化 (Stack System & Build Stabilization)

6.  **トリガースタック (Trigger Stack System) - 実装済み (2025/03/XX)**
    *   トリガー能力（CIP等）を即時実行せず、`PendingEffect` (Type: `TRIGGER_ABILITY`) として待機リストに追加。
    *   `EffectResolver` にて `RESOLVE_EFFECT` アクションを介して順次実行する仕組みを実装。
    *   `python/tests/test_trigger_stack.py` にて基本動作（トリガーの保留と解決）を検証済み。

7.  **ビルド環境の修復と安定化 (Build Fixes)**
    *   `CardDefinition` の文明定義（単数→複数）変更に伴うコンパイルエラー（`ManaSystem`, `JsonLoader`, `CsvLoader`, `TensorConverter`）を修正。
    *   Pythonバインディング (`bindings.cpp`) のインクルード漏れや型定義の不整合（`CardKeywords`, `Zone` Enum）を修正し、`dm_ai_module` の正常ビルドを回復。

8.  **超魂クロス (Metamorph) および 多色マナ厳格化 (Strict Multicolor Mana) - 実装済み (2025/03/XX)**
    *   **超魂クロス**: 817.1a (通常能力との共存) および 817.1b (下のカードから超魂能力のみ付与) のルールを実装。`GenericCardSystem::resolve_trigger` にて、`CardDefinition.metamorph_abilities` を用いて能力を適切に合成するロジックを確立。
    *   **多色マナ支払いの厳格化**: `ManaSystem` にバックトラック法を用いた厳密なコスト支払いロジック (`solve_payment`) を実装。多色カードが1枚につき1文明のみを提供し、かつ必要な全文明を過不足なく満たす組み合わせを探索・適用するように改修。

## 4. 今後のロードマップ (Roadmap)
*   **AlphaZeroサイクル自動化とPBT基盤構築**: 次フェーズ詳細参照。
*   **テスト環境の整備**: `test_just_diver.py` や `test_json_loader.py` の失敗原因を特定し、修正する。
*   **AIモデルの高度化**: PPO + LSTM アーキテクチャの導入。

## 5. AlphaZeroサイクル自動化とPBT実装計画 (Automated AlphaZero Cycle & PBT)

現在、個別に存在している「データ収集 (Collect)」「学習 (Train)」「評価 (Evaluate)」の各コンポーネントを有機的に結合し、継続的な強化学習サイクルを構築します。また、将来的な自己進化 (Population Based Training) のための基盤を整備します。

### 5.1 自動化ループの実装 (Automated Loop)
マスタースクリプト `dm_toolkit/training/run_alphazero_cycle.py` を新規作成し、以下のサイクルを自動化します。

1.  **Collect (収集)**: `dm_toolkit/training/collect_training_data.py` を呼び出し、現在の最良モデル (`best_model.pth`) を用いて自己対戦データを生成。
2.  **Train (学習)**: `dm_toolkit/training/train_simple.py` を呼び出し、生成されたデータを学習して新モデル候補 (`candidate_model.pth`) を作成。
3.  **Evaluate (評価)**: `dm_toolkit/training/verify_performance.py` を呼び出し、新旧モデルを対戦させる (Gatekeeper)。
    *   新モデルの勝率が基準 (例: 55%) を超えた場合、`best_model.pth` を更新。
4.  **Loop**: 設定された時間または世代数に達するまで上記を繰り返す。

**要件**:
*   **時間指定実行**: `--duration 6h` のように実行時間を指定可能にする。
*   **世代管理**: 各世代のモデルを `models/gen_X.pth` として保存し、履歴を残す。

### 5.2 自己対戦への移行 (Transition to Self-Play)
現在の `HeuristicAgent` (ルールベース) 同士の対戦によるデータ収集から、ニューラルネットワークを用いた自己対戦 (`AlphaZeroAgent` vs `AlphaZeroAgent`) へ移行します。

*   **改修対象**: `dm_toolkit/training/collect_training_data.py`
*   **変更点**:
    *   `--model_path` 引数を追加。
    *   モデルパスが指定された場合、`HeuristicAgent` の代わりにニューラルネットワークモデルをロードしたエージェントを使用する。
    *   推論サーバー (Batch Inference) との連携を確認する。

### 5.3 PBT (Population Based Training) 基盤の設計
単一の「最強モデル」だけでなく、複数のエージェント（母集団）を並行して進化させるためのディレクトリ構造と管理クラスを設計します。

*   **ディレクトリ構造**:
    ```
    data/
      agents/
        gen_001/
          agent_01/ (model.pth, stats.json)
          agent_02/
        gen_002/
          ...
    ```
*   **リーグ戦の準備**: `dm_toolkit/training/verify_deck_evolution.py` (現在スタブ) に、複数のエージェントをロードして総当たり戦を行うロジックを実装予定。

### 5.4 世代管理システムとストレージ戦略 (Generation Management System)

PBT（Population Based Training）を見据えた、モデルとデッキの世代管理およびストレージ最適化の要件を定義します。

#### A. 管理対象 (Managed Entities)
1.  **Model (AI Agent)**: `AlphaZeroNetwork`の重みファイル (`.pth`)。
    *   サイズ: 約20MB（推論用） / 約60~80MB（学習用チェックポイント）。
2.  **Deck (Card List)**: デッキ構成情報（JSON）。テキストデータのため容量は軽微。

#### B. ディレクトリ構造 (Directory Structure)
PBTへのスムーズな移行を考慮し、以下の階層構造を採用します。

```text
checkpoints/
  ├── production/          # 本番稼働用（現在の最強モデル）
  │    └── best_model.pth
  │
  ├── hall_of_fame/        # 殿堂入り（過去の強力なモデル、評価対戦用）
  │    ├── gen_0010.pth
  │    ├── gen_0050.pth
  │    └── gen_0100.pth
  │
  └── population/          # 進化中の個体群（一時保存、頻繁に入れ替わる）
       ├── agent_01/
       │    ├── gen_0005.pth
       │    └── deck.json
       └── agent_02/ ...
```

#### C. 世代保持ポリシー (Retention Policy)
ディスク容量の圧迫を防ぎつつ、過学習を防ぐための「アンチ・サイクリック」戦略として、以下の保持ルールを適用します。

1.  **最新 (Latest)**: 各エージェントにつき、**最新3世代**のみを保持し、それ以前は自動削除する。
2.  **殿堂入り (Hall of Fame)**: 対数的な間隔でモデルを永続保存する。
    *   例: Gen 1, 2, 4, 8, 16, 32, 64, 100, 200...
    *   目的: 自己対戦時の対戦相手として過去のモデルを使用し、戦略の多様性を維持する。
3.  **最高傑作 (Best)**: 評価プロセスでチャンピオンを倒したモデルは `production` フォルダへコピーされる。

**推定容量**: 開発初期（シングルエージェント）では、直近5世代＋殿堂入り10個程度で **1GB以下** に収まる設計とします。

#### D. ストレージ管理とデータクリーンアップ (Storage Management)
1.  **学習データ (Replay Buffer)**:
    *   `.npz` 形式の学習データは、最新の **50万〜100万サンプル** 分（スライディングウィンドウ）のみを保持し、古いファイルは自動的に削除する。
    *   目的: 過去の未熟なランダムプレイデータによる学習ノイズを排除するため。
2.  **自動削除ロジック**:
    *   `Trainer` クラスに、新世代保存時に保持ポリシーに基づき古いファイルを削除する `cleanup_old_checkpoints()` メソッドを実装する。

## 6. (Deferred) Phase 9 要件: AIモデルの高度化 (AI Model Refinement with PPO+LSTM)
*本フェーズはPBT基盤構築後に着手します。*

*   **基本構成**: Actor-Critic (PPO) + LSTM
*   **入力層**: 盤面ベクトル + カード埋め込み + 墓地・マナのコンボパーツ密度。
*   **学習戦略**: 模倣学習 -> 報酬シェイピング -> Action Masking。
