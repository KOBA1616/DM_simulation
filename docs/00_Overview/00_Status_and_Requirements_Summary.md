# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

## 2. 実装済み機能 (Completed Features)

### 2.1 コアエンジン (C++)
*   **基本ルール**: ターンの進行、マナチャージ、召喚、攻撃、ブロック、シールドブレイク、勝利条件。
*   **ゾーン管理**: 山札、手札、マナゾーン、バトルゾーン、シールドゾーン、墓地。
*   **カード効果**:
    *   `dm::engine::GenericCardSystem` によるJSONベースの効果処理。
    *   Trigger: ON_PLAY, ON_ATTACK, ON_DESTROY, S_TRIGGER, PASSIVE_CONST (Speed Attacker, Blocker, etc.), ON_ATTACK_FROM_HAND (Revolution Change).
    *   Actions: DRAW, ADD_MANA, DESTROY, TAP, UNTAP, RETURN_TO_HAND, SEARCH_DECK, SHUFFLE_DECK, ADD_SHIELD, SEND_SHIELD_TO_GRAVE, MEKRAID, REVOLUTION_CHANGE.
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

### 2.4 テスト状況と課題 (Testing Status & Identified Issues) (2025/XX/XX 更新)

#### テスト実行結果
既存のPythonテストスイート (`python/tests/`) に対して網羅的な実行を行い、以下の修正を行いました。
*   **テストコードの修正**: C++バインディング (`dm_ai_module`) の最新仕様に合わせて `CardData` コンストラクタ呼び出し等を修正しました。
*   **非推奨コードの削除**: `ActionType.DIRECT_ATTACK` 等の削除された定数への参照を修正しました。

#### 特定された課題 (Engine Issues)
テスト実行により、C++エンジン側の以下の挙動に関する課題が特定されました。

1.  **Meta Counter (Internal Play) の挙動不整合**
    *   **現象**: `ActionType::PLAY_CARD_INTERNAL` が手札 (`HAND_SUMMON`) から生成された場合、`EffectResolver` がカードをスタックへ移動せず、手札に残ったまま解決しようとして失敗する。
    *   **原因**: `EffectResolver` の `resolve_play_from_stack` は、スタックまたはバッファ内のカードのみを検索対象としており、手札からの直接的な内部プレイを想定していない（あるいは移動ロジックが欠落している）。
    *   **Status**: **修正完了 (2025/XX/XX)** - `ActionGenerator` にコントローラー情報の伝播を追加し、`EffectResolver` に `PLAY_CARD_INTERNAL` 実行後のPending Effect削除処理を追加しました。`python/tests/test_meta_counter.py` を通過することを確認済み。

2.  **Ninja Strike (Reaction System) の不発**
    *   **現象**: Ninja Strikeの条件を満たす状況下でも、リアクションウィンドウ（`PendingEffect`）が生成されるものの、`DECLARE_REACTION` アクションが生成されない。
    *   **原因**: エンジンイベント `"ON_ATTACK"` と JSON定義 `"ON_BLOCK_OR_ATTACK"` の文字列完全一致比較により、条件不一致と判定されていた。
    *   **Status**: **修正完了 (2025/02/XX)** - `ReactionSystem` および `ActionGenerator` にて、`"ON_BLOCK_OR_ATTACK"` が `"ON_ATTACK"`/`"ON_BLOCK"` イベントにもマッチするように緩和処理を追加。`python/tests/test_ninja_strike.py` を通過。

3.  **Just Diver (Play Action) の生成失敗**
    *   **現象**: Just Diverを持つクリーチャーのプレイアクションは生成されるが、プレイ後のターンでも対戦相手が対象に取れてしまう。
    *   **原因**: クリーチャーがバトルゾーンに出る際、`CardInstance.turn_played` プロパティが設定されておらず、期間判定（「このターン」）が正しく機能していなかった。
    *   **Status**: **修正完了 (2025/02/XX)** - `EffectResolver::resolve_play_from_stack` にて `turn_played` を設定するよう修正。`python/tests/test_just_diver.py` を通過。

4.  **Pythonバインディングの制約**
    *   `std::vector` を返すプロパティ（`mana_zone` 等）はPython側ではコピーとなるため、要素への代入（`is_tapped = False`）がC++側に反映されない。テストコードでは `add_card_to_mana` 等の専用ヘルパーを使用する必要がある。

## 3. 次のステップの要件 (Next Requirements)

### 3.1 テストコードの整理と動作確認
*   `tests/` ディレクトリ内のアドホックなテストを `python/tests/` へ統合。
*   原子アクション (Draw, Mana Charge, Tap, Break Shield, Move Card) の動作を検証する `python/tests/test_atomic_actions.py` の作成と維持。
*   **Status**: 完了 (2025/XX/XX)

### 3.2 GUIの日本語化拡充
*   `python/gui/card_editor.py` および `localization.py` を更新し、英語のハードコードを排除して日本語表示に対応する。
*   **Status**: 完了 (2025/XX/XX)

### 3.3 PythonコードのC++移行 (Migration Candidates)

パフォーマンス向上とロジックの堅牢化のため、以下のPython実装部分をC++へ移行することを計画しています。

1.  **シナリオ実行 (`ScenarioRunner`)**
    *   現状: `python/training/scenario_runner.py` でPython側でゲームループを回している。
    *   移行案: `ScenarioExecutor` (C++) を実装し、対戦ループをC++側で完結させる。
    *   **Status**: 完了 (2025/12/XX) - `ScenarioExecutor` クラス実装済み、`ScenarioRunner` から利用。

2.  **デッキ進化/検証 (`DeckEvolution` / `VerifyPerformance`)**
    *   現状: `verify_deck_evolution.py` や `verify_performance.py` の一部ロジックがPython。
    *   移行案: 進化ロジック (遺伝的アルゴリズムの選択・交叉など) はPythonでも良いが、評価のための対戦実行ループは完全に `ParallelRunner` (C++) に任せる。
    *   補足: 既に `ParallelRunner` を使用しているが、セットアップや結果集計をよりC++側へ寄せ、Pythonは設定と起動のみにする。

3.  **複雑なカード効果のPython側ロジック (もしあれば)**
    *   現状: ほぼ全てのカード効果は `GenericCardSystem` (C++) に移行済み。
    *   確認事項: Python側で `register_card_functions` 等を使って実装されているレガシーな効果があれば、JSON定義 + C++実装へ完全移行する。

4.  **AI学習データ生成の制御**
    *   現状: `collect_training_data.py` が `dm_ai_module.DataCollector` を呼んでいるが、ループ制御の一部がPython。
    *   移行案: `DataCollector` が指定エピソード数を完遂するまでPythonに制御を戻さないようにする (現状も近い形だが、メモリ管理を厳密にするためC++側で完結させる)。

### 3.4 変数連携システム (Variable Linking System)

アクション間で値を渡すための「実行コンテキスト (execution_context)」を強化し、動的な数値参照を実現します。

*   **コンテキストの実装**: `PendingEffect` および `GenericCardSystem` 内で `std::map<std::string, int> execution_context` を保持・伝播させる。
*   **新規アクション**:
    *   `COUNT_CARDS`: 指定ゾーン（BATTLE_ZONE, GRAVEYARD等）の条件に合うカード数をカウントし、変数に保存する。
    *   `GET_GAME_STAT`: マナゾーンの文明数 (`MANA_CIVILIZATION_COUNT`) 等の統計値を取得し、変数に保存する。
    *   `SEND_TO_DECK_BOTTOM`: 手札等から選択したカードを山札の下に送る。
*   **既存アクションの拡張**:
    *   `DRAW_CARD` 等で `input_value_key` が設定されている場合、固定値ではなくコンテキスト内の変数値を使用する。
*   **目的**: 「自分のクリーチャーの数だけドローする」「マナゾーンの文明数分ドローして戻す」といった複雑な効果をJSON定義のみで実現可能にする。
*   **Status**: 完了 (2025/XX/XX) - C++側実装およびエディタ側対応済み。

### 3.5 カードエディタ GUI実装仕様 (Ver 2.0)

**1. 背景と目的 (Context & Objectives)**
*   **現状の課題**: 従来のリスト形式のエディタでは、変数リンク（値の受け渡し）を含む複雑なロジックの構築が困難であり、JSON直接編集はタイプミスを招きやすい。
*   **目指すゴール**: 原子アクション（Atomic Actions）の組み合わせによる無限に近いカード能力のGUI実装と、視覚的な補助によるロジックミス防止。

**2. デザイン方針：標準IDEレイアウト (Standard IDE Layout)**
*   **採用デザイン**: 「左：ツリービュー / 右：プロパティインスペクタ」 の2ペイン構成。
*   **選定理由**:
    *   **情報の階層化**: Card -> Effect -> Action の入れ子構造を一望して管理するのに最適。
    *   **開発効率**: 標準的なIDEの操作感に合わせ、ユーザーの迷いを減らす。
    *   **実装の堅実性**: ノードグラフ方式よりもPyQt標準ウィジェットでの実装が容易で動作も軽量。

**3. 核となる機能：変数リンクシステム (Variable Linking)**
*   **採用方式**: ドロップダウンリストによる選択方式。
*   **選定理由**:
    *   **安全性**: 存在する変数のみを選択可能にし、表記ゆれによるバグを根絶。
    *   **スコープの明確化**: 「自分より前のアクションの結果しか使えない」ルールをGUI上で自然に強制。

**4. 具体的な実装詳細 (Implementation Details)**
*   **A. クラス・コンポーネント構成**: `python/gui/card_editor.py` を全面的に改修。
    *   `CardEditor`: `QSplitter` を使用し、左右ペインサイズ調整可能なメインウィンドウ。JSON I/Oを担当。
    *   `LogicTreeWidget` (左ペイン): `QTreeView` + `QStandardItemModel`。Card/Effect/Actionの階層データを保持し、ドラッグ＆ドロップによる順序入れ替えをサポート。各アイテムの `UserRole` にデータを保持。
    *   `PropertyInspector` (右ペイン): `QStackedWidget` を内包し、選択アイテムに応じて `CardEditForm`, `EffectEditForm`, `ActionEditForm` を切り替え。変更は即座にツリーモデルへ反映。
*   **B. 変数参照のアルゴリズム**: `ActionEditForm` の「Input Key」候補生成ロジック。
    *   同一Effect内の兄弟アクションを走査し、自身より前にあるアクションの `output_value_key` を収集してリスト化する。
    *   表示形式: `"{key} (from #{index} {type})"`
*   **C. データ永続化 (Saving)**:
    *   `export_model_to_json()`: ツリーのルートから Card -> Effect -> Action を再帰的に辿り、`UserRole` のデータを結合して完全なJSONリストを再構築する。

**Status**: 完了 (2025/XX/XX)

### 3.6 C++移行における改修詳細 (C++ Migration Details)

以下の3つのコンポーネントについて、C++側へのロジック移管と最適化を行います。

**1. データ収集の高速化 (DataCollector)**
*   **対象ファイル**: `src/ai/data_collection/data_collector.hpp`, `.cpp`
*   **目的**: Pythonで行っているゲームループとデータ蓄積をC++へ移管し、オーバーヘッドを削減する。
*   **具体的な実装方法**:
    *   `collect_data_batch_heuristic(int episodes)` メソッドを新規実装する。
    *   内部で `HeuristicAgent` インスタンスを作成する。
    *   `episodes` 回数分、`GameInstance` の生成からゲーム終了までをループさせる。
    *   各ターン、`TensorConverter` を呼び出して `states_masked`, `states_full` を取得し、`HeuristicAgent` の決定した手を `policy` として記録する。
    *   全てのデータを `CollectedBatch` 構造体（`std::vector` の塊）に格納して返す。
*   **課題**:
    *   **メモリ使用量**: 1エピソードあたり平均100ターン、1ターンあたり数KBのデータが発生するため、10万エピソードなどを一度に要求するとメモリ不足になる恐れがある。Python側で適切なチャンクサイズ（例: 1000エピソードごと）に分割して呼び出す設計にする。
*   **Status**: 完了 (2025/XX/XX)

**2. シナリオ検証の最適化 (ParallelRunner for Scenarios)**
*   **対象ファイル**: `src/ai/self_play/parallel_runner.hpp`, `.cpp`
*   **目的**: Python側での `GameInstance` 大量生成（ボトルネック）を回避し、C++内で設定から即座に生成・並列実行する。
*   **具体的な実装方法**:
    *   `play_scenario_match(ScenarioConfig config, int num_games, ...)` メソッドを追加する。
    *   引数として `ScenarioConfig` を受け取る。
    *   OpenMP等の並列処理ブロック内で、この `config` をコピーして `GameInstance::reset_with_scenario` を呼び出し、初期盤面を作成する。
    *   その後、既存の並列対戦ロジックを実行する。
*   **課題**:
    *   **バインディングの整合性**: `ScenarioConfig` の全フィールドが正しくC++に渡るか確認が必要。特に可変長の配列（手札やマナゾーンのカードIDリスト）の変換コストに注意が必要だが、インスタンス生成コストに比べれば軽微。
*   **Status**: 完了 (2025/XX/XX)

**3. デッキ進化システムの基盤 (ParallelRunner for Deck Evolution)**
*   **対象ファイル**: `src/ai/self_play/parallel_runner.hpp`, `.cpp`
*   **目的**: デッキリスト（カードIDの配列）だけを渡して高速に対戦結果を得られるようにする。
*   **具体的な実装方法**:
    *   `play_deck_matchup(vector<int> deck1, vector<int> deck2, int num_games, ...)` を実装する。
    *   `GameInstance` を生成し、指定されたデッキリストで初期化して対戦を行う。
*   **再現性の確保**:
    *   デッキのシャッフルや乱数シードの管理が重要。`ParallelRunner` 内部でスレッドごとに異なるシードを持つ乱数生成器 (`std::mt19937`) を適切に初期化する必要がある。
*   **Status**: 完了 (2025/XX/XX)

## 4. 今後のロードマップ (Roadmap)
*   **Phase 6**: サーチ、シールド操作の実装 (完了)。
*   **Phase 7**: 高度なギミック (超次元、GRなど) の検討。
*   **Phase 8**: AIモデルの高度化 (Transformerなど)。
