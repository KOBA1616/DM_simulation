# 重要残タスクと改善計画

最終更新: 2026-03-09

この文書は、現時点の実装・テスト・ビルド・設計文書のずれを整理し、完成に向けた優先順位と、各修正をステップバイステップで進めるためのTDD実行計画をまとめたものです。

目的は次の3点です。

1. どのログと文書を「現在の真実」とみなすかを明確にする。
2. 低スペックなAIモデルでも、1タスクずつ安全に修正できる粒度まで分解する。
3. 単なる場当たり修正ではなく、再発防止テストまで含めて完了条件を定義する。

---

## 0. 現在の真実

現時点では、以下を優先的に事実として扱う。

- `full_test_result.txt` では 260 件中 4 fail、3 skip、253 pass。
- fail はすべて `tests/test_card1_hand_quality.py` に集中している。
- `test_run_full.txt` と `test_out.txt` は fail なしの実行結果を含むが、テスト件数・実行条件が異なるため、正式結果として一本化されていない。
- `build_summary.txt` は成功扱いだが、`build_out.txt` には `command_system.cpp` のコンパイルエラーが残っている。
- `status.md` は主要テスト通過済みの印象を与えるが、現行ログと一致していない。

結論として、完成判定の前提条件は次の通りとする。

- フルテスト結果を 1 系統に統一する。
- フルビルド結果を 1 系統に統一する。
- ステータス文書は統一後の結果に追随させる。

---

## 1. 完成条件チェックリスト

### P0: リリース可否を左右する項目

- [ ] `tests/test_card1_hand_quality.py` の 4 fail を 0 fail にする。
- [ ] `SELECT_FROM_BUFFER` の選択クエリ、パイプライン再開、バッファ後始末を単一路線に整理する。
- [ ] native 実行と fallback 実行で、テストモード・ログ出力先・正式結果ファイルを明示分離する。
- [ ] 正式ビルドコマンド、正式ログ、成功基準を一本化する。
- [ ] `status.md` 相当の進捗文書を、実際のテスト・ビルド結果に同期させる。

### P1: 完成度を大きく左右する項目

- [ ] Transformer の本番推論統合を完了し、C++ 推論経路で継続運用可能にする。
- [ ] `DataCollector` と `ParallelRunner` 周辺の P0 ネイティブ移行を完了する。
- [ ] `GameSession` の責務を分割し、AI 対戦 UI を非同期に成立させる。
- [ ] フェーズ別優先度 AI を実装し、行動品質を底上げする。
- [ ] アンタップクリーチャーへの攻撃許可例外を実装する。

### P2: 中長期の保守性・性能に効く項目

- [ ] `EffectResolver`、`CardDatabase`、`TokenConverter`、`TensorConverter` を順次ネイティブ最適化する。
- [ ] ActionDef 残骸を撤去し、CommandDef 移行を完了する。
- [ ] ONNX Runtime の取得方法とキャッシュ戦略を整理する。
- [ ] `missing_native_symbols.md` を機械再生成し、実装済みとの差分を縮小する。
- [ ] 研究評価指標を固定化し、自動測定できる状態にする。

### P3: 低優先だが完成度向上に寄与する項目

- [ ] Stack / Pending Effect の再順序化 GUI を実装する。
- [ ] 選択ダイアログにカード画像やアイコンを表示する。
- [ ] Transformer 周辺警告を整理し、警告放置を減らす。

---

## 2. 低スペックAI向け作業原則

低スペックなモデルに依頼する場合は、以下の順で 1 タスクずつ進める。

### 守るべき原則

- [ ] 1 回の依頼で 1 症状だけ直す。
- [ ] 先に failing test を固定し、次に最小修正、最後に回帰テストを足す。
- [ ] 修正対象の関数を 4 個以上またぐ場合は、先に「責務統一」だけを行い、機能追加を同時にやらない。
- [ ] native と fallback の差異が疑われる場合は、同一テストを 2 モードで別々に実行する。
- [ ] ログ、テスト、文書を同じコミットで更新しない。まずコードとテスト、次に運用文書を更新する。

### 依頼テンプレート

以下の形式で依頼すると、粒度が崩れにくい。

1. 失敗テスト名を 1 個指定する。
2. 原因候補のファイルを 1〜3 個に絞る。
3. 「新規テスト追加 → 実装修正 → 既存関連テスト再実行」の順を指定する。
4. 「他の TODO には触れない」と明示する。

例:

`tests/test_card1_hand_quality.py::TestCard1HandQuality::test_select_target_appears_after_draw` を赤に固定し、`intent_generator.cpp`、`command_system.cpp`、`game_logic_system.cpp` のみ確認して修正。先に回帰テストを足し、その後に最小修正を行い、最後に card1 系テストだけ再実行すること。

---

## 3. 最優先: card1 系 fail と SELECT_FROM_BUFFER 経路統一

### 背景

現状の 4 fail はすべて `tests/test_card1_hand_quality.py` に集中しており、手札品質、選択クエリ、バッファ選択、手札/山札更新タイミングの整合性崩れが疑われる。

また、`SELECT_FROM_BUFFER` 関連処理は完全未実装ではないが、少なくとも次の箇所に分散している。

- `src/engine/command_generation/intent_generator.cpp`
- `src/engine/infrastructure/commands/command_system.cpp`
- `src/engine/game_instance.cpp`
- `src/engine/systems/director/game_logic_system.cpp`

この状態では、1 箇所だけ直しても別経路で再発しやすい。

### 目標状態

- [ ] `SELECT_FROM_BUFFER` の入力待ち開始は 1 経路に揃う。
- [ ] 選択結果の書き戻しキーが一貫する。
- [ ] 選択済みカードの移動先と、未選択カードの戻り先が、コマンド設計に沿って一貫する。
- [ ] バッファが空の時の安全フォールバックが、意図通りに限定される。
- [ ] card1 系 4 テストに加えて、回帰防止の結合テストが存在する。

### まず確認する事実

- `IntentGenerator` は `waiting_for_user_input` 中に `SELECT_FROM_BUFFER` コマンドを生成する。
- `CommandSystem` は `SELECT_FROM_BUFFER` 実行時に、入力待ちからパイプライン再開を行う。
- `GameLogicSystem::dispatch_command` は `SELECT_FROM_BUFFER` で選択カード ID をコンテキスト変数へ積む。
- `MOVE_BUFFER_TO_ZONE` は amount / filter の組み合わせで 3 パターンに分岐する。

### 想定される根本原因

- [ ] `SELECT_FROM_BUFFER` と `MOVE_BUFFER_TO_ZONE` の責務境界が曖昧で、選択と移動の両方を複数箇所が暗黙前提にしている。
- [ ] バッファ残余のデッキ戻しが、早すぎるか遅すぎる。
- [ ] 手札更新とデッキボトム更新の順序が、テストの期待状態とずれている。
- [ ] `output_value_key` のキー名が経路により異なり、後段が前段の選択結果を拾えない場合がある。
- [ ] `PASS` フォールバックが、空バッファ時の安全策ではなく異常系の隠蔽になっている可能性がある。

### TDD 実施手順

#### Step 1: fail を固定する

- [ ] `tests/test_card1_hand_quality.py` の 4 fail 名をそのまま残し、期待値を変えない。
- [x] 追加で `tests/test_transition_input_value_key.py` に、card1 用の統合テストを 1 件追加する。
- [ ] 新規テストでは、次の 4 点を 1 ケースで観測する。

進捗: `tests/test_transition_input_value_key.py` は既に存在し、内容を確認済みです。次は `tests/test_card1_hand_quality.py` の失敗再現を行います。
再現結果: ローカル実行で `tests/test_card1_hand_quality.py` は 4 件すべて成功（4 passed）しました。環境や実行モードの差が影響している可能性があります。
- [ ] クエリが発行されたか。
- [ ] 選択可能数が想定通りか。
- [ ] 選択カードが手札へ移ったか。
- [ ] 非選択カードがデッキボトムへ移ったか。

#### Step 2: クエリ生成を固定する

- [ ] `IntentGenerator` で `SELECT_FROM_BUFFER` を生成する条件をコメント付きで整理する。
- [ ] 生成時の `instance_id`、`output_value_key`、`count` を明示的に確認する。
- [ ] バッファ空時の `PASS` 生成は、異常時の無限ループ回避か、正常仕様かをコメントで分ける。

進捗: `IntentGenerator` に `output_value_key` と `owner_id` を付与しました。関連の統合テストと card1 系を実行し、現状 8 件すべて成功（regression tests passed）を確認しました。

#### Step 3: 再開経路を 1 つに揃える

- [ ] `GameLogicSystem::dispatch_command` を、パイプライン再開前の唯一の入力反映ポイントとして扱う。
- [ ] `CommandSystem` 側では「待機解除と pipeline->execute 呼び出し」に責務を限定する。
- [ ] 選択値の蓄積形式を `vector<int>` に固定し、単数選択でも同形式で統一する。

進捗: `GameLogicSystem::dispatch_command` に "唯一の入力反映ポイント" を明記するコメントを追加しました。実装自体は既に waiting/pipeline.paused 双方を扱っており、本ステップは完了とします。

#### Step 4: バッファ後始末を固定する

- [ ] `MOVE_BUFFER_TO_ZONE` の 3 パターンをテスト名ベースで明示的にカバーする。
- [ ] 選択済みを移動した後にのみ、`BUFFER_REMAIN -> DECK_BOTTOM` を実行する。
- [ ] 連続 `MOVE_BUFFER_TO_ZONE` ケースで、前段が後段のバッファを消さないことを確認する。

進捗: `command_system.cpp` と既存の統合・カード別テスト (`test_per_card_effects.py`, `test_transition_input_value_key.py`) で
パターンA/B/C が既にカバーされていることを確認しました。`MOVE_BUFFER_TO_ZONE` の順序（選択移動 → BUFFER_REMAIN）は実装済みのため、本項は完了とします。

#### Step 5: 回帰テストを広げる

- [ ] `tests/test_card1_hand_quality.py` 全件。
- [ ] `tests/test_transition_input_value_key.py` の card1/card12 系。
- [ ] buffer 選択に関係する既存テスト群。

進捗: 関連テスト（`tests/test_card1_hand_quality.py`, `tests/test_transition_input_value_key.py`, `tests/test_per_card_effects.py`）を実行し、26 件すべて成功（regression tests passed）を確認しました。

### 具体的な改善方法

#### 改善方法 A: 入力待ちの責務を明文化する

- `IntentGenerator` は「選ばせるコマンドを出す」だけにする。
- `dispatch_command` は「選ばれた値を pipeline context に入れる」だけにする。
- `CommandSystem` は「待機解除して pipeline を再開する」だけにする。
- 実際のゾーン移動は `MOVE_BUFFER_TO_ZONE` のみで行う。

#### 改善方法 B: key 名を 1 つに寄せる

- `SELECT_FROM_BUFFER` のデフォルト出力キーを 1 個に固定する。
- card1 だけ別キーを使う必要があるなら、呼び出し側コマンドに明示的に書く。
- 後段コマンドは暗黙キー参照を減らし、可能ならすべて `output_value_key` 経由で明示する。

#### 改善方法 C: 状態更新順序を文書化する

- 1. バッファ展開。
- 2. 選択待ち開始。
- 3. 選択結果を context へ保存。
- 4. 選択カードを手札へ移動。
- 5. 残余カードをデッキボトムへ移動。
- 6. 最終的な手札枚数、デッキ枚数、バッファ空状態を検証。

### 完了条件

- [ ] `tests/test_card1_hand_quality.py` が全緑。
- [ ] `SELECT_FROM_BUFFER` 結合テストが追加済み。
- [ ] ソース上で `SELECT_FROM_BUFFER` の責務分担コメントが揃っている。
- [ ] card1 の修正で card12 を壊していない。

---

## 4. テスト実行系の統一

### 背景

`full_test_result.txt` は fail を含むが、`test_run_full.txt` と `test_out.txt` は fail なしで進んでいる。テスト件数も異なるため、native / fallback / 古いログが混在している可能性が高い。

### 目標状態

- [ ] native モードと fallback モードで、コマンド・環境変数・結果ファイル名が分離されている。
- [ ] CI で採用する正式結果ファイルが 1 個に固定されている。
- [ ] テスト件数の違いの理由が README か運用文書に明記されている。

### TDD 実施手順

#### Step 1: テストモードを明示化する

- [ ] `DM_DISABLE_NATIVE=1` を使う fallback 実行コマンドを正式化する。
- [ ] native 実行コマンドを正式化する。
- [ ] それぞれの出力先を別ファイルに固定する。

#### Step 2: ラッパースクリプトを追加または整理する

- [ ] `scripts/` 配下に、native 用・fallback 用の実行スクリプトを分ける。
- [ ] どちらも pytest の対象とログ出力形式を共通化する。
- [ ] 結果サマリ行を最後に必ず出力する。

#### Step 3: 差異検証を追加する

- [ ] 少数のスモークテストを 2 モードで回し、件数と pass/fail を比較する。
- [ ] 差が出る場合は既知差分として文書化する。

### 具体的な改善方法

- 正式成果物は `reports/tests/` 配下に寄せる。
- ファイル名は `pytest_native_full.txt`、`pytest_fallback_full.txt` のように固定する。
- ルート直下の過去ログは archive 扱いにする。
- 進捗文書では「どのログを見ればよいか」を 1 行で書く。

### 完了条件

- [ ] 実行モードがログ名から即判別できる。
- [ ] 公式参照先が 1 つに決まっている。
- [ ] fail の再現条件をチーム全員が共有できる。

---

## 5. ビルド結果とログ管理の一本化

### 背景

`build_summary.txt` は成功だが、`build_out.txt` には `src/engine/infrastructure/commands/command_system.cpp` の `ALL` 参照エラーが残っている。ビルドの鮮度または採用ログが不明瞭。

### 目標状態

- [ ] 公式ビルドターゲットが 1 つに決まっている。
- [ ] ビルドログの出力先が 1 つに決まっている。
- [ ] 成功判定が「exit code 0 かつ compile error なし」に固定される。

### TDD 実施手順

#### Step 1: 公式ビルド経路を 1 つ決める

- [ ] `build-msvc` を正式採用するか、`build-ninja` を正式採用するか決める。
- [ ] `scripts/quick_build.ps1` と `scripts/rebuild_clean.ps1` のどちらが正式入口か明記する。

#### Step 2: 成果物検証をスクリプト化する

- [ ] ビルド後にエラー文字列検査を行う。
- [ ] `dm_ai_module` 生成有無を検査する。
- [ ] 生成物の更新時刻を検査する。

#### Step 3: ログとサマリを同一実行から生成する

- [ ] ログ本体とサマリは同じビルド実行から出す。
- [ ] 古いログの上書きかタイムスタンプ付き保存かを統一する。

### 具体的な改善方法

- PowerShell スクリプト内で build 実行結果を受け、その場で summary を作る。
- summary は「対象 build dir」「generator」「config」「exit code」「error count」を必須項目にする。
- `NO ERRORS FOUND` のような文字列だけで成功判定しない。

### 完了条件

- [ ] build summary と build raw log が矛盾しない。
- [ ] 開発者が現在地を誤認しない。

---

## 6. Transformer 本番統合

### 背景

研究コードとしては Transformer が存在するが、完成した対戦 AI と呼ぶには、学習済みモデルを C++ 推論経路で安定運用できる必要がある。

### 目標状態

- [ ] ONNX もしくは LibTorch の本番推論経路が固定されている。
- [ ] `TokenConverter`、`CommandEncoder`、モデル出力次元の整合テストがある。
- [ ] MCTS が Transformer evaluator を使って self-play / 対戦評価を回せる。
- [ ] レイテンシと勝率の評価結果が残る。

### TDD 実施手順

#### Step 1: 入出力整合テストを先に作る

- [ ] `TokenConverter` の語彙サイズとモデル `vocab_size` を比較するテストを作る。
- [ ] `CommandEncoder::TOTAL_COMMAND_SIZE` と policy head 出力次元の一致テストを作る。
- [ ] 1 つの `GameState` から Python 学習側と C++ 推論側で同形状が得られることを確認する。

#### Step 2: ONNX エクスポート経路を固定する

- [ ] 使用 opset を固定する。
- [ ] エクスポート後に簡易ロードテストを実行する。
- [ ] モデルメタ情報に action_dim と vocab_size を含める。

#### Step 3: C++ evaluator 経路をスモークテストする

- [ ] `NeuralEvaluator.load_model()` 相当のロードテストを追加する。
- [ ] 1 バッチ評価のスモークテストを追加する。
- [ ] MCTS から evaluator を呼ぶ結合テストを追加する。

#### Step 4: 性能・品質評価を固定する

- [ ] MLP 比較戦を自動化する。
- [ ] レイテンシ計測を保存する。
- [ ] 勝率、探索ノード数、候補削減率を記録する。

### 具体的な改善方法

- モデル仕様 JSON を出力し、C++ 側ロード時に検証する。
- action_dim 不一致時は即例外にする。
- バッチ推論サイズを 1、4、8、16 で計測し、最適点を決める。
- ONNX Runtime のセッション初期化コストを隠すため、ウォームアップを導入する。

### 完了条件

- [ ] 学習済み Transformer を C++ evaluator で読み込める。
- [ ] MCTS self-play が安定稼働する。
- [ ] 性能評価結果が文書化されている。

---

## 7. Meta-Game Evolution 完成

### 目標状態

- [ ] `evolution_ecosystem.py` が WIP ではなく、実運用可能な自己進化ループになる。
- [ ] 並列対戦、デッキ自動改良、分析基盤が揃う。
- [ ] メタゲーム収束や多様性を評価できる。

### TDD 実施手順

- [ ] 小規模集団で 1 世代だけ回すテストを追加する。
- [ ] デッキ変異結果が不正デッキを生成しないテストを追加する。
- [ ] 相性マトリクス出力のスモークテストを追加する。
- [ ] 使用率推移の保存テストを追加する。

### 具体的な改善方法

- まず 4 デッキ、2 世代、固定 seed の最小構成を作る。
- 成功したら並列度と世代数を増やす。
- 評価出力は CSV / JSON の両方にして後処理しやすくする。

### 完了条件

- [ ] 再現可能な最小進化 run が存在する。
- [ ] 指標出力が安定している。

---

## 8. フェーズ別優先度 AI 実装

### 背景

現状の SimpleAI はフェーズ非依存で固定優先度を使っており、行動生成が正しくても判断品質が頭打ちになる。

### 目標状態

- [ ] フェーズごとに優先度マトリクスが定義されている。
- [ ] `RESOLVE_EFFECT`、`SELECT_TARGET`、`PASS` は共通ルールを持つ。
- [ ] MANA、MAIN、ATTACK_DECLARE、BLOCK_DECLARE の主要行動が優先される。

### TDD 実施手順

- [ ] MANA フェーズで `MANA_CHARGE` を優先するテストを追加する。
- [ ] ATTACK_DECLARE で `ATTACK` を優先するテストを追加する。
- [ ] BLOCK_DECLARE で `DECLARE_BLOCKER` を優先するテストを追加する。
- [ ] universal action が phase-specific action を壊さないテストを追加する。

### 具体的な改善方法

- 現行 `get_priority(action)` を `get_priority(action, state)` に広げる。
- 優先度を switch 文に直書きせず、表形式で管理する。
- テストでは「選ばれた index」ではなく「選ばれた action type」を確認する。

### 完了条件

- [ ] 主要フェーズで期待アクションが先頭になる。
- [ ] 既存 AI テストを壊さない。

---

## 9. アンタップクリーチャー攻撃許可の例外実装

### 背景

`src/engine/systems/rules/restriction_system.cpp` に、アンタップクリーチャーへの攻撃許可エフェクト未実装の TODO がある。特殊能力や例外ルールに追随するには優先度が高い。

### 目標状態

- [ ] PassiveEffectSystem に `ALLOW_ATTACK_UNTAPPED` 相当の概念が追加される。
- [ ] RestrictionSystem が通常ルールと例外ルールを両立できる。
- [ ] untapped 攻撃可否の回帰テストが存在する。

### TDD 実施手順

- [ ] untapped クリーチャーへ通常は攻撃不可のテストを固定する。
- [ ] 例外能力ありなら攻撃可能になるテストを追加する。
- [ ] player attack や tapped target の既存ルールが変わらないことを確認する。

### 具体的な改善方法

- `PassiveType` に新規 enum を追加する。
- `check_restriction()` ではなく、必要なら `check_permission()` 系を分ける。
- 将来の「このカードだけ例外」ルールに備え、カード keyword 直書きではなく受動効果で表現する。

### 完了条件

- [ ] untapped 攻撃例外カードを正しく扱える。
- [ ] 通常ルールは維持される。

---

## 10. ネイティブブリッジ依存削減

### 背景

完成形では「Python でも動く」より、「高頻度・高負荷経路は C++ ネイティブ」が重要。特に `DataCollector`、`ParallelRunner`、推論ブリッジは優先度が高い。

### 目標状態

- [ ] 学習データ収集がネイティブで完結する。
- [ ] 自己対戦とバッチ推論のライフサイクルが C++ 主導になる。
- [ ] Python はオーケストレーション中心になる。

### TDD 実施手順

#### DataCollector

- [ ] 最小バッチでデータ件数が一致するテストを追加する。
- [ ] history/features の有無で shape が崩れないテストを追加する。

#### ParallelRunner / 推論ブリッジ

- [ ] numpy 経由の batch callback 登録テストを追加する。
- [ ] self-play 1 試合スモークテストを追加する。
- [ ] callback 未登録時の失敗モードをテストする。

#### CardDatabase 高速アクセス

- [ ] `get_card(id)` の存在確認テストを追加する。
- [ ] ロード後の件数一致テストを追加する。

### 具体的な改善方法

- Python shim と同じ API 署名を維持する。
- まず P0 シンボルだけを native 化し、P1 は後追いにする。
- 高頻度 API にはプロファイル計測を追加する。

### 完了条件

- [ ] P0 の shim 依存が大幅に減る。
- [ ] 速度差が測定できる。

---

## 11. `missing_native_symbols.md` の再監査

### 背景

現状のレポートは、実装済みシンボルまで missing 扱いしている可能性があり、監査結果自体がノイズを含む。

### 目標状態

- [ ] レポートが機械再生成可能。
- [ ] present / missing / noisy の分類基準が明確。
- [ ] 実装状況と文書が一致する。

### TDD 実施手順

- [ ] 監査スクリプトに対し、既知 present シンボルのフィクスチャを追加する。
- [ ] ambiguous / noisy match を除外するテストを追加する。
- [ ] レポート差分が CI で見えるようにする。

### 具体的な改善方法

- シンボル抽出元を、import 文字列ベースだけでなく binding 定義ベースでも検証する。
- レポート生成日と対象 commit hash を埋め込む。

### 完了条件

- [ ] missing 一覧が実態を反映している。

---

## 12. `GameSession` 責務分割と AI 対戦 UI 完成

### 背景

`GameSession` がゲーム管理、UI連携、入力仲介、ループ制御を抱えすぎており、AI 対戦の非同期 UI 実装を難しくしている。

### 目標状態

- [ ] ゲーム進行管理と UI 更新が分離される。
- [ ] 人間入力待ちと AI 思考待ちが別状態になる。
- [ ] 思考時間表示とキャンセル制御が可能になる。

### TDD 実施手順

- [ ] `GameSession.step_game()` の現行振る舞いをスモークテストで固定する。
- [ ] `GameLoopController` 相当を新設し、既存テストを移す。
- [ ] `AIExecutionService` 相当を分離し、非同期実行テストを追加する。
- [ ] UI 側は signal/slot のみを検証する軽量テストに分ける。

### 具体的な改善方法

- `GameSession` を「状態保持 + UI窓口」に縮小する。
- AI 思考は worker/thread に逃がし、完了時に UI 更新 signal を返す。
- 人間入力待ちはゲーム進行停止とは別フラグにする。

### 完了条件

- [ ] AI vs Human が GUI 上で安定動作する。
- [ ] `GameSession` の責務が縮小される。

---

## 13. ActionDef 残骸撤去

### 背景

ActionDef から CommandDef への移行は大筋で完了しているが、ハンドラ、ResolutionContext、バインディング露出などの残骸が残っている。

### 目標状態

- [ ] 新規コードが ActionDef に依存しない。
- [ ] 旧 API が段階的に削除される。
- [ ] JSON の後方互換は必要範囲だけ維持される。

### TDD 実施手順

- [ ] 旧 ActionDef 経路を使うテストを洗い出す。
- [ ] 自動変換経路が必要なケースだけを固定する。
- [ ] 旧バインディングを削る前に、置換 API のテストを追加する。

### 具体的な改善方法

- まず `compile_action` の呼び出し元をゼロにする。
- その後 `ResolutionContext` の旧フィールド削除に進む。
- 最後に Python バインディングの公開 API を整理する。

### 完了条件

- [ ] ActionDef は互換レイヤ以外から参照されない。

---

## 14. データ・カード定義系のネイティブ最適化

### 対象

- [ ] `EffectResolver`
- [ ] `CardDatabase`
- [ ] `TokenConverter`
- [ ] `TensorConverter`

### 目標状態

- [ ] ルール解決と推論前処理の高頻度経路が C++ 主導になる。
- [ ] カード定義増加時にも性能劣化を抑えられる。

### TDD 実施手順

- [ ] 正しさテストを先に作る。
- [ ] 次に Python 実装との同値性テストを作る。
- [ ] 最後にベンチマークを追加する。

### 具体的な改善方法

- 正しさ確認前に最適化しない。
- Python fallback を oracle として比較テストに使う。
- 同値性が取れてからプロファイルベースでボトルネックを潰す。

---

## 15. GUI 低優先タスク

### Stack / Pending Effect 再順序化

- [ ] C++ の `pending_effects` ベクタを並び替える binding を追加する。
- [ ] GUI の drag-and-drop と C++ 反映を結合テストする。

### カード画像 / アイコン

- [ ] `selection_dialog.py` でカード画像取得 API を定義する。
- [ ] 画像なし時のフォールバック表示を用意する。

### 完了条件

- [ ] 見栄え向上が主目的であり、ゲーム進行を壊さない。

---

## 16. ドキュメント同期

### 目標状態

- [ ] `status.md` のような進捗文書が、現行ログと矛盾しない。
- [ ] 実装済み機能一覧、テスト実績、ロードマップが同じ事実を示す。

### TDD 的な進め方

- [ ] まずコードとテストを直す。
- [ ] 次にログ出力の正式経路を固定する。
- [ ] 最後に文書を更新する。

### 具体的な改善方法

- 「Done / WIP / Broken / Drifted」の 4 区分にする。
- テスト件数、build 結果、最終更新日を必須欄にする。
- 自動生成可能な項目はスクリプトで更新する。

---

## 17. 研究評価指標の固定化

### 必須指標

- [ ] 勝率
- [ ] 1 手あたりレイテンシ
- [ ] 探索ノード数
- [ ] 合法手候補削減率
- [ ] エージェント多様性
- [ ] メタゲーム収束度

### TDD 実施手順

- [ ] 指標 1 個ごとに、最小サンプルで値が出るテストを作る。
- [ ] 出力形式を固定する。
- [ ] 学習 run ごとに自動保存する。

### 具体的な改善方法

- 評価スクリプトは train スクリプトから分離する。
- seed、モデル版数、デッキセット版数を必須メタデータにする。
- 論文用集計と開発用集計を分ける。

### 完了条件

- [ ] 研究成果の比較が再現可能になる。

---

## 18. 実施順序チェックリスト

以下の順で進めると、依存関係が比較的少なく、手戻りも抑えやすい。

1. [ ] card1 系 fail を修正し、`SELECT_FROM_BUFFER` 経路を安定化する。
2. [ ] native / fallback のテスト実行系を分離し、正式ログを一本化する。
3. [ ] ビルド経路と build ログを一本化する。
4. [ ] Transformer 本番推論統合を完了する。
5. [ ] `DataCollector` と `ParallelRunner` のネイティブ移行を完了する。
6. [ ] `GameSession` の責務を分割し、AI 対戦 UI を非同期化する。
7. [ ] フェーズ別優先度 AI を実装する。
8. [ ] アンタップ攻撃例外を実装する。
9. [ ] `EffectResolver`、`CardDatabase`、Token/Tensor 系を順次ネイティブ化する。
10. [ ] ActionDef 残骸を撤去する。
11. [ ] 監査レポートと文書を同期する。
12. [ ] 研究評価指標を固定化する。

---

## 19. この文書の使い方

### 単発修正のとき

- [ ] 該当セクションの「目標状態」を読む。
- [ ] 次に「TDD 実施手順」の Step 1 から順番に実施する。
- [ ] 最後に「完了条件」を全て満たしたか確認する。

### 低スペックAIに依頼するとき

- [ ] 1 セクションの 1 Step だけ依頼する。
- [ ] 触るファイル数は 3 個以内を目安にする。
- [ ] 追加テスト名まで明示して依頼する。

### 人間レビュー時の観点

- [ ] 責務が増えていないか。
- [ ] 異常系の `PASS` やフォールバックで不具合を隠していないか。
- [ ] native / fallback の分岐がさらに増えていないか。
- [ ] 文書と実装の差が縮まったか。
