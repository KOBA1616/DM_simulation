# カードエディタ改善・統合・データ構造再設計 実装計画

最終更新: 2026-03-15
対象: `dm_toolkit/gui/editor/` 一式
目的: 未完了タスクに集中し、小さな TDD サイクルで安全に改善を進める

---

## 0. 現状

- フェーズA〜Fの主要実装は完了済み。
- `safe_connect` 統一、保存前整合性チェック、設定SSOT化、ACTION互換整理、主要モデル型付け、CIR最小統合までは到達済み。
- 直近では `text_generator.py` の分岐削減を進め、`_format_reaction`、`original_cmd_type`、`condition.type` の一部をマップ化済み。
- 現在の主課題は、分岐削減の最終到達、条件設定UIの保守性向上、CIRの編集フロー実利用の拡張。

完了済みの詳細ログは本ファイルから削除した。必要に応じて Git 履歴を参照すること。

---

## 1. 優先度付き残タスク

### 最優先

1. `window.py` の構造更新処理の分岐削減を継続する
   - 目標: `on_structure_update()` と周辺処理を、構造コマンドごとの完全ディスパッチへ寄せる
   - 完了条件:
     - [ ] `STRUCT_CMD_* -> handler` の責務分離を完了
     - [ ] 共通後処理を 1 箇所へ固定
     - [ ] 主要構造変更の回帰テストを維持

2. 分岐削減の定量目標を更新して追跡する
   - ベースライン: 2026-03-14 時点で Python 全体の branch 合計 5096
   - 最新スキャン: 2026-03-15 実行
     - Files with branches: 309
     - Total if: 4544, elif: 565, total branches: 5109
     - Top files by branch count:
       - dm_toolkit/gui/editor/text_generator.py: 873
       - dm_toolkit/engine/compat.py: 165
       - training/head2head.py: 160
       - dm_toolkit/gui/app.py: 134
       - dm_toolkit/gui/editor/forms/unified_action_form.py: 129
   - 完了条件:
     - [x] `scripts/count_branches.py` を再実行して最新値を記録
     - [ ] 上位ファイルの削減対象を再選定
     - [ ] `window.py` と `text_generator.py` の局所改善結果を数値で残す

### 高優先

3. `schema_config.py` の重複定義を実体移管する
   - 現状: TODO コメントで停止中
   - 完了条件:
     - [ ] 重複定義の一覧を作成
     - [ ] `dm_toolkit/consts.py` または `data/configs/command_ui.json` に寄せる
     - [ ] 読み込み側の参照先を 1 系統へ整理
     - `schema_config.py` の `TARGET_SCOPES` と `DURATION_OPTIONS` を `dm_toolkit.consts` の定義に移管しました（`TargetScope` と `DURATION_TYPES` を参照）。
     - 追加移管: `MUTATION_TYPES`, `EFFECT_IDS`, `APPLY_MODIFIER_OPTIONS`, `MUTATION_KINDS_FOR_MUTATE` を `dm_toolkit.consts` 側に移動し、`schema_config.py` 側を参照に更新しました（小さな段階移管）。
    - 追加移管2: `DELAYED_EFFECT_IDS` を `dm_toolkit.consts` 側に移動しました（`REGISTER_DELAYED_EFFECT` の options に使用）。
    - 検証: `schema_config` が `DELAYED_EFFECT_IDS` を `consts` と一致して参照することを検証するユニットテストを追加しました（python/tests/unit/test_schema_consts_sot.py）。
    - テスト結果: 1 passed
     - 次工程: `window.py` の構造更新処理の分岐削減を継続しました。具体的には `_handle_add_child_effect` の if/elif チェーンをハンドラ辞書に置換し、分岐数と可読性を改善しました。
     - 検証: 変更は headless テスト群に影響を与えていないことを確認済み（該当ユニットテスト実行で回帰無し）。
     - 動作確認: 関連ユニットテストを実行し回帰なし（複数テスト合格）。

4. CIR を編集フローで実利用できる形に進める
   - 現状: 読み込み・保持・一部UI反映までは完了
   - 完了条件:
     - [ ] `Apply CIR` の適用範囲を複合フィールドまで拡張
     - [ ] CIR と現在フォーム値の差分表示を常設UIにする
     - [ ] 永続化責務を serializer 側に固定する

### 中優先

5. `CommandModel.params` の型付け対象を追加する
   - 現状: `QueryParams` / `TransitionParams` / `ModifierParams` は導入済み
   - 完了条件:
     - [ ] 高頻度コマンドの `params: Any` 残件を洗い出す
     - [ ] 追加の型モデルを導入する
     - [ ] 型付きモデルの serialize / deserialize テストを追加する
    - 進捗: 2026-03-15
      - `CommandModel` の `params` に対して `SearchParams` を追加し、`SEARCH_DECK` コマンドを型化しました。
      - 単体テスト `python/tests/unit/test_commandmodel_searchparams.py` を追加し、`SEARCH_DECK` の ingest/serialize が型付きで動作することを検証（1 passed）。
      - 次: 他の高頻度コマンド（`LOOK_AND_ADD`, `MEKRAID`, `ADD_KEYWORD` など）について同様の型導入を進める予定です。

---

## 2. カードエディタ条件設定実装の改善点

### 2.1 条件モデルをUI都合から切り離す

問題:
- 条件設定がフォーム部品の可視性と保存形式に強く引っ張られている
- `condition.type` ごとの表示切替と保存分岐が複数箇所に散っている

改善案:
- `ConditionModel` を導入し、最低限以下を明示する
  - `type`
  - 比較演算子
  - 比較対象値
  - 参照元キー
  - 対象スコープ
- UI は `ConditionModel` を編集するだけにし、保存形式への変換は serializer / transform 層に閉じ込める

期待効果:
- 条件種別追加時の変更箇所を減らせる
- バリデーションを UI から分離できる

### 2.2 条件入力UIを宣言的にする

問題:
- `condition.type` ごとに `if/elif` でウィジェットの表示・非表示を切り替えている箇所が増えやすい

改善案:
- `CONDITION_FORM_SCHEMA` のような辞書を作り、各条件タイプごとに必要フィールドを宣言する
- 例:
  - `OPPONENT_DRAW_COUNT -> [value]`
  - `COMPARE_STAT -> [stat_key, op, value]`
  - `COMPARE_INPUT -> [input_value_key, op, value]`
  - `MANA_CIVILIZATION_COUNT -> [op, value]`
- `widget_factory` または専用 builder が schema を解釈して入力欄を生成する

期待効果:
- 条件追加時に UI 分岐を書き足さずに済む
- テストは schema の検証に寄せられる

- 実装メモ: `CONDITION_UI_CONFIG` を用いた宣言的切替を既存ウィジェットに導入済み。小さなカバレッジテストを追加してスキーマ整合性を検証（python/tests/unit/test_condition_ui_schema.py）。

### 2.3 条件プレビューを常時表示する

問題:
- ユーザーが設定した条件が最終的にどんな日本語テキストになるかを、保存やプレビューまで把握しにくい

改善案:
- 条件編集フォームに「条件プレビュー」欄を追加する
- `text_generator.py` の条件フォーマッタと同じロジック、またはその薄いラッパを使って即時表示する

期待効果:
- 誤設定の早期発見
- UI とテキスト生成の乖離検出

### 2.4 条件バリデーションを入力時点で返す

問題:
- 現状は保存時バリデーション中心で、条件設定の誤りに気付きにくい

改善案:
- 条件フォーム単位で `validate_condition_model()` を実行する
- 不足項目、演算子不整合、型不整合をその場で表示する
- 保存前バリデーションは最終ゲートとして残す

期待効果:
- 保存前エラーの集中を防げる
- 入力フローが軽くなる

進捗: 2026-03-15
- 小タスク実施: `ConditionEditorWidget.validate_condition_model()` を追加し、条件データの即時検証を可能にしました。
- 追加テスト: `python/tests/unit/test_condition_validation.py` を追加し、ヘッドレス環境での検証を確認（3 passed）。
- 備考: まずは必須フィールドの有無を検出する軽量実装とし、将来的に型チェック・相互矛盾チェックを拡張予定。

### 2.5 trigger 条件と target_filter 条件を分離する

問題:
- trigger の発火条件と対象選択条件が似た構造で混ざりやすい
- `condition`, `trigger_filter`, `target_filter` の責務境界が曖昧な箇所がある

改善案:
- 役割を明文化する
  - `condition`: 発火可否の条件
  - `trigger_filter`: 発火イベント側の対象条件
  - `target_filter`: 効果適用先の対象条件
- フォーム上でも編集セクションを分ける

期待効果:
- 保存JSONの意味が明確になる
- テキスト生成の誤解釈を減らせる

### 2.6 条件テンプレートを用意する

問題:
- よく使う条件が毎回手入力になり、表記揺れや設定漏れが起きる

改善案:
- 条件テンプレート候補を追加する
  - 自分のシールド枚数条件
  - 相手ドロー枚数条件
  - 自分ターン中 / 相手ターン中
  - 文明数条件
  - 入力値比較条件

期待効果:
- 入力速度向上
- 条件表現の標準化

---

## 3. GUI改善案

1. `UnifiedActionForm` に差分パネルを常設する
   - CIR
   - 現在フォーム値
   - 保存予定値

2. 構造変更後の後処理をポリシー化する
   - 選択同期
   - プレビュー更新
   - dirty state 更新
   - ログ反映

3. 条件設定フォームの見出しを役割別に分ける
   - 発火条件
   - イベント対象
   - 効果対象
   - 比較入力

4. 条件ごとの入力不足を即時ハイライトする

---

## 4. データ構造改善案

1. `ConditionModel` を導入し、条件設定の保存形式を一段正規化する
2. `FilterModel` の UI入力形式と永続化形式の境界を serializer に固定する
3. `CommandModel.params` の追加型付け対象を計測ベースで増やす
4. `text_generator.py` のフォーマッタは「辞書駆動 + 小関数群」に統一する

---

## 5. 次の実装候補

候補A: `window.py` の構造変更後処理をポリシー化する

候補B: 条件設定フォームに schema 駆動の表示切替を導入する

候補C: `text_generator.py` の残り条件フォーマッタを追加マップ化する
   - `MUTATE` 処理の分岐をハンドラ辞書化してマップ駆動に変更しました（`_format_special_effect_command` 内）。
   - 併せてユニットテスト `python/tests/unit/test_text_generator_mutate_map.py` を追加し、基本ケースを検証するようにしました。

---

   - 次工程: `window.py` の構造更新処理の分岐削減を継続しました。具体的には `_handle_add_child_effect` の if/elif チェーンをハンドラ辞書に置換し、分岐数と可読性を改善しました。
   - 検証: 変更は headless テスト群に影響を与えていないことを確認済み（該当ユニットテスト実行で回帰無し）。
## 6. 実行ルール

- 1回の実装は 1タスク・1症状・1〜3ファイル変更を原則とする
- 必ず `RED -> GREEN -> REFACTOR` で進める
- 実装後は関係する最小テストだけを先に回す
- 大量の完了ログはこのファイルに戻さず、必要なら別レポートか Git 履歴へ残す

---

## 7. フィルタ実装の再点検と改善点（追記）

### 7.1 フィルタ定義の一元化を最優先で進める

問題:
- `target_filter` / `trigger_filter` / static `filter` が実質同じ概念なのに、検証・表示・保存の入口が分散している
- 空フィルタ文言やセクション表示キーの不整合が発生している

改善案:
- `FilterSpec` を単一スキーマとして定義し、用途は `context` で切り替える
  - `context=TARGET`
  - `context=TRIGGER`
  - `context=STATIC`
- `UnifiedFilterHandler` は `FilterSpec` の adapter のみ担当させる
- `filter_widget` の表示制御キーを enum 化し、不正キー入力時は警告ログを出す

期待効果:
- フィルタ関連の実装/検証/文言生成のズレを抑止できる
- 画面表示と保存データの責務境界が明確になる

### 7.2 フィルタ値型の統一（bool/int混在解消）

問題:
- `is_tapped` などのフラグが UI では bool、validator では int(0/1) 前提になっている

改善案:
- フィルタフラグは bool に統一する
- 既存データ読込時のみ互換変換を行う（0/1 -> bool）
- validator 側は bool を正とし、int は互換警告に落とす

期待効果:
- 入出力の型不整合による誤検知を防げる
- mypy 対応とテスト記述が簡潔になる

### 7.3 フィルタ文言生成の単一路線化

問題:
- `text_generator` と `unified_filter_handler` 側で近い責務が重複し、文言ゆれが起きやすい

改善案:
- `describe_filter(filter_spec, usage)` を単一 API として切り出す
- 画面要約、トリガー説明、本文生成は同 API を使う

期待効果:
- 文言差分の再発防止
- 文言変更時の影響範囲を限定できる

---

## 8. コマンドグループ/コマンド種別の再確認と改善点（追記）

### 8.1 コマンド定義SSOTの再構成

問題:
- コマンド種別の定義が複数箇所に分散し、集合差分が恒常化している
  - `command_ui.json`
  - `schema_config.py`
  - `card_validator.py`
  - `consts.py` fallback

改善案:
- `CommandRegistry`（SSOT）を新設し、以下を同居させる
  - `type`
  - `group`
  - `fields`
  - `visibility`
  - `validator rule`
  - `text generation hint`
- 既存ファイルは registry から生成/参照する薄い層に置換する

期待効果:
- 定義ズレの根本原因を除去できる
- 新コマンド追加時の変更漏れを防げる

### 8.2 CIでの差分検出を必須化

問題:
- 定義差分がレビュー時に見落とされやすい

改善案:
- CI に「コマンド集合一致チェック」を追加する
  - UI設定 vs schema
  - schema vs validator
  - validator vs consts fallback

期待効果:
- 差分の早期検出
- 仕様変更の追従漏れを自動で止められる

### 8.3 旧形式定義ファイルの段階的縮退

問題:
- `command_schema.json` の内容が現行実装と乖離し、混乱源になっている

改善案:
- `command_schema.json` を「互換用/参照専用」に明示するか、registry 生成物に置換する
- 参照コードを段階的に削除して、読むべき定義を一本化する

期待効果:
- 開発者の認知負荷低減
- 誤参照による修正ミスを防げる

---

## 9. 効果タイプ実装の見直し（追記）

### 9.1 用語と責務の分離

問題:
- 「効果タイプ」が Editor 概念・JSON概念・C++ pending 実行概念で混線している

改善案:
- 用語を3軸に分離して明示する
  - `AbilityKind`（TRIGGERED / STATIC / REPLACEMENT / REACTION）
  - `TriggerType`（ON_PLAY など）
  - `PendingEffectType`（C++実行制御）
- 変換は `effect_normalizer` 1箇所に集約する

期待効果:
- 仕様議論と実装の対応関係が明確になる
- 変換漏れによるバグを追跡しやすくなる

### 9.2 TriggerType の世代混在を解消

問題:
- `ON_*` 系と `AT_*` 系の旧名が validator / logic mask に残り、判定が不安定

改善案:
- 正式 TriggerType を `consts.py` と C++ enum に固定する
- 旧名は読み込み時 alias 変換のみ許可し、保存時には正規化する

期待効果:
- トリガー候補表示とバリデーションの不一致を解消
- データの世代差を吸収しつつ新規作成の一貫性を保てる

### 9.3 C++ pending 分岐のテーブル駆動化

問題:
- `pending_strategy.cpp` の分岐集中で追加時リスクが高い

改善案:
- `EffectType -> handler` の dispatch table を導入する
- 共通前処理（優先度・controller判定）と個別処理を分離する

期待効果:
- 可読性向上
- 効果タイプ追加時の回帰リスク低減

---

## 10. フィルタ統合案（実装ロードマップ追記）

1. Step 1: `FilterSpec` と serializer を導入する
   - 完了条件:
     - [ ] `FilterSpec`（型定義）追加
     - [ ] 旧dict <-> FilterSpec 変換 adapter を実装

2. Step 2: UI層を `FilterSpec` に切替える
   - 完了条件:
     - [ ] `filter_widget` の `get_data/set_data` を FilterSpec 前提に更新
     - [ ] `UnifiedFilterHandler` の不正表示キーを排除

3. Step 3: 検証・文言生成を統合する
   - 完了条件:
     - [ ] `FilterValidator` を FilterSpec ベースへ移行
     - [ ] フィルタ説明APIを単一化

4. Step 4: コマンド定義と接続する
   - 完了条件:
     - [ ] `CommandRegistry` から filter field 制約を参照
     - [ ] コマンドごとの allowed filter fields を統一ルールで適用

5. Step 5: 回帰テストと差分検査を固定化する
   - 完了条件:
     - [ ] フィルタ統合テスト（UI->保存->再読込）を追加
     - [ ] コマンド定義差分テストを CI に追加
     - [ ] TriggerType 正規化テストを追加
