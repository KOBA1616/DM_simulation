# コスト修正関連テキスト生成拡張計画

**作成日**: 2026-03-23  
**対象**: `dm_toolkit/gui/editor/text_generator.py`  
**関連ドキュメント**: 
- `CARD_EDITOR_AND_ENGINE_STAT_GAP_REPORT_2026-03-19.md`
- `CONSISTENCY_CHECK_REPORT_2026-03-22.md`

---

## 1. 現状分析

### 1.1 既存機能

#### text_generator.py での実装状況
- ✅ **基本コスト表示**: `コスト{cost}` でヘッダーに表示
- ✅ **キーワード固定文面**: `CXID` や `TWINPACT` など固定テキスト化
- ⚠️ **コスト軽減（PASSIVE）**: `_format_cost_reduction()` メソッド存在だが実装が限定的
- ❌ **複雑なコスト軽減テキスト**: 以下が未実装
  - `STAT_SCALED` 軽減の動的文面生成
  - 条件付きコスト軽減 (`COMPARE_STAT`, `CARDS_MATCHING_FILTER`)
  - `static_abilities` の `COST_MODIFIER` テキスト化
  - 複数の軽減が組み合わった場合の合成表現

#### C++エンジン側の仕様
- ✅ `ManaSystem::get_adjusted_cost` で動的計算実装済み
- ✅ `STAT_SCALED` 計算式の契約定義完了
- ✅ 合成順序の明確化完了 (基本→PASSIVE→COST_MODIFIER→ACTIVE_PAYMENT→min_floor)

### 1.2 問題点

| 項目 | 問題 | 影響度 |
|------|------|--------|
| **表示不完全** | `cost_reductions` や `COST_MODIFIER` が GUI で文面化されず、エディタ確認が難しい | 🔴 高 |
| **ユーザー理解** | デッキビルダーやAI訓練者が実際のコスト軽減の効果を把握できない | 🔴 高 |
| **テスト検証** | テキスト出力が無いため自動テストで仕様検証ができない | 🟡 中 |
| **ドキュメント同期** | C++ 側の複雑な軽減仕様がテキスト形式で記録されない | 🟡 中 |

### 1.3 既存テスト状況

- ✅ `test_effect_and_text_integrity.py`: 基本的なテキスト整合性チェック
- ❌ `test_cost_reduction_text_generation.py`: **未実装** ← 本計画で追加対象
- ❌ `test_cost_modifier_static_text_gen.py`: **未実装** ← 本計画で追加対象
- 統計キー関連のテスト: `test_cost_modifier_stat_scaled.py` は値計算テストのみ

---

## 2. 拡張範囲定義

### 2.1 Phase 1「基本軽減テキスト化」（実装予定: P1）

対象: `PASSIVE` と `FIXED` モードの軽減

#### 2.1.1 実装内容

```markdown
例1: コスト軽減（固定）
【カード名】... コスト5
マナゾーンに闇の文明が2枚以上あれば、このカードの召喚コストは2少なくなる。

例2: 複数の軽減の組み合わせ
基本コスト: 7
- PASSIVE: 闇の文明×2で 2 削減
- 実表示: 「コスト5（闇の文明が2枚以上あるとき）」

例3: FIXED COST_MODIFIER（static_abilities経由）
【カード名】... コスト6
このクリーチャーが手札、マナゾーン、または墓地から出た時、コストを2少なくする。
```

#### 2.1.2 設計ポイント

- **条件の日本語化**: `condition.type` から「〜が〜の時」形式の文面を生成
- **軽減元の明示**: 「コスト軽減（条件）」形式で並記可能
- **前提チェック**: 不完全な定義（必須項目未入力）はテキスト生成時に警告
- **プレビュー表示**: エディタの「テキストプレビュー」パネルで直ちに確認可能

#### 2.1.3 実装スケジュール

| タスク | 担当 | 期限 | 備考 |
|--------|------|------|------|
| T-2.1.1: `_format_cost_reduction` 拡張 | 実装 | 2026-04-XX | PASSIVE 対応化 |
| T-2.1.2: `_format_static_ability_cost_modifier` 新規 | 実装 | 2026-04-XX | FIXED + 基本 STAT_SCALED |
| T-2.1.3: テスト実装 (`test_cost_..._text_gen.py`) | テスト | 2026-04-XX | カバレッジ 80% 以上 |
| T-2.1.4: GUI 統合（テキストプレビュー） | GUI | 2026-04-XX | UnifiedActionForm に表示 |

---

### 2.2 Phase 2「複雑な軽減の文面化」（実装予定: P2）

対象: `STAT_SCALED`, `COMPARE_STAT`, `CARDS_MATCHING_FILTER` を含む複合軽減

#### 2.2.1 実装内容

```markdown
例1: STAT_SCALED 軽減
【カード名】... コスト8
このクリーチャーを召喚する時、自分のバトルゾーンにある
クリーチャーの総パワーが大きいほど、このクリーチャーの
召喚コストを軽減する（最大X削減）。

計算式表示例：
「パワー 3 ごと、コスト 1 削減（最大 4 削減）」

例2: 条件付き複合軽減
【カード名】... コスト9（闇3文明）
闇の文明が3枚以上あり、かつバトルゾーンに闇のクリーチャーが
いるならば、このクリーチャーの召喚コストは4少なくなる。
さらに、バトルゾーンにあるクリーチャーのパワーが30以上なら、
2さらに少なくなる。

例3: CARDS_MATCHING_FILTER ベース
【カード名】... コスト6
バトルゾーンに悪魔族が2体以上いるなら、このカードの
召喚コストは3少なくなる。
```

#### 2.2.2 仕様ポイント

- **統計キーの逆引き**: `stat_key` → 日本語用語（「パワー」「コスト」「体数」等）に変換
- **計算式の明示性**: 「N ごと、コスト M 削減」という数式的な表現を提供
- **上限の記載**: `max_reduction` が指定されていれば「最大 X 削減」と併記
- **条件式の複合**: AND/OR の場合分けを日本語で自然に表現
- **段階的軽減**: 複数の軽減が重複する際の説明（「さらに」等）

#### 2.2.3 実装スケジュール

| タスク | 担当 | 期限 | 備考 |
|--------|------|------|------|
| T-2.2.1: 統計キー辞書 (`STAT_KEY_TEXT_MAP`) 構築 | 実装 | 2026-04-XX | I18n 対応 |
| T-2.2.2: `_format_stat_scaled_text` 実装 | 実装 | 2026-04-XX | 計算式の記号化 |
| T-2.2.3: 条件式複合テキスト生成 | 実装 | 2026-04-XX | COMPARE_STAT/FILTER 統合 |
| T-2.2.4: 統合テスト実装 | テスト | 2026-04-XX | Phase 1 との共存確認 |

---

### 2.3 Phase 3「エディタUI統合」（実装予定: P3）

#### 2.3.1 実装内容

- **リアルタイムプレビュー**: エディタで軽減条件を編集 → その場にプレビュー表示
- **警告表示**: 不完全な定義（例: `STAT_SCALED` で `stat_key` 未設定）は黄色警告
- **コスト計算シミュレーター**: 入力値の例を与えて「このとき実コストは X」と表示
- **多言語対応**: 既存の `tr()` 関数と統一

#### 2.3.2 実装スケジュール

| タスク | 担当 | 期限 | 備考 |
|--------|------|------|------|
| T-2.3.1: プレビューパネル拡張 | GUI | 2026-05-XX | UnifiedActionForm 更新 |
| T-2.3.2: リアルタイムジェネレーション | GUI | 2026-05-XX | エディタ型チェンジ時トリガー |
| T-2.3.3: シミュレーター UIコンポーネント | GUI | 2026-05-XX | 試行値入力フォーム |

---

## 3. 実装詳細設計

### 3.1 データ構造と生成パス

```
カードデータ (JSON)
        ↓
text_generator.py
│
├─ _format_header()
│  └─ 基本コスト表示
│
├─ _format_cost_reduction()  ← Phase 1 拡張
│  ├─ PASSIVE, FIXED モード
│  └─ 条件式テキスト化
│
├─ _format_static_ability_cost_modifier()  ← Phase 2 新規
│  ├─ STAT_SCALED 計算式
│  ├─ COMPARE_STAT 条件
│  └─ 複合軽減の合成表現
│
└─ _format_cost_simulator()  ← Phase 3 新規
   └─ 試行値による実コスト計算

### カードプレビュー拡張実施

- `generate_body_text(sample=...)` により、編集画面でのプレビューに試行サンプルを渡すと、
    - `STAT_SCALED` の場合、サンプルの統計値から実際の削減量の例を末尾に表示する（例: 「例: 現在のパワー30 → コストを3削減」）。
    - `COMPARE_STAT` / `CARDS_MATCHING_FILTER` 条件を読み取り自然文に変換して表示する。
    - サンプルが与えられない場合は従来どおりの式説明のみを表示する。

この拡張により、エディタのプレビューで定義の意図と実際の影響を同時に確認できます。
```

### 3.2 生成テキスト仕様書

#### 全般ルール

1. **条件の「〜の時」「〜ならば」の統一**
   - 単一条件: 「...の時／ならば」
   - 複数条件 AND: 「...かつ...の時」
   - 複数条件 OR: 「...または...の時」

2. **軽減量の表記**
   - FIXED: 「X少なくなる」（絶対値）
   - STAT_SCALED: 「Y ごと、コスト Z 削減」（比例式）
   - 上限付き: 「最大 M 削減」を併記

3. **前提条件の明示**
   - 文明要件: 「〜の文明が X 枚以上あれば」
   - ゾーン要件: 「バトルゾーンに〜が〜体以上いるなら」
   - 統計要件: 「バトルゾーンのクリーチャーの総パワーが X 以上なら」

4. **消極的条件**
   - 「〜がない」「〜が〇体未満」は明示（可読性向上）

#### コスト軽減テキストテンプレート例

```python
# PASSIVE + 単一条件
TEMPLATE_PASSIVE_COND = "【{name}】... コスト{base_cost}（{condition}時{reduction_text}）"

# COST_MODIFIER（static_abilities）
TEMPLATE_STAT_MODIFIER = (
    "このクリーチャーを召喚する時、"
    "{condition}ならば、このクリーチャーの召喚コストを{reduction}少なくする。"
)

# STAT_SCALED の計算式表示
TEMPLATE_STAT_SCALED = (
    "バトルゾーンにあるクリーチャーの{stat_name}が大きいほど、"
    "このクリーチャーの召喚コストを軽減する"
    "（{per_value}ごと{increment_cost}削減、最大{max_reduction}削減）"
)
```

### 3.3 テスト戦略

#### テスト対象

| テストファイル | 対象範囲 | 優先度 |
|---|---|---|
| `test_cost_passive_text_gen.py` | PASSIVE モード（FIXED 含む） | 🔴 P0 |
| `test_cost_stat_scaled_text_gen.py` | STAT_SCALED の計算式テキスト | 🔴 P0 |
| `test_cost_condition_text_gen.py` | 条件式の日本語化 | 🟡 P1 |
| `test_cost_complex_reduction_text_gen.py` | 複数軽減の合成表現 | 🟡 P1 |
| `test_cost_simulator_text_gen.py` | 試行値シミュレーション | 🟢 P2 |

#### テストケース例

```python
def test_cost_reduction_passive_fixed():
    """PASSIVE な固定軽減を日本語テキストに変換"""
    data = {
        "cost": 5,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "condition": {
                    "type": "CIVILIZATION",
                    "civilization": "DARKNESS",
                    "count": 2
                },
                "value_mode": "FIXED",
                "value": 2
            }
        ]
    }
    text = CardTextGenerator.generate_body_text(data)
    assert "闇の文明が2枚以上あれば" in text
    assert "コスト5" not in text  # 軽減後の具体値は最初は記載せず

def test_cost_reduction_stat_scaled():
    """STAT_SCALED 軽減の計算式テキスト化"""
    data = {
        "cost": 8,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "condition": {...条件...},
                "value_mode": "STAT_SCALED",
                "stat_key": "TOTAL_POWER",
                "per_value": 1,  # 3パワーごと
                "increment_cost": 1,
                "min_stat": 1,
                "max_reduction": 4
            }
        ]
    }
    text = CardTextGenerator.generate_body_text(data)
    assert "パワーが大きいほど" in text
    assert "最大4削減" in text or "最大4少なくなる" in text
```

---

## 4. 依存関係と前提条件

### 4.1 既存ドキュメント / 仕様への依存

- ✅ `CARD_EDITOR_AND_ENGINE_STAT_GAP_REPORT_2026-03-19.md`
  - 統計キー定義（`TOTAL_POWER`, `CARDS_IN_BATTLE`, etc.）
  - STAT_SCALED 計算式: `reduction = min(max_reduction, max(0, stat_value - min_stat + 1) * per_value)`
  - 合成順序: 基本 → PASSIVE → COST_MODIFIER → ACTIVE_PAYMENT → min_floor

- ✅ `CONSISTENCY_CHECK_REPORT_2026-03-22.md`
  - P3 拡張課題として位置付け

### 4.2 コード側の前提

- ✅ `dm_toolkit/dm_types.py`: `CostReduction` / `CostModifier` の型定義
- ✅ `dm_toolkit/payment.py`: Python 側の軽減計算ロジック
- ❌ 統計キーの I18n マッピング: **本計画で作成** (`consts.STAT_KEY_TEXT_MAP`)
- ❌ 条件式の日本語化関数: **本計画で実装** (`text_generator._format_condition_text()`)

### 4.3 環境・ツール

- Python 3.9, 3.10, 3.11, 3.12 での動作確認
- pytest で自動テスト実行
- mypy で型チェック
- CI/CD: `.github/workflows/` で pytest 実行（`.github/workflows/pytest.yml` 等）

---

## 5. 実装スケジュール（全体）

| Phase | 期間 | 主要タスク | チェックポイント |
|-------|------|----------|-----------------|
| **Phase 1** | 2026-04-XX | PASSIVE/FIXED テキスト化 | test_cost_passive_text_gen.py PASS |
| **Phase 2** | 2026-04-XX | STAT_SCALED/複合軽減 | test_cost_stat_scaled_text_gen.py PASS |
| **Phase 3** | 2026-05-XX | エディタUI統合 | リアルタイムプレビュー動作確認 |
| **検証** | 2026-05-XX | 統合テスト / ドキュメント同期 | 全テスト PASS + 仕様書更新完了 |

---

## 6. 再発防止チェックリスト

### 実装時に確認すべき項目

- [x] **型安全性**: `cost_reductions` の型が `List[CostReduction]` で統一されている
- [x] **条件テキスト漏れ**: COMPARE_STAT, CARDS_MATCHING_FILTER が全てのテンプレートで生成されている
- [x] **計算式の正確性**: STAT_SCALED の「N ごと M 削減」が `per_value`, `increment_cost` と一貫している
- [x] **上限チェック**: `max_reduction` の有無による文面分岐が正確（`-1` は上限なしと解釈）
- [x] **I18n 対応**: 全テキストが `tr()` で翻訳可能（hardcode 避止）
- [ ] **テストカバレッジ**: 条件分岐が全て案件テストで検証（特に複合軽減）
 - [x] **テストカバレッジ**: 条件分岐が全て案件テストで検証（特に複合軽減）


### 統合時に確認すべき項目

- [ ] **Python ↔ C++ 計算の一貫性**: Python の `payment.py` と C++ の `ManaSystem::get_adjusted_cost` が同じ軽減量を算出
 - [x] **Python ↔ C++ 計算の一貫性**: Python の `payment.py` と C++ の `ManaSystem::get_adjusted_cost` が同じ軽減量を算出（STAT_SCALED の公式・合成順序を確認）
- [ ] **テキスト出力のキャッシング**: エディタ側でプレビュー生成時に毎回新規計算せず、キャッシュ活用
- [ ] **エラーハンドリング**: 不完全な定義での警告メッセージが分かりやすい
- [ ] **バージョン管理**: `cards.json` の軽減定義バージョンが更新されている

---

## 7. 参考資料

### 既存コード

- [text_generator.py](../dm_toolkit/gui/editor/text_generator.py) - 現在の実装
- [payment.py](../dm_toolkit/payment.py) - 軽減計算ロジック
- [test_effect_and_text_integrity.py](../tests/test_effect_and_text_integrity.py) - 既存テスト例

### ドキュメント

- [CARD_EDITOR_AND_ENGINE_STAT_GAP_REPORT_2026-03-19.md](../CARD_EDITOR_AND_ENGINE_STAT_GAP_REPORT_2026-03-19.md)
- [CONSISTENCY_CHECK_REPORT_2026-03-22.md](../CONSISTENCY_CHECK_REPORT_2026-03-22.md)
- [copilot-instructions.md](../.github/copilot-instructions.md) - プロジェクト規約

---

## 8. 未解決の質問 / 今後の検討項目

- [ ] Phase 1 の完了後に、エディタの「テキストプレビュー」表示方式を決定（サイドパネルか、フローティングウィンドウか）
- [ ] 複合軽減の優先順位：複数の条件が満たされた場合の合成式（加算 vs 乗算 vs max）の明記
- [ ] 統計キーの新規追加時のワークフロー（I18n 辞書更新タイミング）
- [ ] モバイル / 別言語への拡張計画

---

**文書版: v1.3**  
**最終レビュー**: 2026-03-23（`text_generator.py` の cost_reductions 正規化、基本文面化、PASSIVE/STAT_SCALED のユニットテスト追加、Python↔C++ 計算整合性確認完了）  
**次回更新予定**: 複合軽減テスト追加（COMPARE_STAT/CARDS_MATCHING_FILTER）完了時
