```markdown
# Card Editor Specifications (要件定義書 03)

## 1. 概要
Card Editor (Ver 2.0) は、JSON形式のカードデータをGUIベースで作成・編集するためのツールです。
Python (`PyQt6`) で実装されており、C++エンジンの `JsonLoader` と互換性のあるデータを生成します。

## 2. 機能要件 (Functional Requirements)

### 2.1 データ構造サポート
*   **JSON Compatibility**: `src/core/card_json_types.hpp` で定義された `CardDefinition`, `EffectDef`, `ActionDef` 構造を完全にサポート。
*   **Localization**: 内部値（ENUM文字列）と表示値（日本語）の相互変換。

### 2.2 編集機能
*   **Tree View**: カード -> 効果 (Effect) -> アクション (Action) の階層構造を視覚的に編集可能。
*   **Variable Linking**: `GET_GAME_STAT` 等の出力変数を、後続のアクションの入力としてプルダウン選択可能にする "Smart Link" 機能。
*   **Condition Editor**: `ConditionDef` (発動条件) の専用編集UI。数値比較、ゾーン指定などをサポート。

### 2.3 UI/UX
*   **Localization**: 全UIコンポーネントの日本語化。
*   **Validation**: 必須フィールドの入力チェック。
*   **Help/Tooltip**: 複雑な機能（Filter等）に対するツールチップヘルプ。

## 3. 今後の実装予定 (Pending Features)

### 3.1 Logic Mask (矛盾防止)
*   選択したカードタイプ（例：呪文）に対して無効なフィールド（例：パワー、種族）を非表示または編集不可にする機能。

### 3.2 Visual Effect Builder
*   より直感的にエフェクトチェーンを構築できるノードベースのエディタ（構想段階）。

```
