目的
- `ACTION` と `COMMAND` を段階的に融合するための中間表現（Canonical Internal Representation; CIR）を定義する。
- 最初は破壊的変更を避け、非破壊で読み取り専用の正規化関数を実装して既存ロジックと並行運用する。

設計方針
1. 非破壊: 元データは変更しない。CIR はコピー/ビューとして扱う。
2. 漸進移行: まず読み取り（表示・編集支援）で CIR を使い、後段で書き出し変換を導入する。
3. 単純で可検証: CIR は明確なフィールド（kind/type/options/branches/payload）を持ち、ユニットテストで双方向整合性を検証可能にする。

CIR フィールド（最小セット）
- kind: "ACTION" | "COMMAND" | "UNKNOWN"
- type: (ACTION の場合) action の `type` 値。
- options: 標準化されたオプション一覧。常に list[list[canonical_node]] 形式（選択肢ごとにノード列）。
- branches: COMMAND 用の分岐辞書: {"if_true": [canonical_nodes], "if_false": [canonical_nodes]}。
- payload: 元の dict をそのまま格納（参照用）。
- uid: 可能なら元の uid（無ければ生成し内部キャッシュキーに使用）

正規化ルール（概要）
- Action -> kind=ACTION, type=action['type'], options = action.get('options',[]) を list[list] 形式に整形。
- Command -> kind=COMMAND, branches = {'if_true':..., 'if_false':...}, options = command.get('options',[]) を list[list] 形式に整形。
- options 内の各要素はさらに canonicalize を再帰的に適用する。

段階的実装計画
1. `normalize.canonicalize(node)` を実装（読み取り専用、非破壊）。
2. `CardDataManager._internalize_item` を canonicalize を使うように拡張（既にキャッシュに登録するフックあり）。
3. 編集 UI（オプション追加、ブランチ作成）で internal cache を参照して表示改善。まずは読み取りのみに適用。
4. 単体テスト: 代表ケースを3つ（単純 action、command with branches、nested options）で canonicalize の出力を検証。
5. 次フェーズ: 書き出し変換（CIR -> legacy JSON）を用意して既存保存ロジックと統合。

テスト/検証
- `tests/editor/test_normalize.py` を作成し、各ケースで `payload` が不変であることと `options/branches` の形式が期待通りになることを検証する。

後続タスク（優先順）
A. 実装とテスト（本日）
B. UI 表示の利用（preview や inspector で CIR を参照、表示改善）
C. 編集操作での CIR 更新/反映（安全に実装）
D. 出力変換（書き出しの段階的置換）

備考
- まずは読み取り専用で既存動作への影響を最小化します。