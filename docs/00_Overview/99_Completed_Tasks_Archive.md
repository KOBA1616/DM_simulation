# Archive of Completed Implementation Tasks

このファイルは `docs/00_Overview/00_Status_and_Requirements_Summary.md` から移動された、完了済みまたは廃止されたタスクの記録です。

## Phase 1 - 3: 完了タスク (Completed Tasks)

### GUI/Editor 実装
*   **革命チェンジのデータ構造不整合の解消**:
    *   Editor UIをボタンからチェックボックスに変更し、条件（Condition）の設定画面を追加しました。
    *   Engineが要求するルートレベルの `revolution_change_condition` と、ロジックツリー上の `revolution_change` キーワードの同期を実装しました。
*   **文明指定のキー不整合の解消**:
    *   Editorでの保存形式をEngine推奨の複数形 `"civilizations"` に統一しました。
*   **UI改善 (Visual Improvements)**:
    *   カードの縁（多色選択時の紫色など）を黒の細線に変更。
    *   マナコストの丸表示を文明色で構成し、多色は等分割表示に対応。文字色は細い黒縁のある白文字に変更。
    *   ツインパクトカードのパワー表記位置を左下に調整。
    *   プレビュー更新ボタンの追加。
    *   シールド枚数のデッキ風ブロック表示。
    *   エクストライフキーワードの追加。
    *   トリガー：「呪文を唱えた時」の追加。
    *   ツインパクトのエフェクト追加時の自動スライド処理。
    *   アクション選択時の入力欄マスキング（不要なフィールドの非表示化）。
    *   アクションプルダウンの日本語化。
    *   グラデーション色の調整（濃くする）。

### Engine 実装
*   **ONNX Runtime (C++) 統合**:
    *   PyTorchモデルのエクスポート、C++ `NeuralEvaluator` への統合、推論の完全C++化を完了。
*   **デッキ進化ロジックのC++化**:
    *   PythonスクリプトからC++モジュールへの移行を完了。
*   **アクションシステム**:
    *   `MOVE_CARD`, `COMPARE_STAT`, `SELECT_OPTION` などの汎用アクションの実装完了。
    *   変数リンクシステムの主要ハンドラへの適用完了。

---
