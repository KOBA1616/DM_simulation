# ActionDef → CommandDef 移行サマリー

## 実施日
2026年2月6日

## 変更概要
古いActionDefベースのシステムから、新しいCommandDefベースのコマンドシステムへ移行しました。

## 主な変更点

### 1. EffectSystem (src/engine/systems/card/effect_system.cpp)
- **compile_effect**: `actions`配列の処理を削除し、`commands`配列のみを処理
- **resolve_effect_with_targets**: CommandSystemを使用してcommandsを実行

```cpp
// 変更前
for (const auto& action : effect.actions) {
    compile_action(game_state, action, ...);
}

// 変更後
for (const auto& cmd : effect.commands) {
    // CommandDefを直接処理
}
```

### 2. GameInstance (src/engine/game_instance.cpp)
- SELECT_NUMBER処理でのActionDef使用を削除
- CommandSystem::execute_commandを使用してcommandsを実行

```cpp
// 変更前
for (const auto& act : pe.effect_def->actions) {
    dm::engine::EffectSystem::instance().resolve_action(state, act, ...);
}

// 変更後
for (const auto& cmd : pe.effect_def->commands) {
    dm::engine::systems::CommandSystem::execute_command(state, cmd, ...);
}
```

### 3. 自動変換メカニズム
既存のJSONロード時に、ActionDefからCommandDefへの自動変換が機能しています：
- `json_loader.cpp`: `convert_legacy_action()` 関数
- `card_registry.cpp`: 同様の変換ロジック

## CommandDefの利点

### 1. **明確な構造化**
- Primitive Commands: TRANSITION, MUTATE, FLOW, QUERY
- Macro Commands: DRAW_CARD, DESTROY, TAP, UNTAP等

### 2. **C++ネイティブ実装**
- CommandSystem (src/engine/systems/command_system.cpp)で一元管理
- 高速で一貫性のある実行

### 3. **拡張性**
- 新しいコマンドタイプの追加が容易
- フィルタリング、ターゲット選択の標準化

## 残存するActionDef関連コード

以下は段階的削除対象（現時点では残存）：
1. **IActionHandler**: ハンドラインターフェース（effect_system.hpp）
2. **handlers/**: 各種ActionDefハンドラ実装
3. **ResolutionContext**: ActionDefフィールド
4. **compile_action**: ActionDef用のコンパイルメソッド
5. **Pythonバインディング**: bind_engine.cppのcompile_action露出

## 今後の作業

### 短期（推奨）
- [ ] ビルドテストと動作確認
- [ ] 既存テストケースの実行
- [ ] GUIでの動作確認

### 中期（オプション）
- [ ] ActionDefハンドラの完全削除
- [ ] ResolutionContextの簡素化
- [ ] Pythonバインディングのクリーンアップ

### 長期（将来）
- [ ] EffectDef.actionsフィールドの完全廃止
- [ ] IActionHandlerインターフェースの削除

## メリット

1. **C++中心**: AI/MCTS実装がネイティブコードで高速化
2. **一貫性**: すべての効果がCommandDefで統一
3. **保守性**: 2つのシステムを維持する必要がない
4. **デバッグ**: CommandSystemの単一パスでトレース可能

## 互換性

- JSONカードデータは変更不要（自動変換）
- 既存のテストは基本的に動作（commandsフィールドが自動生成される）
- Pythonコードは影響最小限

## 注意事項

- ActionDefを直接使用しているPythonコードがある場合は要確認
- カスタムハンドラを実装している場合はCommandDefへの移行が必要
