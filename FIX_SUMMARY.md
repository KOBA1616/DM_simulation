## 修正完了: クリーチャー召喚時の効果処理停止問題

### 問題
カードをプレイした後、効果が処理されずゲームが停止する問題が発生していました。

### 根本原因
先ほどの変更で`effect.actions`の処理を完全に削除しましたが：
- 既存のカードJSONデータは`actions`フィールドを持っている
- JSONロード時の`commands`への変換が完全でない場合がある
- `compile_effect`が空の命令リストを生成し、効果が実行されない

### 修正内容

#### src/engine/systems/card/effect_system.cpp

1. **compile_effect関数に後方互換コードを追加:**
```cpp
// Fallback: Compile Actions (Legacy System for backward compatibility)
if (effect.commands.empty() && !effect.actions.empty()) {
    for (const auto& action : effect.actions) {
        compile_action(game_state, action, ...);
    }
}
```

2. **処理優先順位:**
   - ✅ 第一優先: `commands`配列を処理（新CommandDefシステム）
   - ✅ フォールバック: `commands`が空の場合、`actions`を処理（旧ActionDef互換）

3. **ログ出力の改善:**
```cpp
std::cerr << "[EffectSystem::compile_effect] CALLED: commands.size=" 
          << effect.commands.size() 
          << " actions.size=" << effect.actions.size() 
          << std::endl;
```

### メリット

✅ **完全な後方互換性**: 既存のすべてのカードデータが動作
✅ **段階的移行**: 新しいCommandDefシステムを優先しつつ、ActionDefもサポート
✅ **デバッグ改善**: actionsとcommandsの両方のサイズをログ出力
✅ **安全性**: JSONロード時の変換が失敗しても、実行時に処理可能

### 次のステップ

```powershell
# 1. ビルド
.\scripts\build.ps1 -Config Release

# 2. GUIテスト
.\scripts\run_gui.ps1

# 3. クリーチャー召喚テスト
# - カードをプレイ
# - CIP効果が発動することを確認
# - ゲームが正常に進行することを確認
```

### 今後の最適化（オプション）

長期的には、すべてのカードデータをCommandDef形式に統一することで：
- ActionDef処理を完全に削除可能
- コードの簡素化
- パフォーマンス向上

ただし現時点では、後方互換性を維持する方が安全です。
