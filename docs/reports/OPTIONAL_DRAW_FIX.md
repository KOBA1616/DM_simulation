## CommandSystem optionalサポート追加

### 問題
月光電人オボロカゲロウをプレイすると、optional=trueのDRAW_CARDが自動実行されてゲームが停止する。

### 修正内容

#### src/engine/systems/command_system.cpp

DRAW_CARDコマンドに`optional`フラグのサポートを追加：

```cpp
// optional=trueかつcount>0の場合、SELECT_NUMBER待ちを作成
if (cmd.optional && count > 0) {
    PendingEffect pending(EffectType::SELECT_NUMBER, source_instance_id, player_id);
    pending.execution_context["_min_select"] = 0;
    pending.execution_context["_max_select"] = count;
    
    // 継続効果として、選択した枚数を引くコマンドを設定
    CommandDef draw_cmd;
    draw_cmd.input_value_key = "_selected_number";
    ...
    state.pending_effects.push_back(pending);
    return; // ここで処理を中断し、プレイヤーの選択を待つ
}
```

#### src/engine/game_instance.cpp

SELECT_NUMBER処理で`_selected_number`キーに値を格納：

```cpp
pe.execution_context["_selected_number"] = chosen_number;
```

### 動作フロー

1. クリーチャーをプレイ -> CIP効果発動
2. CommandSystem: DRAW_CARDがoptional=true -> SELECT_NUMBER待ちを作成
3. ゲームがpending_effectsの処理待ちで一時停止
4. コマンド生成: SELECT_NUMBERコマンドを生成
5. プレイヤーが枚数を選択
6. GameInstance: 選択した値を`_selected_number`に格納
7. 継続コマンド実行: 選択した枚数だけカードを引く

### テスト

```powershell
.\scripts\build.ps1 -Config Release
.\scripts\run_gui.ps1
```

クリーチャーをプレイした際に、SELECT_NUMBERの選択画面が表示されることを確認。
