# -*- coding: utf-8 -*-
"""
id=6 (歌舞音愛 ヒメカット) のデータ整合性チェックとテキスト生成検証
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def load_card_by_id(card_id: int):
    """cards.json からカードをロード"""
    cards_path = project_root / "data" / "cards.json"
    with open(cards_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    for card in cards:
        if card.get("id") == card_id:
            return card
    return None


def check_card_structure(card):
    """カードデータの構造をチェック"""
    print("\n" + "=" * 80)
    print("カードデータ構造チェック")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # 基本情報
    print(f"\n✓ ID: {card.get('id')}")
    print(f"✓ 名前: {card.get('name')}")
    print(f"✓ タイプ: {card.get('type')}")
    print(f"✓ 文明: {card.get('civilizations')}")
    print(f"✓ コスト: {card.get('cost')}")
    print(f"✓ パワー: {card.get('power')}")
    
    # キーワード
    keywords = card.get('keywords', {})
    if keywords:
        print(f"\n✓ キーワード:")
        for kw, val in keywords.items():
            print(f"  - {kw}: {val}")
    
    # エフェクト
    effects = card.get('effects', [])
    print(f"\n✓ エフェクト数: {len(effects)}")
    
    for i, effect in enumerate(effects, 1):
        print(f"\n  エフェクト {i}:")
        print(f"    UID: {effect.get('uid', 'なし')}")
        print(f"    トリガー: {effect.get('trigger', 'なし')}")
        
        # トリガースコープ
        trigger_scope = effect.get('trigger_scope', 'NONE')
        if trigger_scope and trigger_scope != 'NONE':
            print(f"    トリガースコープ: {trigger_scope}")
        
        # 条件
        condition = effect.get('condition', {})
        if condition:
            print(f"    条件タイプ: {condition.get('type', 'NONE')}")
            if condition.get('type') != 'NONE':
                print(f"    条件詳細: {condition}")
        
        # コマンド
        commands = effect.get('commands', [])
        print(f"    コマンド数: {len(commands)}")
        
        for j, cmd in enumerate(commands, 1):
            cmd_type = cmd.get('type', 'UNKNOWN')
            print(f"      コマンド {j}: {cmd_type}")
            
            # IFコマンドの特別処理
            if cmd_type == "IF":
                # 条件
                cond = cmd.get('condition', {}) or cmd.get('target_filter', {})
                if cond:
                    print(f"        条件: {cond.get('type', 'NONE')} = {cond.get('value', 'なし')}")
                
                # if_true
                if_true = cmd.get('if_true', [])
                print(f"        if_true分岐: {len(if_true)}個のコマンド")
                for k, subcmd in enumerate(if_true, 1):
                    subcmd_type = subcmd.get('type', 'UNKNOWN')
                    print(f"          {k}. {subcmd_type}", end="")
                    if subcmd_type == "DRAW_CARD":
                        print(f" (amount={subcmd.get('amount', 0)}, optional={subcmd.get('optional', False)})", end="")
                    print()
                
                # if_false
                if_false = cmd.get('if_false', [])
                if if_false:
                    print(f"        if_false分岐: {len(if_false)}個のコマンド")
                    for k, subcmd in enumerate(if_false, 1):
                        subcmd_type = subcmd.get('type', 'UNKNOWN')
                        print(f"          {k}. {subcmd_type}")
    
    # Spell Side
    spell_side = card.get('spell_side')
    if spell_side:
        print(f"\n✓ 呪文側あり:")
        print(f"    名前: {spell_side.get('name')}")
        print(f"    コスト: {spell_side.get('cost')}")
        spell_effects = spell_side.get('effects', [])
        print(f"    エフェクト数: {len(spell_effects)}")
    
    # 整合性チェック
    print("\n" + "=" * 80)
    print("整合性チェック")
    print("=" * 80)
    
    # 1. friend_burstキーワードの確認
    if keywords.get('friend_burst'):
        print("✓ friend_burst キーワード: あり")
    else:
        warnings.append("friend_burst キーワードが設定されていません")
    
    # 2. エフェクトのトリガーチェック
    for i, effect in enumerate(effects, 1):
        trigger = effect.get('trigger', 'NONE')
        if trigger == 'NONE':
            warnings.append(f"エフェクト{i}: トリガーが NONE です")
        
        # 3. コマンドの存在チェック
        commands = effect.get('commands', [])
        if not commands:
            issues.append(f"エフェクト{i}: コマンドが空です")
        
        # 4. IFコマンドのif_trueチェック
        for j, cmd in enumerate(commands, 1):
            if cmd.get('type') == 'IF':
                if_true = cmd.get('if_true', [])
                if not if_true:
                    warnings.append(f"エフェクト{i}のコマンド{j}: IFコマンドにif_trueが空です")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("チェック結果")
    print("=" * 80)
    
    if issues:
        print(f"\n❌ エラー: {len(issues)}件")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"\n⚠️ 警告: {len(warnings)}件")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ 問題なし: データ構造は正常です")
    
    return len(issues) == 0


def test_text_generation(card):
    """テキスト生成のテスト"""
    print("\n" + "=" * 80)
    print("テキスト生成テスト")
    print("=" * 80)
    
    # 全体テキスト生成
    full_text = CardTextGenerator.generate_text(card)
    print("\n【生成されたテキスト】\n")
    print(full_text)
    
    # エフェクトごとのテキスト生成
    print("\n" + "=" * 80)
    print("エフェクト別テキスト")
    print("=" * 80)
    
    effects = card.get('effects', [])
    for i, effect in enumerate(effects, 1):
        print(f"\n--- エフェクト {i} ---")
        effect_text = CardTextGenerator._format_effect(effect, is_spell=False)
        print(effect_text)
        
        # コマンド別テキスト
        commands = effect.get('commands', [])
        for j, cmd in enumerate(commands, 1):
            print(f"\n  コマンド {j} ({cmd.get('type', 'UNKNOWN')}):")
            cmd_text = CardTextGenerator._format_command(cmd, is_spell=False)
            print(f"  {cmd_text}")


def test_expected_text(card):
    """期待されるテキストと比較"""
    print("\n" + "=" * 80)
    print("期待テキストとの比較")
    print("=" * 80)
    
    expected_texts = {
        "creature": [
            "■ 相手がカードを引いた時、2枚目以降なら、カードを1枚引いてもよい。"
        ],
        "spell": [
            "■ 相手のエレメントを2体まで選び、持ち主の手札に戻す。"
        ]
    }
    
    # クリーチャー側
    print("\n【クリーチャー側】")
    effects = card.get('effects', [])
    for i, effect in enumerate(effects):
        generated = CardTextGenerator._format_effect(effect, is_spell=False)
        print(f"\n生成: {generated}")
        if i < len(expected_texts["creature"]):
            expected = expected_texts["creature"][i]
            print(f"期待: {expected}")
            if generated.strip() == expected.strip():
                print("✅ 一致")
            else:
                print("⚠️ 不一致")
    
    # 呪文側
    spell_side = card.get('spell_side')
    if spell_side:
        print("\n【呪文側】")
        spell_effects = spell_side.get('effects', [])
        for i, effect in enumerate(spell_effects):
            generated = CardTextGenerator._format_effect(effect, is_spell=True)
            print(f"\n生成: {generated}")
            if i < len(expected_texts["spell"]):
                expected = expected_texts["spell"][i]
                print(f"期待: {expected}")
                if generated.strip() == expected.strip():
                    print("✅ 一致")
                else:
                    print("⚠️ 不一致")


def main():
    """メイン処理"""
    print("\n" + "=" * 80)
    print("id=6 整合性チェックとテキスト生成検証")
    print("=" * 80)
    
    # カードをロード
    card = load_card_by_id(6)
    if not card:
        print("\n❌ エラー: id=6 のカードが見つかりません")
        return 1
    
    try:
        # 1. データ構造チェック
        structure_ok = check_card_structure(card)
        
        # 2. テキスト生成テスト
        test_text_generation(card)
        
        # 3. 期待テキストとの比較
        test_expected_text(card)
        
        print("\n" + "=" * 80)
        if structure_ok:
            print("✅ 全チェック完了")
        else:
            print("⚠️ チェック完了（警告あり）")
        print("=" * 80 + "\n")
        
        return 0 if structure_ok else 1
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
