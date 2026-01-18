#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""整合性チェックスクリプト"""

import json
import sys
import io
import os
from pathlib import Path

# 標準出力を UTF-8 に設定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_project_root():
    """プロジェクトルートディレクトリを取得"""
    return Path(__file__).resolve().parent.parent

def check_command_ui_json():
    """command_ui.json の整合性チェック"""
    print("=" * 60)
    print("1. command_ui.json の整合性チェック")
    print("=" * 60)
    
    root = get_project_root()
    path = root / 'data/configs/command_ui.json'

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    groups = data.get('COMMAND_GROUPS', {})
    all_commands = {}
    duplicates = []
    
    for group_name, commands in groups.items():
        for cmd in commands:
            if cmd in all_commands:
                duplicates.append((cmd, all_commands[cmd], group_name))
            else:
                all_commands[cmd] = group_name
    
    print(f"総グループ数: {len(groups)}")
    print(f"総コマンド数: {len(all_commands)}")
    print(f"グループ: {', '.join(groups.keys())}")
    
    if duplicates:
        print(f"\n❌ 重複コマンド検出:")
        for cmd, old_group, new_group in duplicates:
            print(f"  {cmd}: {old_group} <-> {new_group}")
        return False
    else:
        print("\n✅ 重複なし")
    
    # UI定義とグループの対応チェック
    defined_commands = [k for k in data.keys() if k != 'COMMAND_GROUPS' and k.isupper()]
    print(f"\n定義済みコマンドUI: {len(defined_commands)}")
    
    missing = []
    undefined = []
    for cmd in all_commands:
        if cmd not in defined_commands:
            missing.append(cmd)
    
    for cmd in defined_commands:
        if cmd not in all_commands and cmd != 'NONE':
            undefined.append(cmd)
    
    if missing:
        print(f"\n⚠️ UI定義がないコマンド: {missing}")
    if undefined:
        print(f"\n⚠️ グループに登録されていないコマンド: {undefined}")
    
    if not missing and not undefined:
        print("\n✅ コマンドUI定義とグループの対応が完全")
        return True
    return False

def check_ja_json():
    """ja.json の翻訳キー確認"""
    print("\n" + "=" * 60)
    print("2. ja.json の翻訳キー確認")
    print("=" * 60)
    
    root = get_project_root()
    path = root / 'data/locale/ja.json'

    with open(path, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    
    # 必要な翻訳キー
    required_keys = [
        'DRAW', 'CARD_MOVE', 'REPLACEMENT', 'DECK_OPS', 'PLAY',
        'BUFFER', 'CHEAT_PUT', 'GRANT', 'LOGIC', 'BATTLE',
        'RESTRICTION', 'SPECIAL', 'REPLACE_CARD_MOVE'
    ]
    
    missing = []
    for key in required_keys:
        if key not in translations:
            missing.append(key)
    
    if missing:
        print(f"❌ 不足している翻訳キー: {missing}")
        return False
    else:
        print("✅ 全ての必要な翻訳キーが存在:")
        for key in required_keys:
            print(f"  {key}: {translations[key]}")
        return True

def check_schema_config():
    """schema_config.py のスキーマ確認"""
    print("\n" + "=" * 60)
    print("3. schema_config.py のスキーマ確認")
    print("=" * 60)
    
    # プロジェクトルートをsys.pathに追加してインポートできるようにする
    sys.path.insert(0, str(get_project_root()))

    try:
        from dm_toolkit.gui.editor.schema_config import register_all_schemas
        from dm_toolkit.gui.editor.schema_def import get_schema, SCHEMA_REGISTRY
        
        register_all_schemas()
        
        critical_commands = [
            'REPLACE_CARD_MOVE', 'MUTATE', 'DRAW_CARD', 'TRANSITION',
            'QUERY', 'FLOW', 'CHOICE', 'IF', 'IF_ELSE'
        ]
        
        missing_schemas = []
        for cmd in critical_commands:
            schema = get_schema(cmd)
            if schema is None:
                missing_schemas.append(cmd)
        
        if missing_schemas:
            print(f"❌ スキーマが登録されていないコマンド: {missing_schemas}")
            return False
        else:
            print("✅ 全ての重要なコマンドスキーマが登録されている:")
            for cmd in critical_commands:
                schema = get_schema(cmd)
                fields = [f.key for f in schema.fields]
                print(f"  {cmd}: {len(fields)} fields {fields[:3]}...")
            return True
    except Exception as e:
        print(f"❌ スキーマ確認失敗: {e}")
        return False

def check_i18n_integration():
    """i18n 統合確認"""
    print("\n" + "=" * 60)
    print("4. i18n 統合確認")
    print("=" * 60)
    
    # プロジェクトルートをsys.pathに追加
    sys.path.insert(0, str(get_project_root()))

    try:
        from dm_toolkit.gui.i18n import tr
        
        test_keys = [
            'REPLACEMENT', 'REPLACE_CARD_MOVE', 'DRAW', 'CARD_MOVE'
        ]
        
        for key in test_keys:
            result = tr(key)
            print(f"  tr('{key}'): {result}")
        
        print("\n✅ i18n 翻訳が正常に動作")
        return True
    except Exception as e:
        print(f"❌ i18n 翻訳失敗: {e}")
        return False

def _iter_commands(commands):
    for cmd in commands or []:
        yield cmd
        for branch in ['if_true', 'if_false']:
            if cmd.get(branch):
                yield from _iter_commands(cmd.get(branch))
        options = cmd.get('options') or []
        for opt in options:
            yield from _iter_commands(opt)

def check_grant_keyword_actions():
    """ADD_KEYWORD/MUTATE アクションの整合性チェック"""
    print("\n" + "=" * 60)
    print("5. 付与・キーワード付与アクションの整合性チェック")
    print("=" * 60)

    root = get_project_root()
    path = root / 'data/cards.json'

    with open(path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    issues = []

    def check_cmd(cmd, card):
        ctype = cmd.get('type')
        if ctype == 'ADD_KEYWORD':
            str_val = cmd.get('str_val')
            legacy = cmd.get('str_param')
            if not str_val:
                issues.append((card.get('id'), card.get('name'), cmd.get('uid'), 'ADD_KEYWORD', 'str_val_missing'))
            if legacy:
                issues.append((card.get('id'), card.get('name'), cmd.get('uid'), 'ADD_KEYWORD', 'legacy_str_param'))
        elif ctype == 'MUTATE':
            mk = cmd.get('mutation_kind')
            if not mk:
                issues.append((card.get('id'), card.get('name'), cmd.get('uid'), 'MUTATE', 'mutation_kind_missing'))

    for card in cards:
        for ability_key in ['effects', 'static_abilities', 'reaction_abilities']:
            for ability in card.get(ability_key, []) or []:
                for cmd in _iter_commands(ability.get('commands', [])):
                    check_cmd(cmd, card)

    if not issues:
        print("✅ ADD_KEYWORD/MUTATE の整合性に問題なし")
        return True

    print(f"❌ 整合性問題: {len(issues)} 件")
    for card_id, name, uid, ctype, reason in issues:
        print(f"  id={card_id} {name} ({ctype}, uid={uid}) -> {reason}")
    return False

if __name__ == '__main__':
    results = []
    results.append(("command_ui.json", check_command_ui_json()))
    results.append(("ja.json", check_ja_json()))
    results.append(("schema_config", check_schema_config()))
    results.append(("i18n", check_i18n_integration()))
    results.append(("grant_keyword_actions", check_grant_keyword_actions()))
    
    print("\n" + "=" * 60)
    print("整合性チェック結果")
    print("=" * 60)
    for name, result in results:
        status = "✅ OK" if result else "❌ NG"
        print(f"{status} {name}")
    
    all_ok = all(r for _, r in results)
    sys.exit(0 if all_ok else 1)
