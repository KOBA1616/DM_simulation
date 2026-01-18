# -*- coding: utf-8 -*-
"""
IFコマンド処理のデバッグ
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_if_command():
    """IFコマンド単体のテスト"""
    
    # Simulate the IF command from id=6 card
    if_command = {
        "uid": "42d69007-e0b4-4794-9981-039a48ca4e86",
        "type": "IF",
        "input_value_usage": "",
        "target_filter": {
            "type": "OPPONENT_DRAW_COUNT",
            "value": 2
        },
        "if_true": [
            {
                "uid": "180f492e-7e80-444b-ae36-a9fa2d808b5b",
                "type": "DRAW_CARD",
                "amount": 1,
                "input_value_usage": "",
                "optional": True,
                "target_group": "PLAYER_SELF",
                "up_to": False
            }
        ]
    }
    
    print("=" * 80)
    print("IFコマンドテスト")
    print("=" * 80)
    print("\n入力コマンド:")
    print(json.dumps(if_command, indent=2, ensure_ascii=False))
    
    # Format the IF command
    result = CardTextGenerator._format_command(if_command, is_spell=False)
    
    print("\n生成結果:")
    print(result)
    
    print("\n期待結果:")
    print("相手がカードを2枚目以上引いたなら、カードを1枚引いてもよい")
    
    # Also test the inner DRAW_CARD command separately
    print("\n" + "=" * 80)
    print("内部DRAW_CARDコマンド単体テスト")
    print("=" * 80)
    
    draw_cmd = if_command["if_true"][0]
    print("\n入力コマンド:")
    print(json.dumps(draw_cmd, indent=2, ensure_ascii=False))
    
    draw_result = CardTextGenerator._format_command(draw_cmd, is_spell=False)
    print("\n生成結果:")
    print(draw_result)
    
    print("\n期待結果:")
    print("カードを1枚引いてもよい。")


if __name__ == '__main__':
    test_if_command()
