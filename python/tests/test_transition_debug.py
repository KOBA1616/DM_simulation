# -*- coding: utf-8 -*-
"""
id=6 呪文側のTRANSITIONコマンドテスト
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_transition_command():
    """TRANSITIONコマンド単体のテスト"""
    
    # id=6 spell side command
    transition_cmd = {
        "uid": "39309ebe-d70d-46c2-8541-ac0e66849781",
        "type": "TRANSITION",
        "amount": 2,
        "from_zone": "BATTLE",
        "input_value_usage": "",
        "optional": False,
        "target_filter": {
            "types": ["ELEMENT"]
        },
        "target_group": "PLAYER_OPPONENT",
        "to_zone": "HAND",
        "up_to": True
    }
    
    print("=" * 80)
    print("TRANSITIONコマンドテスト")
    print("=" * 80)
    print("\n入力コマンド:")
    print(json.dumps(transition_cmd, indent=2, ensure_ascii=False))
    
    # Format the TRANSITION command
    result = CardTextGenerator._format_command(transition_cmd, is_spell=True)
    
    print("\n生成結果:")
    print(result)
    
    print("\n期待結果:")
    print("相手のエレメントを2体まで選び、手札に戻す。")
    
    # Test zone normalization
    print("\n" + "=" * 80)
    print("ゾーン正規化テスト")
    print("=" * 80)
    
    from_zone_normalized = CardTextGenerator._normalize_zone_name("BATTLE")
    to_zone_normalized = CardTextGenerator._normalize_zone_name("HAND")
    
    print(f"BATTLE → {from_zone_normalized}")
    print(f"HAND → {to_zone_normalized}")


if __name__ == '__main__':
    test_transition_command()
