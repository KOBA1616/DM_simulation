"""カードID別の整合性チェックスクリプト
- カードテキスト(stored) vs 生成テキスト(generated)
- C++実装済みコマンドタイプのチェック
"""
import json
import sys
from typing import Iterator

sys.path.insert(0, '.')

# C++ enum に存在するすべてのコマンドタイプ (card_json_types.hpp より)
CPP_ENUM_TYPES = {
    "NONE", "TRANSITION", "MUTATE", "FLOW", "QUERY", "DRAW_CARD", "DISCARD", "DESTROY",
    "BOOST_MANA", "TAP", "UNTAP", "POWER_MOD", "ADD_KEYWORD", "RETURN_TO_HAND",
    "BREAK_SHIELD", "SEARCH_DECK", "SHIELD_TRIGGER", "MOVE_CARD", "ADD_MANA", "SEND_TO_MANA",
    "PLAYER_MANA_CHARGE", "SEARCH_DECK_BOTTOM", "ADD_SHIELD", "SEND_TO_DECK_BOTTOM",
    "ATTACK_PLAYER", "ATTACK_CREATURE", "BLOCK", "RESOLVE_BATTLE", "RESOLVE_PLAY",
    "RESOLVE_EFFECT", "SHUFFLE_DECK", "LOOK_AND_ADD", "MEKRAID", "REVEAL_CARDS",
    "PLAY_FROM_ZONE", "CAST_SPELL", "SUMMON_TOKEN", "SHIELD_BURN", "SELECT_NUMBER",
    "CHOICE", "LOOK_TO_BUFFER", "REVEAL_TO_BUFFER", "SELECT_FROM_BUFFER",
    "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE", "FRIEND_BURST", "REGISTER_DELAYED_EFFECT",
    "IF", "IF_ELSE", "ELSE", "PASS", "USE_ABILITY", "MANA_CHARGE", "SELECT_TARGET",
    "APPLY_MODIFIER", "GRANT_KEYWORD", "PUT_CREATURE", "REPLACE_CARD_MOVE",
    "ADD_RESTRICTION", "SELECT_OPTION",
    # 再発防止: 新しく追加した型
    "DRAW",            # DRAW_CARD の別名 (FLOW内サブコマンド)
    "REPLACE_MOVE_CARD",  # マグナム系置換効果
}

# C++ command_system.cpp の switch で case ハンドラが存在するコマンドタイプ
CPP_IMPLEMENTED = {
    "TRANSITION", "MUTATE", "FLOW", "IF", "IF_ELSE", "ELSE", "QUERY", "SHUFFLE_DECK",
    "MANA_CHARGE",
    "DRAW_CARD", "BOOST_MANA", "ADD_MANA", "DESTROY", "DISCARD",
    "TAP", "UNTAP", "RETURN_TO_HAND", "BREAK_SHIELD", "POWER_MOD",
    "ADD_KEYWORD", "GRANT_KEYWORD", "SEARCH_DECK", "SEND_TO_MANA",
    "SELECT_NUMBER", "SELECT_OPTION", "APPLY_MODIFIER", "PUT_CREATURE",
    "CAST_SPELL", "ADD_RESTRICTION", "REPLACE_CARD_MOVE",
    # 再発防止: 新しく実装した型
    "DRAW",               # DRAW_CARD の別名
    "REPLACE_MOVE_CARD",  # GAME_ACTION(REPLACE_MOVE_CARD) を生成
    "REVEAL_TO_BUFFER",   # DECK_TOPからBUFFERへの移動
    "SELECT_FROM_BUFFER", # WAIT_INPUT(SELECT_FROM_BUFFER)
    "MOVE_BUFFER_TO_ZONE", # BUFFERから対象ゾーンへの移動
}

# 再発防止: IF条件評価は command_system.cpp の generate_primitive_instructions で
#   cmd.target_filter.type が存在する場合に ConditionDef を合成して評価する実装済み。
#   (OPPONENT_DRAW_COUNT / COMPARE_INPUT 対応)
# IF_CONDITION_BUG_NOTE は削除済み（フラグを使わないのでコメントアウトのみ）
IF_CONDITION_BUG_NOTE = ""  # 空にすることで旧バグノートを無効化


def walk_cmds(cmds: list) -> Iterator[dict]:
    for cmd in cmds:
        yield cmd
        for sub_list_key in ("sub_commands", "if_true", "if_false"):
            if cmd.get(sub_list_key):
                yield from walk_cmds(cmd[sub_list_key])
        if cmd.get("options"):
            for opt in cmd["options"]:
                if isinstance(opt, list):
                    yield from walk_cmds(opt)
                elif isinstance(opt, dict) and opt.get("commands"):
                    yield from walk_cmds(opt["commands"])


def check_card(card: dict) -> dict:
    cid = card["id"]
    name = card["name"]
    effects = card.get("effects", [])

    all_types = []
    issues = []
    notes = []

    for eff in effects:
        for cmd in walk_cmds(eff.get("commands", [])):
            ctype = cmd.get("type", "UNKNOWN")
            all_types.append(ctype)

            if ctype not in CPP_ENUM_TYPES:
                issues.append(f"❌ C++ enumに未定義(JSON→NONEにフォールバック): {ctype}")
            elif ctype not in CPP_IMPLEMENTED:
                issues.append(f"⚠️  C++ enumに存在するが case ハンドラ未実装(no-op): {ctype}")
            else:
                # 実装済みでも条件評価が特殊なケース
                if ctype == "IF":
                    tf = cmd.get("target_filter", {})
                    if tf and isinstance(tf, dict) and tf.get("type") not in (None, "", "NONE"):
                        # 再発防止: target_filter.type による IF 条件は
                        #   command_system.cpp で FilterDef.type → ConditionDef 変換して評価済み
                        note = IF_CONDITION_BUG_NOTE
                        if note:  # 空文字でなければ追加 (空=問題解消済み)
                            notes.append(note)

    # テキスト生成チェック (text_generator があれば)
    text_match = None
    generated = ""
    stored = card.get("text", "")
    try:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        gen = CardTextGenerator()
        generated = gen.generate_body_text(card)
        # stored が空の場合は設計上の仕様 (テキストは動的生成のみ)
        if stored == "":
            text_match = None  # 比較対象なし = N/A (設計上の仕様)
        else:
            text_match = generated.strip() == stored.strip()
    except Exception as e:
        text_match = None
        generated = f"ERROR: {e}"

    return {
        "id": cid,
        "name": name,
        "types_used": list(dict.fromkeys(all_types)),  # unique, order-preserving
        "issues": list(dict.fromkeys(issues)),
        "notes": list(dict.fromkeys(notes)),
        "text_match": text_match,
        "stored_text": stored,
        "generated_text": generated,
    }


with open("data/cards.json", encoding="utf-8") as f:
    cards = json.load(f)

print("=" * 70)
print("カードID別 整合性チェックレポート")
print("=" * 70)
print()

for card in cards:
    result = check_card(card)
    cid = result["id"]
    name = result["name"]
    types_str = ", ".join(result["types_used"]) if result["types_used"] else "(なし)"
    
    status_icon = "✅" if not result["issues"] else "❌"
    tm = result["text_match"]
    text_icon = "✅" if tm is True else ("❌" if tm is False else "N/A")

    print(f"┌─ Card {cid}: {name}")
    print(f"│  コマンド: {types_str}")
    if result["issues"]:
        for iss in result["issues"]:
            print(f"│  {iss}")
    else:
        print(f"│  C++実装: {status_icon} すべて実装済み")
    if result["notes"]:
        for note in result["notes"]:
            print(f"│  {note}")
    # 生成テキストを表示
    gen_text = result["generated_text"]
    if gen_text and not gen_text.startswith("ERROR"):
        # 改行を | に変換して1行に収める
        gen_preview = gen_text.replace("\n", " | ")[:120]
        print(f"│  生成テキスト: {gen_preview!r}")
    elif gen_text.startswith("ERROR"):
        print(f"│  テキスト生成エラー: {gen_text}")
    else:
        print(f"│  生成テキスト: (空)")
    print()

# サマリー
print("=" * 70)
print("サマリー")
print("=" * 70)
all_results = [check_card(c) for c in cards]
ok = [r for r in all_results if not r["issues"] and not r["notes"]]
warn_if = [r for r in all_results if r["notes"] and not r["issues"]]
noop = [r for r in all_results if any("no-op" in i for i in r["issues"])]
enum_missing = [r for r in all_results if any("enumに未定義" in i for i in r["issues"])]
print(f"✅ 問題なし         : {len(ok)} カード (ID: {[r['id'] for r in ok]})")
print(f"⚠️  IFの条件評価バグ : {len(warn_if)} カード (ID: {[r['id'] for r in warn_if]})")
print(f"⚠️  C++ no-opコマンド: {len(noop)} カード (ID: {[r['id'] for r in noop]})")
print(f"❌ enum未定義コマンド: {len(enum_missing)} カード (ID: {[r['id'] for r in enum_missing]})")
print()
print("--- 未実装コマンド 詳細 ---")
cmd_to_cards: dict = {}
for r in all_results:
    for iss in r["issues"]:
        if iss not in cmd_to_cards:
            cmd_to_cards[iss] = []
        cmd_to_cards[iss].append(r["id"])
for iss, card_ids in cmd_to_cards.items():
    print(f"  {iss}  (カードID: {card_ids})")
