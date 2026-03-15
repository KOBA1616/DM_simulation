"""通常ゲーム進行シミュレーション - 優先順位付き行動選択版

再発防止:
- CardDatabase() は空DBを返すため JsonLoader.load_cards() を必ず使用すること
- SELECT_NUMBER は target_instance が選択値。min値のコマンドを選べば効果を最小実行できる
- CHOICE    は target_instance がオプションインデックス。0 を選べば最初の選択肢
"""
import sys
import os
sys.path.insert(0, 'bin/Release')
import dm_ai_module
from dm_toolkit.consts import ZONES

_CARDS_JSON = os.path.join(os.path.dirname(__file__), 'data', 'cards.json')

PRIORITY_KEYWORDS = [
    "BREAK_SHIELD",       # シールドブレイク（ATTACK_PLAYERの後）
    "ATTACK_PLAYER",      # 直接攻撃
    "ATTACK_CREATURE",    # クリーチャー攻撃
    "PLAY_FROM_ZONE",     # クリーチャー召喚/呪文詠唱
    "MANA_CHARGE",        # マナチャージ
    "RESOLVE_EFFECT",     # エフェクト解決
]


def pick_action(legal):
    """優先順位に従って行動を選択する。
    SELECT_NUMBER/CHOICE は target_instance=0（最小値）を選ぶ。
    """
    # SELECT_NUMBER / CHOICE は特別処理: target_instance が最小のコマンドを選ぶ
    select_cmds = [c for c in legal if any(
        kw in str(getattr(c, 'type', '')).upper()
        for kw in ('SELECT_NUMBER', 'CHOICE', 'SELECT_OPTION')
    )]
    if select_cmds:
        # target_instance が最小のものを選ぶ（min値 = 最小選択）
        return min(select_cmds, key=lambda c: getattr(c, 'target_instance', 0))

    for keyword in PRIORITY_KEYWORDS:
        for cmd in legal:
            t = str(getattr(cmd, 'type', '')).upper()
            if keyword in t:
                return cmd
    return legal[0] if legal else None


if __name__ == '__main__':
    # 再発防止: CardDatabase() ではなく JsonLoader.load_cards() を使用
    db = dm_ai_module.JsonLoader.load_cards(_CARDS_JSON)
    db_size = len(db) if hasattr(db, '__len__') else '?'
    print(f"=== DBロード: {db_size} 枚のカード定義 ===")
    gi = dm_ai_module.GameInstance(42, db)
    gi.state.set_deck(0, [1]*40)
    gi.state.set_deck(1, [1]*40)
    gi.start_game()
    s = gi.state

    print("=== ゲーム開始時状態 ===")
    for pid in [0, 1]:
        p = s.players[pid]
        print("  P{}: hand={} deck={} shield={} mana={} bz={}".format(
            pid, len(p.hand), len(p.deck), len(p.shield_zone), len(p.mana_zone), len(p.battle_zone)))
    print("  turn={} active={} phase={}".format(s.turn_number, s.active_player_id, s.current_phase))

    # Use canonical zone names from dm_toolkit.consts and map to player attributes
    _ZONE_ATTRS = [z.lower() for z in ZONES]
    total_cards_start = sum(
        len(getattr(p, z, []))
        for p in s.players
        for z in _ZONE_ATTRS
    )
    print("  総カード数: {}".format(total_cards_start))
    print()

    MAX_STEPS = 600
    history = []
    cmd_counts = {}
    events = []
    last_turn = -1

    for step in range(MAX_STEPS):
        if s.game_over:
            events.append("[step={}] GAME_OVER turn={} winner={}".format(step, s.turn_number, s.winner))
            break

        turn = s.turn_number
        pid = s.active_player_id
        phase = str(s.current_phase)

        if turn != last_turn:
            p0, p1 = s.players[0], s.players[1]
            events.append("--- TURN={} P{} | P0[mana={} bz={} sh={}] P1[mana={} bz={} sh={}] ---".format(
                turn, pid,
                len(p0.mana_zone), len(p0.battle_zone), len(p0.shield_zone),
                len(p1.mana_zone), len(p1.battle_zone), len(p1.shield_zone)))
            last_turn = turn

        legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if not legal:
            history.append((step, turn, pid, phase, 'NO_CMD', 0))
            dm_ai_module.PhaseManager.next_phase(s, db)
            continue

        chosen = pick_action(legal)
        ctype = str(getattr(chosen, 'type', 'PASS'))
        cmd_counts[ctype] = cmd_counts.get(ctype, 0) + 1
        history.append((step, turn, pid, phase, ctype, len(legal)))

        ctype_up = ctype.upper()
        is_pass = 'PASS' in ctype_up

        if is_pass:
            dm_ai_module.PhaseManager.next_phase(s, db)
        else:
            try:
                gi.resolve_command(chosen)
                iid = getattr(chosen, 'instance_id', '-')
                tgt = getattr(chosen, 'target_instance', '-')
                events.append("  [step={} T{} P{}] {} iid={} tgt={} ph={}".format(
                    step, turn, pid, ctype, iid, tgt, phase))
            except Exception as e:
                events.append("  [step={}] ERR {}: {}".format(step, ctype, str(e)[:80]))
                dm_ai_module.PhaseManager.next_phase(s, db)
    else:
        events.append("=== {}ステップ完了 game_over={} ===".format(MAX_STEPS, s.game_over))

    # ---- レポート ----
    print("=== ゲーム進行ログ ===")
    for e in events:
        print(e)

    print()
    print("=== コマンド実行集計 ===")
    for k, v in sorted(cmd_counts.items(), key=lambda x: -x[1]):
        print("  {:45s}: {}回".format(k, v))

    print()
    print("=== 最終状態 ===")
    all_iids = []
    zones = _ZONE_ATTRS
    for pid in [0, 1]:
        p = s.players[pid]
        print("  P{}: hand={} deck={} shield={} mana={} bz={} grave={}".format(
            pid, len(p.hand), len(p.deck), len(p.shield_zone), len(p.mana_zone),
            len(p.battle_zone), len(p.graveyard)))
        for z in zones:
            for c in getattr(p, z, []):
                all_iids.append(c.instance_id)

    total_end = len(all_iids)
    dups = len(all_iids) - len(set(all_iids))
    print()
    print("=== ゾーン整合性 ===")
    print("  開始枚数: {}  終了枚数: {}  差分: {}".format(total_cards_start, total_end, total_end - total_cards_start))
    print("  instance_id重複: {}  game_over: {}".format(dups, s.game_over))
