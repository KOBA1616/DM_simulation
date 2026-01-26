#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential self-play environment with phased turns:
  - Untap
  - Draw
  - Mana charge (move one card from hand to mana)
  - Summon (move one card from hand to battle)
  - Attack (if battle creatures exist)

This uses engine methods when available (`resolve_action`), otherwise falls
back to manipulating `gi.state` directly. Start by running a single traced
game to validate behavior.
"""
import sys
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dm_ai_module as dm
from types import SimpleNamespace

# Simple instance id counter for created CardInstance objects
_NEXT_INSTANCE_ID = 1


def _alloc_instance_id():
    global _NEXT_INSTANCE_ID
    v = _NEXT_INSTANCE_ID
    _NEXT_INSTANCE_ID += 1
    return v


def _zone_ids(zone):
    try:
        return [getattr(c, 'instance_id', c) for c in zone]
    except Exception:
        return list(zone)


def untap_phase(state):
    for p in state.players:
        for c in getattr(p, 'battle_zone', []) or []:
            try:
                c.is_tapped = False
            except Exception:
                pass
        for c in getattr(p, 'mana_zone', []) or []:
            try:
                c.is_tapped = False
            except Exception:
                pass


def draw_phase(state, trace: bool = False):
    # NOTE: kept signature `state` for compatibility but prefer using GameInstance when available
    ap = state.active_player_id
    player = state.players[ap]
    if len(getattr(player, 'deck', [])) > 0:
        # Best-effort: create a CardInstance and append to hand.
        # If CardInstance isn't available (stub), use CardStub from dm_ai_module.
        try:
            ci = dm.CardInstance()
            ci.card_id = player.deck.pop(0)
            ci.instance_id = _alloc_instance_id()
            player.hand.append(ci)
            if trace:
                print(f"  draw_phase: appended CardInstance id={ci.instance_id}")
        except Exception:
            # Fallback to using CardStub if present, or append a simple placeholder
            try:
                card_id = player.deck.pop(0)
                if hasattr(dm, 'CardStub'):
                    # Prefer state.get_next_instance_id() if available
                    next_id_src = getattr(state, 'get_next_instance_id', None)
                    if callable(next_id_src):
                        iid = next_id_src()
                    else:
                        iid = _alloc_instance_id()
                    stub = dm.CardStub(card_id, iid)
                    player.hand.append(stub)
                    if trace:
                        print(f"  draw_phase: appended CardStub id={iid}")
                else:
                    player.hand.append(card_id)
                    if trace:
                        print(f"  draw_phase: appended raw card id={card_id}")
            except Exception:
                pass


def mana_charge_phase(state, trace: bool = False):
    # Accept either GameState or GameInstance.state
    # If caller passes GameInstance object instead of state, handle that
    if hasattr(state, 'state'):
        gi = state
        state = gi.state
    else:
        gi = None

    ap = state.active_player_id
    player = state.players[ap]
    if len(getattr(player, 'hand', [])) > 0:
        # Try engine action first when GameInstance is available
        try:
            if gi is not None and hasattr(gi, 'resolve_action'):
                card = player.hand[0]
                a = dm.Action()
                a.type = dm.ActionType.MANA_CHARGE
                if hasattr(card, 'instance_id'):
                    a.source_instance_id = card.instance_id
                if trace:
                    print(f"  mana_charge_phase: using resolve_action, source_instance={getattr(card,'instance_id',None)}")
                try:
                    from dm_toolkit.compat_wrappers import execute_action_compat
                    execute_action_compat(gi.state, a, None)
                except Exception:
                    try:
                        gi.resolve_action(a)
                    except Exception:
                        pass
                if trace:
                    print(f"  mana_charge_phase: after resolve_action hand={len(getattr(player,'hand',[]))} mana={len(getattr(player,'mana_zone',[]))} ids_hand={_zone_ids(getattr(player,'hand',[]))} ids_mana={_zone_ids(getattr(player,'mana_zone',[]))}")
                return
        except Exception:
            if trace:
                print("  mana_charge_phase: resolve_action failed, falling back")
            pass

        # Fallback: move one card from hand to mana_zone
        try:
            player.mana_zone.append(player.hand.pop(0))
            if trace:
                print(f"  mana_charge_phase: fallback moved card, mana_size={len(getattr(player,'mana_zone',[]))}")
        except Exception:
            try:
                _ = player.hand.pop(0)
            except Exception:
                pass


def summon_phase(state, trace: bool = False):
    # Accept GameInstance or GameState
    if hasattr(state, 'state'):
        gi = state
        state = gi.state
    else:
        gi = None

    ap = state.active_player_id
    player = state.players[ap]
    # If have mana and hand, attempt to play a card
    if len(getattr(player, 'mana_zone', [])) > 0 and len(getattr(player, 'hand', [])) > 0:
        card = player.hand[0]
        try:
            if gi is not None and hasattr(gi, 'resolve_action'):
                a = dm.Action()
                a.type = dm.ActionType.PLAY_CARD
                if hasattr(card, 'instance_id'):
                    a.source_instance_id = card.instance_id
                # If card has card_id attribute, include it
                if hasattr(card, 'card_id'):
                    a.card_id = getattr(card, 'card_id')
                if trace:
                    print(f"  summon_phase: using resolve_action, source_instance={getattr(card,'instance_id',None)}, card_id={getattr(card,'card_id',None)}")
                try:
                    from dm_toolkit.compat_wrappers import execute_action_compat
                    execute_action_compat(gi.state, a, None)
                except Exception:
                    try:
                        gi.resolve_action(a)
                    except Exception:
                        pass
                if trace:
                    print(f"  summon_phase: after resolve_action hand={len(getattr(player,'hand',[]))} battle={len(getattr(player,'battle_zone',[]))} ids_hand={_zone_ids(getattr(player,'hand',[]))} ids_battle={_zone_ids(getattr(player,'battle_zone',[]))}")
                return
        except Exception:
            if trace:
                print("  summon_phase: resolve_action failed, falling back")
            pass

        # Fallback: move card from hand to battle
        try:
            player.battle_zone.append(player.hand.pop(0))
            if trace:
                print(f"  summon_phase: fallback moved card to battle_zone, battle_size={len(getattr(player,'battle_zone',[]))}")
        except Exception:
            try:
                _ = player.hand.pop(0)
            except Exception:
                pass


def attack_phase(gi, card_db=None, trace: bool = False):
    state = gi.state
    ap = state.active_player_id
    target = 1 - ap

    # If we have a card_db, prefer using the engine's IntentGenerator to pick a legal attack action
    try:
        if card_db is not None:
            try:
                legal = dm.IntentGenerator.generate_legal_actions(state, card_db)
            except Exception:
                legal = None

            if trace:
                try:
                    print(f"  attack_phase: legal_actions count={len(legal) if legal is not None else 'None'}")
                except Exception:
                    print("  attack_phase: legal_actions present (len unknown)")

            if legal:
                if trace:
                    # show a few action summaries
                    try:
                        sample = []
                        for i, la in enumerate(legal[:6]):
                            try:
                                sample.append(getattr(la, 'type', None))
                            except Exception:
                                sample.append(str(la))
                        print(f"  attack_phase: legal sample types={sample}")
                    except Exception:
                        pass
                chosen = None
                for act in legal:
                    try:
                        t = getattr(act, 'type', None)
                        if t == getattr(dm, 'ActionType', None).ATTACK_PLAYER:
                            chosen = act
                            break
                        if hasattr(t, 'name') and t.name == 'ATTACK_PLAYER':
                            chosen = act
                            break
                    except Exception:
                        continue

                if chosen is not None:
                    try:
                        if trace:
                            try:
                                print(f"  attack_phase: using legal action: {chosen.to_string()}")
                            except Exception:
                                print(f"  attack_phase: using legal action: type={getattr(chosen,'type',None)} src={getattr(chosen,'source_instance_id',None)} tgt={getattr(chosen,'target_player',None)}")
                        try:
                            from dm_toolkit.compat_wrappers import execute_action_compat
                            execute_action_compat(gi.state, chosen, card_db if 'card_db' in locals() else None)
                        except Exception:
                            try:
                                gi.resolve_action(chosen)
                            except Exception:
                                pass
                        tp = state.players[target]
                        if trace:
                            print(f"  attack_phase: after resolve_action target_shields={len(getattr(tp,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))} winner={state.winner}")
                        return
                    except Exception:
                        if trace:
                            print("  attack_phase: resolve_action(legal) failed, falling back")
                        pass

        # Fallthrough to manual action construction
    except Exception:
        if trace:
            print("  attack_phase: intent-generation failed, falling back")
        pass

    try:
        def _action_dict(act):
            # Collect known fields defensively for logging
            fields = {}
            for k in ("type", "target_player", "source_instance_id", "card_id", "slot_index", "value1", "target_instance_id", "target_slot_index"):
                try:
                    fields[k] = getattr(act, k)
                except Exception:
                    fields[k] = None
            # Represent enum type nicely
            try:
                t = fields.get('type')
                if hasattr(t, 'name'):
                    fields['type_repr'] = t.name
                else:
                    fields['type_repr'] = str(t)
            except Exception:
                fields['type_repr'] = None
            return fields

        # Try engine flow commands (SET_ATTACK_SOURCE -> SET_ATTACK_PLAYER -> RESOLVE_BATTLE)
        try:
            if hasattr(dm, 'CommandSystem') and hasattr(dm, 'FlowCommand') and hasattr(dm, 'FlowType'):
                # pick attacker instance
                ap_player = state.players[ap]
                attacker = None
                for c in getattr(ap_player, 'battle_zone', []) or []:
                    attacker = c
                    break
                if attacker is not None and hasattr(attacker, 'instance_id'):
                    src_id = attacker.instance_id
                    if trace:
                        print(f"  attack_phase: attempting FlowCommand attack flow, source={src_id} target={target}")
                    try:
                        cmd_source = dm.FlowCommand(dm.FlowType.SET_ATTACK_SOURCE, src_id)
                        gi.state.execute_command(cmd_source)
                        cmd_target = dm.FlowCommand(dm.FlowType.SET_ATTACK_PLAYER, target)
                        gi.state.execute_command(cmd_target)
                        # After setting source/target, invoke RESOLVE_BATTLE via CommandDef
                        try:
                            # First try: ask engine to resolve a RESOLVE_BATTLE Action (preferred)
                            try:
                                try:
                                    a_res = dm.Action()
                                    a_res.type = dm.ActionType.RESOLVE_BATTLE
                                    a_res.source_instance_id = src_id
                                    # prefer targeting a shield instance when available
                                    try:
                                        tp_tmp = state.players[target]
                                        shield_tmp = _zone_ids(getattr(tp_tmp, 'shield_zone', []))
                                        if len(shield_tmp) > 0:
                                            a_res.target_instance_id = shield_tmp[-1]
                                        else:
                                            a_res.target_player = target
                                    except Exception:
                                        a_res.target_player = target
                                    if trace:
                                        print(f"    attempt: resolve_action(RESOLVE_BATTLE) source={a_res.source_instance_id} target_instance={getattr(a_res,'target_instance_id',None)} target_player={getattr(a_res,'target_player',None)}")
                                    try:
                                        from dm_toolkit.compat_wrappers import execute_action_compat
                                        execute_action_compat(gi.state, a_res, card_db if 'card_db' in locals() else None)
                                    except Exception:
                                        try:
                                            gi.resolve_action(a_res)
                                        except Exception:
                                            pass
                                    if trace:
                                        tp_post = state.players[target]
                                        print(f"    after resolve_action(RESOLVE_BATTLE) target_shields={len(getattr(tp_post,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp_post,'shield_zone',[]))}")
                                except Exception:
                                    if trace:
                                        print("    resolve_action(RESOLVE_BATTLE) not available or failed, falling back to pipeline/CommandDef")
                                    raise
                            except Exception:
                                pass
                            # First try: directly execute a minimal RESOLVE_BATTLE instruction
                            try:
                                inst = dm.Instruction(dm.InstructionOp.GAME_ACTION)
                                # Prefer targeting a shield instance if present; fall back to -1
                                try:
                                    tp_for_def = state.players[target]
                                    shield_ids_for_def = _zone_ids(getattr(tp_for_def, 'shield_zone', []))
                                    if len(shield_ids_for_def) > 0:
                                        defender_for_inst = shield_ids_for_def[-1]
                                    else:
                                        defender_for_inst = -1
                                except Exception:
                                    defender_for_inst = -1
                                inst.set_args({"type": "RESOLVE_BATTLE", "attacker": src_id, "defender": defender_for_inst})
                                if trace:
                                    try:
                                        print(f"    pipeline: RESOLVE_BATTLE attacker={src_id} defender={defender_for_inst} shield_ids={shield_ids_for_def if 'shield_ids_for_def' in locals() else None}")
                                    except Exception:
                                        pass
                                pe = dm.PipelineExecutor()
                                # Card DB mapping for PipelineExecutor
                                try:
                                    db_map = dm.CardRegistry.get_all_cards()
                                except Exception:
                                    db_map = card_db
                                pe.execute([inst], gi.state, db_map)
                                if trace:
                                    print("    pipeline: executed RESOLVE_BATTLE instruction via PipelineExecutor")
                            except Exception:
                                if trace:
                                    print("    pipeline: direct PipelineExecutor.execute failed, falling back to CommandDef path")

                            # Fallback: attempt to call CommandSystem with a CommandDef
                            try:
                                res_cmd = dm.CommandDef()
                                res_cmd.type = dm.CommandType.RESOLVE_BATTLE
                                # Ensure attacker is set so handler can compile RESOLVE_BATTLE
                                res_cmd.instance_id = src_id
                                # Use CommandSystem wrapper which expects a CommandDef
                                try:
                                    dm.CommandSystem.execute_command(gi.state, res_cmd, src_id, ap, {})
                                except Exception:
                                    # Fallback: try via GameState.command_system helper
                                    try:
                                        gi.state.command_system.execute_command(gi.state, res_cmd, src_id, ap, {})
                                    except Exception:
                                        pass
                            except Exception:
                                # If CommandDef not available or execution fails, attempt to resume pipeline
                                try:
                                    dm.EffectResolver.resume(gi.state, card_db, [])
                                except Exception:
                                    pass
                        except Exception:
                            # If CommandDef not available or execution fails, attempt to resume pipeline
                            try:
                                dm.EffectResolver.resume(gi.state, card_db, [])
                            except Exception:
                                pass
                        tp = state.players[target]
                        if trace:
                            print(f"  attack_phase: after flow target_shields={len(getattr(tp,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))} winner={state.winner}")
                        # If flow had effect, return
                        return
                    except Exception as e:
                        if trace:
                            print("  attack_phase: FlowCommand sequence failed, falling back", e)
                        pass
        except Exception:
            pass

        a = dm.Action()
        a.type = dm.ActionType.ATTACK_PLAYER
        a.target_player = target
        # Choose an attacker instance if available (engine often requires source_instance_id)
        ap_player = state.players[ap]
        attacker = None
        for c in getattr(ap_player, 'battle_zone', []) or []:
            attacker = c
            break
        if attacker is not None and hasattr(attacker, 'instance_id'):
            a.source_instance_id = attacker.instance_id
            if trace:
                print(f"  attack_phase: selected attacker instance={attacker.instance_id}")
        tp = state.players[target]
        pre_shields = len(getattr(tp, 'shield_zone', []))
        if hasattr(gi, 'resolve_action'):
            if trace:
                print(f"  attack_phase: using resolve_action attacking player {target} (pre_shields={pre_shields})")
                print(f"    action before resolve: {_action_dict(a)}")
            try:
                from dm_toolkit.compat_wrappers import execute_action_compat
                execute_action_compat(gi.state, a, card_db if 'card_db' in locals() else None)
            except Exception:
                try:
                    gi.resolve_action(a)
                except Exception:
                    pass
            post_shields = len(getattr(tp, 'shield_zone', []))
            if trace:
                print(f"  attack_phase: after resolve_action target_shields={post_shields} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))} winner={state.winner}")
            # If resolve_action didn't change shields, try a BREAK_SHIELD fallback
            if post_shields == pre_shields:
                try:
                    fb = dm.Action()
                    fb.type = dm.ActionType.BREAK_SHIELD
                    fb.target_player = target
                    # Try a plain BREAK_SHIELD first
                    if hasattr(gi, 'execute_action'):
                        if trace:
                            print("  attack_phase: resolve_action had no effect, trying execute_action(BREAK_SHIELD)")
                            print(f"    fallback action: {_action_dict(fb)}")
                            try:
                                from dm_toolkit.compat_wrappers import execute_action_compat
                                execute_action_compat(gi.state, fb, card_db if 'card_db' in locals() else None)
                            except Exception:
                                try:
                                    gi.execute_action(fb)
                                except Exception:
                                    pass
                    elif hasattr(gi, 'resolve_action'):
                        if trace:
                            print("  attack_phase: resolve_action had no effect, trying resolve_action(BREAK_SHIELD)")
                            print(f"    fallback action: {_action_dict(fb)}")
                                try:
                                    from dm_toolkit.compat_wrappers import execute_action_compat
                                    execute_action_compat(gi.state, fb, card_db if 'card_db' in locals() else None)
                                except Exception:
                                    try:
                                        gi.resolve_action(fb)
                                    except Exception:
                                        pass

                    # If still no effect, attempt to target a specific shield instance
                    post_shields2 = len(getattr(tp, 'shield_zone', []))
                    if post_shields2 == pre_shields and len(getattr(tp, 'shield_zone', [])) > 0:
                        shield_ids = _zone_ids(getattr(tp, 'shield_zone', []))
                        target_shield = shield_ids[-1]
                        fb2 = dm.Action()
                        fb2.type = dm.ActionType.BREAK_SHIELD
                        fb2.target_player = target
                        # Some engines expect the shield instance in slot_index/value1
                        try:
                            # Set engine-preferred target fields when available
                            if hasattr(fb2, 'target_instance_id'):
                                fb2.target_instance_id = target_shield
                            # target_slot_index likely expects a slot index (0..n-1), not an instance id
                            if hasattr(fb2, 'target_slot_index'):
                                fb2.target_slot_index = len(getattr(tp, 'shield_zone', [])) - 1
                            # Keep legacy fallback names if present
                            if hasattr(fb2, 'slot_index'):
                                fb2.slot_index = target_shield
                        except Exception:
                            pass
                        if hasattr(gi, 'execute_action'):
                            if trace:
                                print("  attack_phase: trying execute_action(BREAK_SHIELD) with shield instance")
                                print(f"    fallback action 2: {_action_dict(fb2)}")
                            try:
                                from dm_toolkit.compat_wrappers import execute_action_compat
                                execute_action_compat(gi.state, fb2, card_db if 'card_db' in locals() else None)
                            except Exception:
                                try:
                                    gi.execute_action(fb2)
                                except Exception:
                                    try:
                                        gi.resolve_action(fb2)
                                    except Exception:
                                        pass
                        elif hasattr(gi, 'resolve_action'):
                            if trace:
                                print("  attack_phase: trying resolve_action(BREAK_SHIELD) with shield instance")
                                print(f"    fallback action 2: {_action_dict(fb2)}")
                            try:
                                from dm_toolkit.compat_wrappers import execute_action_compat
                                execute_action_compat(gi.state, fb2, card_db if 'card_db' in locals() else None)
                            except Exception:
                                try:
                                    gi.resolve_action(fb2)
                                except Exception:
                                    pass

                    if trace:
                        print(f"  attack_phase: after fallback target_shields={len(getattr(tp,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))} winner={state.winner}")
                except Exception:
                    if trace:
                        print("  attack_phase: fallback BREAK_SHIELD failed")
            return
    except Exception:
        if trace:
            print("  attack_phase: resolve_action failed, falling back")
        pass

    # Fallback: direct shield removal
    tp = state.players[target]
    if len(getattr(tp, 'shield_zone', [])) > 0:
        tp.shield_zone.pop()
    else:
        state.winner = dm.GameResult.P1_WIN if ap == 0 else dm.GameResult.P2_WIN


def play_one_full(seed: int | None = None, trace: bool = False) -> int:
    if seed is not None:
        random.seed(seed)

    card_db = dm.JsonLoader.load_cards('data/cards.json')
    gi = dm.GameInstance(0, card_db)
    deck = [1] * 40
    gi.state.set_deck(0, deck[:])
    gi.state.set_deck(1, deck[:])
    gi.start_game()

    # Randomize starting player
    gi.state.active_player_id = random.choice([0, 1])

    turn = 0
    while gi.state.winner == dm.GameResult.NONE and turn < 200:
        turn += 1
        if trace:
            print(f"\n=== Turn {turn} (active P{gi.state.active_player_id}) ===")

        # Untap
        untap_phase(gi.state)
        if trace:
            pass

        # Draw
        draw_phase(gi.state, trace)
        if trace:
            p = gi.state.players[gi.state.active_player_id]
            print(f"  After draw: hand={len(getattr(p,'hand',[]))} deck={len(getattr(p,'deck',[]))}")

        # Mana charge
        mana_charge_phase(gi.state, trace)
        if trace:
            p = gi.state.players[gi.state.active_player_id]
            print(f"  Mana zone size: {len(getattr(p,'mana_zone',[]))}")

        # Summon
        summon_phase(gi.state, trace)
        if trace:
            p = gi.state.players[gi.state.active_player_id]
            print(f"  Battle zone size: {len(getattr(p,'battle_zone',[]))}")

        # Attack
        attack_phase(gi, card_db, trace)
        if trace:
            tp = gi.state.players[1 - gi.state.active_player_id]
            print(f"  Opponent shields: {len(getattr(tp,'shield_zone',[]))}")

        # Alternate active player
        gi.state.active_player_id = 1 - gi.state.active_player_id

        # Optional: increment turn counter every full round
        if turn % 2 == 0:
            gi.state.turn_number += 1

    return int(gi.state.winner)


def minimal_attack_test(seed: int | None = None, trace: bool = False) -> int:
    if seed is not None:
        random.seed(seed)
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    gi = dm.GameInstance(0, card_db)
    gi.start_game()

    # Ensure we have a single attacker and a single shield to test RESOLVE_BATTLE path
    try:
        atk = dm.CardInstance()
        atk.card_id = 1
        atk.instance_id = _alloc_instance_id()
    except Exception:
        try:
            atk = dm.CardStub(1, _alloc_instance_id())
        except Exception:
            atk = SimpleNamespace(card_id=1, instance_id=_alloc_instance_id())

    try:
        sh = dm.CardInstance()
        sh.card_id = 2
        sh.instance_id = _alloc_instance_id()
    except Exception:
        try:
            sh = dm.CardStub(2, _alloc_instance_id())
        except Exception:
            sh = SimpleNamespace(card_id=2, instance_id=_alloc_instance_id())

    # Place attacker and shield
    gi.state.players[0].battle_zone.clear()
    gi.state.players[1].shield_zone.clear()
    gi.state.players[0].battle_zone.append(atk)
    gi.state.players[1].shield_zone.append(sh)
    gi.state.active_player_id = 0

    if trace:
        print(f"Mini test: attacker.instance={getattr(atk,'instance_id',None)} shield.instance={getattr(sh,'instance_id',None)}")

    # Run attack_phase once
    attack_phase(gi, card_db, trace)

    tp = gi.state.players[1]
    if trace:
        print(f"Mini test: after attack target_shields={len(getattr(tp,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))}")

    # If RESOLVE_BATTLE had no effect, try issuing a direct BREAK_SHIELD action targeting the shield instance
    try:
        post = len(getattr(tp, 'shield_zone', []))
        if post > 0:
            # attempt BREAK_SHIELD on the shield instance
            try:
                fb = dm.Action()
                fb.type = dm.ActionType.BREAK_SHIELD
                if hasattr(fb, 'target_instance_id'):
                    fb.target_instance_id = getattr(sh, 'instance_id', None)
                else:
                    fb.target_player = 1
                if trace:
                    print(f"Mini test: attempting resolve_action(BREAK_SHIELD) target_instance={getattr(sh,'instance_id',None)}")
                try:
                    from dm_toolkit.compat_wrappers import execute_action_compat
                    execute_action_compat(gi.state, fb, card_db if 'card_db' in locals() else None)
                except Exception:
                    try:
                        gi.resolve_action(fb)
                    except Exception:
                        pass
                if trace:
                    print(f"Mini test: after BREAK_SHIELD target_shields={len(getattr(tp,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))}")
            except Exception:
                if trace:
                    print("Mini test: resolve_action(BREAK_SHIELD) failed")
    except Exception:
        pass

    return int(gi.state.winner)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--mini-test', action='store_true', help='Run a minimal single-attack test')
    args = parser.parse_args()
    if args.mini_test:
        res = minimal_attack_test(args.seed, args.trace)
        print('\nMini test result enum int:', res)
    else:
        res = play_one_full(args.seed, args.trace)
        print('\nResult enum int:', res)


if __name__ == '__main__':
    main()


def minimal_attack_test(seed: int | None = None, trace: bool = False) -> int:
    if seed is not None:
        random.seed(seed)
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    gi = dm.GameInstance(0, card_db)
    gi.start_game()

    # Ensure we have a single attacker and a single shield to test RESOLVE_BATTLE path
    try:
        atk = dm.CardInstance()
        atk.card_id = 1
        atk.instance_id = _alloc_instance_id()
    except Exception:
        try:
            atk = dm.CardStub(1, _alloc_instance_id())
        except Exception:
            atk = SimpleNamespace(card_id=1, instance_id=_alloc_instance_id())

    try:
        sh = dm.CardInstance()
        sh.card_id = 2
        sh.instance_id = _alloc_instance_id()
    except Exception:
        try:
            sh = dm.CardStub(2, _alloc_instance_id())
        except Exception:
            sh = SimpleNamespace(card_id=2, instance_id=_alloc_instance_id())

    # Place attacker and shield
    gi.state.players[0].battle_zone.clear()
    gi.state.players[1].shield_zone.clear()
    gi.state.players[0].battle_zone.append(atk)
    gi.state.players[1].shield_zone.append(sh)
    gi.state.active_player_id = 0

    if trace:
        print(f"Mini test: attacker.instance={getattr(atk,'instance_id',None)} shield.instance={getattr(sh,'instance_id',None)}")

    # Run attack_phase once
    attack_phase(gi, card_db, trace)

    tp = gi.state.players[1]
    if trace:
        print(f"Mini test: after attack target_shields={len(getattr(tp,'shield_zone',[]))} shield_ids={_zone_ids(getattr(tp,'shield_zone',[]))}")

    return int(gi.state.winner)
