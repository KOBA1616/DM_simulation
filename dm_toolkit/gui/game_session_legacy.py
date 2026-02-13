# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Callable, Dict, Tuple
import random
import os
import sys
import json

from dm_toolkit.dm_types import GameState, CardDB
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command

# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from training.ai_player import AIPlayer
except ImportError:
    AIPlayer = None

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None


class GameSession:
    DEFAULT_DECK: List[int] = []

    def __init__(self,
                 callback_update_ui: Optional[Callable[[], None]] = None,
                 callback_log: Optional[Callable[[str], None]] = None,
                 callback_input_request: Optional[Callable[[], None]] = None,
                 callback_action_executed: Optional[Callable[[Any], None]] = None):
        self.callback_update_ui = callback_update_ui or (lambda: None)
        self.callback_log = callback_log or (lambda m: None)
        self.callback_input_request = callback_input_request or (lambda: None)
        self.callback_action_executed = callback_action_executed or (lambda a: None)

        self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
        self.is_running = False
        self.is_processing = False
        self.last_action = None
        self.card_db: CardDB = {}
        self.gs: Optional[GameState] = None
        self.ai_player = None

    def _load_latest_ai(self):
        if not AIPlayer:
            return
        try:
            models_dir = os.path.join(project_root, "models")
            if not os.path.exists(models_dir):
                self.callback_log("Models directory not found. Using random AI.")
                return

            files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pth')]
            if not files:
                self.callback_log("No trained models found. Using random AI.")
                return

            latest_model = max(files, key=os.path.getmtime)
            self.callback_log(f"Loading AI Model: {os.path.basename(latest_model)}")
            self.ai_player = AIPlayer(latest_model, device='cpu')
        except Exception as e:
            self.callback_log(f"Failed to load AI: {e}")
            self.ai_player = None

    def initialize_game(self, card_db: CardDB, seed: int = 42) -> None:
        self.card_db = card_db
        if dm_ai_module:
            try:
                if hasattr(dm_ai_module, 'CardDatabase') and hasattr(dm_ai_module.CardDatabase, '_cards'):
                    if isinstance(self.card_db, dict) and not getattr(dm_ai_module.CardDatabase, '_loaded', False):
                        try:
                            dm_ai_module.CardDatabase._cards.update(self.card_db)
                            dm_ai_module.CardDatabase._loaded = True
                            self.callback_log(" injected card_db into dm_ai_module.CardDatabase")
                        except Exception:
                            pass
            except Exception as e:
                self.callback_log(f"Warning: Failed to inject CardDatabase: {e}")

            self.gs = dm_ai_module.GameState(seed)
            self.gs.setup_test_duel()

            deck0 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else []
            deck1 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else []
            if not deck0:
                try:
                    deck0 = self._build_deck_from_card_db(self.card_db)
                except Exception:
                    deck0 = []
            if not deck1:
                try:
                    deck1 = self._build_deck_from_card_db(self.card_db)
                except Exception:
                    deck1 = []

            try:
                self.gs.set_deck(0, deck0)
                self.gs.set_deck(1, deck1)
            except Exception:
                pass

            try:
                self.callback_log(f"P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
                self.callback_log(f"P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")
            except Exception:
                pass

            applied = False
            if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
                native_db = None
                try:
                    if hasattr(dm_ai_module, 'JsonLoader'):
                        native_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                        self.callback_log("Loaded native CardDatabase via JsonLoader")
                except Exception as e:
                    self.callback_log(f"Warning: JsonLoader failed: {e}")

                if native_db is None:
                    try:
                        if hasattr(dm_ai_module, 'CardRegistry') and hasattr(dm_ai_module.CardRegistry, 'get_all_definitions'):
                            native_db = dm_ai_module.CardRegistry.get_all_definitions()
                            self.callback_log("Loaded native CardDatabase via CardRegistry.get_all_definitions")
                    except Exception as e:
                        self.callback_log(f"Warning: CardRegistry lookup failed: {e}")

                if native_db is None:
                    native_db = self.card_db

                try:
                    if native_db is not self.card_db:
                        self.card_db = native_db

                    dm_ai_module.PhaseManager.start_game(self.gs, native_db)
                    self.callback_log("start_game executed successfully")
                    applied = True
                except Exception as e:
                    self.callback_log(f"Warning: start_game failed: {e}")

                try:
                    dump = EngineCompat.dump_state_debug(self.gs, max_samples=1)
                    self.callback_log(f"Debug: post-start_game native_present={dump.get('native_present')}")
                    if not dump.get('native_present'):
                        try:
                            attrs = [a for a in dir(self.gs) if not a.startswith('__')][:40]
                            self.callback_log(f"Debug: GameState attrs sample: {attrs}")
                        except Exception:
                            pass
                except Exception:
                    pass

            if not applied:
                try:
                    self._fallback_apply_shields_and_draw()
                except Exception as e:
                    self.callback_log(f"Fallback during initialize_game failed: {e}")

            try:
                self.callback_log(f"After start_game - P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
                self.callback_log(f"After start_game - P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")
            except Exception:
                pass
        else:
            self.gs = None

        self.callback_log(tr("Game Initialized via Session"))
        self.callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        if not dm_ai_module:
            return

        seed = random.randint(0, 100000)
        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        deck0 = p0_deck if p0_deck else list(self.DEFAULT_DECK)
        deck1 = p1_deck if p1_deck else list(self.DEFAULT_DECK)

        if not deck0:
            try:
                deck0 = self._build_deck_from_card_db(self.card_db)
            except Exception:
                deck0 = []
        if not deck1:
            try:
                deck1 = self._build_deck_from_card_db(self.card_db)
            except Exception:
                deck1 = []

        try:
            self.gs.set_deck(0, deck0)
            self.gs.set_deck(1, deck1)
        except Exception:
            pass

        try:
            self.callback_log(f"P0 deck size (post-set): {len(self.gs.players[0].deck)}")
            self.callback_log(f"P1 deck size (post-set): {len(self.gs.players[1].deck)}")
        except Exception:
            pass

        try:
            has_pm = hasattr(dm_ai_module, 'PhaseManager')
            has_start = has_pm and hasattr(dm_ai_module.PhaseManager, 'start_game')
        except Exception:
            has_pm = False
            has_start = False
        try:
            self.callback_log(f"Debug: PhaseManager present={has_pm}, has start_game={has_start}")
        except Exception:
            pass

        if has_pm and has_start:
            native_db = None
            try:
                if hasattr(dm_ai_module, 'JsonLoader'):
                    native_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                    self.callback_log("Loaded native CardDatabase via JsonLoader")
            except Exception as e:
                self.callback_log(f"Warning: JsonLoader failed: {e}")

            if native_db is None:
                try:
                    if hasattr(dm_ai_module, 'CardRegistry') and hasattr(dm_ai_module.CardRegistry, 'get_all_definitions'):
                        native_db = dm_ai_module.CardRegistry.get_all_definitions()
                        self.callback_log("Loaded native CardDatabase via CardRegistry.get_all_definitions")
                except Exception as e:
                    self.callback_log(f"Warning: CardRegistry lookup failed: {e}")

            if native_db is None:
                native_db = self.card_db

            try:
                if native_db is not self.card_db:
                    self.card_db = native_db

                dm_ai_module.PhaseManager.start_game(self.gs, native_db)
                applied = True
            except Exception as e:
                self.callback_log(f"Warning: start_game failed during reset_game: {e}")
                applied = False
            else:
                try:
                    dump = EngineCompat.dump_state_debug(self.gs, max_samples=1)
                    self.callback_log(f"Debug: post-start_game native_present={dump.get('native_present')}")
                    if not dump.get('native_present'):
                        try:
                            attrs = [a for a in dir(self.gs) if not a.startswith('__')][:40]
                            self.callback_log(f"Debug: GameState attrs sample: {attrs}")
                        except Exception:
                            pass
                except Exception:
                    pass
        else:
            try:
                if not globals().get('applied', False):
                    self._fallback_apply_shields_and_draw()
            except Exception as e:
                self.callback_log(f"Fallback deck setup failed: {e}")

        try:
            self.callback_log(f"After start_game - P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
            self.callback_log(f"After start_game - P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")
        except Exception:
            pass

        self.callback_log(tr("Game Reset"))
        self.callback_update_ui()

    def _fallback_apply_shields_and_draw(self) -> None:
        try:
            self.callback_log(f"Pre-fallback deck sizes: P0={len(self.gs.players[0].deck)}, P1={len(self.gs.players[1].deck)}")
        except Exception:
            pass

        if dm_ai_module and hasattr(dm_ai_module, 'DevTools') and hasattr(dm_ai_module, 'Zone'):
            for pid in (0, 1):
                p = self.gs.players[pid]
                for _ in range(5):
                    if not p.deck:
                        break
                    iid = p.deck[-1].instance_id
                    try:
                        dm_ai_module.DevTools.move_cards(self.gs, iid, dm_ai_module.Zone.DECK, dm_ai_module.Zone.SHIELD)
                    except Exception:
                        pass
                for _ in range(5):
                    if not p.deck:
                        break
                    iid = p.deck[-1].instance_id
                    try:
                        dm_ai_module.DevTools.move_cards(self.gs, iid, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND)
                    except Exception:
                        pass
        else:
            for pid in (0, 1):
                p = self.gs.players[pid]
                for _ in range(5):
                    if not p.deck:
                        break
                    card = p.deck.pop()
                    try:
                        p.shield_zone.append(card)
                    except Exception:
                        pass
                for _ in range(5):
                    if not p.deck:
                        break
                    card = p.deck.pop()
                    try:
                        p.hand.append(card)
                    except Exception:
                        pass

        try:
            self.callback_log(f"Post-fallback deck sizes: P0={len(self.gs.players[0].deck)}, P1={len(self.gs.players[1].deck)}")
        except Exception:
            pass
        self.callback_log(tr("Fallback: performed minimal deck setup (shields+draw)"))

    def _build_deck_from_card_db(self, db: CardDB, count: int = 30) -> List[int]:
        ids: List[int] = []
        try:
            if isinstance(db, dict):
                ids = [int(k) for k in db.keys()]
            elif isinstance(db, list):
                for entry in db:
                    try:
                        if isinstance(entry, dict) and 'id' in entry:
                            ids.append(int(entry['id']))
                    except Exception:
                        continue
        except Exception:
            ids = []

        if not ids:
            return []

        deck: List[int] = []
        i = 0
        while len(deck) < count:
            deck.append(ids[i % len(ids)])
            i += 1

        return deck

    def set_player_mode(self, player_id: int, mode: str):
        self.player_modes[player_id] = mode

    def get_player_mode(self, player_id: int) -> str:
        return self.player_modes.get(player_id, 'AI')

    def execute_action(self, action: Any) -> None:
        if not self.gs:
            return

        self.last_action = action

        try:
            if hasattr(action, 'to_dict'):
                raw_action = action.to_dict()
            else:
                raw_action = action

            cmd_dict = ensure_executable_command(raw_action)
        except Exception as e:
            self.callback_log(tr("Command Conversion Error: {error}").format(error=e))
            return

        try:
            if cmd_dict.get('type') == 'MANA_CHARGE' and 'instance_id' in cmd_dict:
                print(f"[DEBUG] Creating ManaChargeCommand for instance_id={cmd_dict['instance_id']}")
                if dm_ai_module and hasattr(dm_ai_module, 'ManaChargeCommand'):
                    cpp_cmd = dm_ai_module.ManaChargeCommand(int(cmd_dict['instance_id']))
                    print(f"[DEBUG] Created C++ ManaChargeCommand: {cpp_cmd}")
                    self.gs.execute_command(cpp_cmd)
                    print("[DEBUG] ManaChargeCommand executed")
                else:
                    print("[DEBUG] Using EngineCompat.ExecuteCommand for MANA_CHARGE")
                    EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)
            elif cmd_dict.get('type') == 'PLAY_FROM_ZONE' or cmd_dict.get('legacy_original_type') == 'DECLARE_PLAY':
                print(f"[DEBUG] Creating PlayCard Command for instance_id={cmd_dict.get('instance_id')}")
                if dm_ai_module and hasattr(dm_ai_module, 'PlayCardCommand'):
                    instance_id = int(cmd_dict.get('instance_id') or cmd_dict.get('source_instance_id', -1))
                    cpp_cmd = dm_ai_module.PlayCardCommand(instance_id)
                    print(f"[DEBUG] Created C++ PlayCardCommand: {cpp_cmd}")
                    self.gs.execute_command(cpp_cmd)
                    print("[DEBUG] PlayCardCommand executed")
                else:
                    print("[DEBUG] Using EngineCompat.ExecuteCommand for PLAY_FROM_ZONE")
                    EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)
            elif cmd_dict.get('type') == 'PASS':
                if dm_ai_module and hasattr(dm_ai_module, 'PassCommand'):
                    cpp_cmd = dm_ai_module.PassCommand()
                    self.gs.execute_command(cpp_cmd)
                else:
                    EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)
            else:
                EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)

            log_str = f"P{EngineCompat.get_active_player_id(self.gs)} {tr('Action')}: {cmd_dict.get('type', 'UNKNOWN')}"
            if 'to_zone' in cmd_dict:
                log_str += f" -> {cmd_dict['to_zone']}"
            self.callback_log(log_str)

            if self.callback_action_executed:
                self.callback_action_executed(cmd_dict)

        except RuntimeError as e:
            self.callback_log(tr("Engine Error: {error}").format(error=e))
        except Exception as e:
            self.callback_log(tr("Execution Error: {error}").format(error=e))

        if self.check_and_handle_input_wait():
            return

        self.callback_update_ui()

    def check_and_handle_input_wait(self) -> bool:
        if not self.gs:
            return False
        try:
            waiting = getattr(self.gs, 'waiting_for_user_input', False)
        except Exception:
            waiting = False
        if not waiting:
            return False

        if self.is_running:
            self.is_running = False

        if self.callback_input_request:
            try:
                self.callback_input_request()
            except Exception:
                pass

        self.callback_update_ui()
        return True

    def resume_from_input(self, result: Any):
        EngineCompat.EffectResolver_resume(self.gs, self.card_db, result)
        self.step_phase()

    def step_phase(self) -> None:
        if self.is_processing:
            return
        self.is_processing = True
        try:
            if self.check_and_handle_input_wait():
                return

            if self.gs and dm_ai_module and hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'check_game_over'):
                try:
                    try:
                        is_over, winner = EngineCompat.PhaseManager_check_game_over(self.gs)
                        if is_over:
                            self.gs.game_over = True
                            try:
                                self.callback_log(tr("Game Over - Winner: {winner}").format(winner=winner if winner is not None else getattr(self.gs, 'winner', None)))
                            except Exception:
                                pass
                            return
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self.callback_log(f"Warning: check_game_over failed: {e}")
                    except Exception:
                        pass
            elif self.gs and getattr(self.gs, 'game_over', False):
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (self.player_modes.get(active_pid) == 'Human')

            try:
                waiting = getattr(self.gs, 'waiting_for_user_input', False)
            except Exception:
                waiting = False
            try:
                pending_count = self.gs.get_pending_effect_count()
            except Exception:
                pending_count = 'NA'
            try:
                current_phase = EngineCompat.get_current_phase(self.gs)
            except Exception:
                current_phase = 'NA'
            try:
                self.callback_log(f"Debug: active_pid={active_pid}, waiting={waiting}, pending_effects={pending_count}, phase={current_phase}")
            except Exception:
                pass

            from dm_toolkit import commands_v2
            cmds = commands_v2.generate_legal_commands(self.gs, self.card_db)

            try:
                max_dump = 30
                total = len(cmds)
                if total == 0:
                    self.callback_log(f"Debug: legal commands total=0 for P{active_pid}")
                else:
                    if total <= max_dump:
                        self.callback_log(f"Debug: legal commands total={total} for P{active_pid} (full dump)")
                        for i, c in enumerate(cmds):
                            try:
                                tname = type(c).__name__
                                raw = None
                                try:
                                    raw = c.to_dict()
                                except Exception:
                                    raw = str(c)
                                src = None
                                if isinstance(raw, dict):
                                    src = raw.get('source_instance_id', raw.get('instance_id', ''))
                                self.callback_log(f"Debug: Legal[{i}] type={tname} repr={raw} src={src}")
                            except Exception:
                                try:
                                    self.callback_log(f"Debug: Legal[{i}] raw_fallback={str(c)}")
                                except Exception:
                                    pass
                    else:
                        self.callback_log(f"Debug: legal commands count={total} for P{active_pid} (showing first {max_dump})")
                        for i, c in enumerate(cmds[:max_dump]):
                            try:
                                tname = type(c).__name__
                                raw = None
                                try:
                                    raw = c.to_dict()
                                except Exception:
                                    raw = str(c)
                                src = None
                                if isinstance(raw, dict):
                                    src = raw.get('source_instance_id', raw.get('instance_id', ''))
                                self.callback_log(f"Debug: Legal[{i}] type={tname} repr={raw} src={src}")
                            except Exception:
                                try:
                                    self.callback_log(f"Debug: Legal[{i}] raw_fallback={str(c)}")
                                except Exception:
                                    pass
                    if total > max_dump:
                        try:
                            self.callback_log(f"Debug: (truncated, total={total})")
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                self.callback_log(f"Debug: legal commands count={len(cmds)} for P{active_pid}")
            except Exception:
                pass

            try:
                if len(cmds) == 1:
                    only = cmds[0]
                    try:
                        od = only.to_dict()
                    except Exception:
                        od = {}
                    if od.get('type') == 'PASS':
                        try:
                            p = self.gs.players[active_pid]
                            try:
                                gs_attrs = {}
                                for attr in ('phase', 'phase_name', 'turn', 'active_player', 'current_player'):
                                    try:
                                        gs_attrs[attr] = getattr(self.gs, attr)
                                    except Exception:
                                        gs_attrs[attr] = None
                                self.callback_log(f"Debug PASS-only -> GameState attrs: {gs_attrs}")
                            except Exception:
                                pass
                            def ids_from_zone(zone):
                                try:
                                    return [getattr(c, 'instance_id', None) or getattr(c, 'id', None) for c in list(zone)[:10]]
                                except Exception:
                                    try:
                                        return [str(x) for x in list(zone)[:10]]
                                    except Exception:
                                        return []

                            deck_cnt = len(getattr(p, 'deck', []))
                            hand_cnt = len(getattr(p, 'hand', []))
                            mana_cnt = len(getattr(p, 'mana_zone', [])) if hasattr(p, 'mana_zone') else 'NA'
                            shield_cnt = len(getattr(p, 'shield_zone', []))

                            self.callback_log(f"Debug PASS-only -> P{active_pid} zones: deck_count={deck_cnt}, hand_count={hand_cnt}, mana_count={mana_cnt}, shield_count={shield_cnt}")
                            try:
                                self.callback_log(f"Debug PASS-only -> P{active_pid} deck top ids: {ids_from_zone(getattr(p, 'deck', []))}")
                                self.callback_log(f"Debug PASS-only -> P{active_pid} hand ids: {ids_from_zone(getattr(p, 'hand', []))}")
                                if hasattr(p, 'mana_zone'):
                                    self.callback_log(f"Debug PASS-only -> P{active_pid} mana ids: {ids_from_zone(getattr(p, 'mana_zone', []))}")
                                self.callback_log(f"Debug PASS-only -> P{active_pid} shields ids: {ids_from_zone(getattr(p, 'shield_zone', []))}")

                                def sample_zone(zone, n=3):
                                    out = []
                                    try:
                                        lst = list(zone)
                                    except Exception:
                                        try:
                                            lst = list(getattr(zone, '__iter__', lambda: [])() )
                                        except Exception:
                                            lst = []
                                    for i, c in enumerate(lst[:n]):
                                        try:
                                            tname = type(c).__name__
                                            r = repr(c)
                                            if len(r) > 200:
                                                r = r[:200] + '...'
                                            out.append((i, tname, r))
                                        except Exception as e:
                                            out.append((i, 'ERROR', str(e)))
                                    return out

                                try:
                                    self.callback_log(f"Debug PASS-only -> P{active_pid} deck samples: {sample_zone(getattr(p, 'deck', []), 3)}")
                                    self.callback_log(f"Debug PASS-only -> P{active_pid} hand samples: {sample_zone(getattr(p, 'hand', []), 3)}")
                                    if hasattr(p, 'mana_zone'):
                                        self.callback_log(f"Debug PASS-only -> P{active_pid} mana samples: {sample_zone(getattr(p, 'mana_zone', []), 3)}")
                                    self.callback_log(f"Debug PASS-only -> P{active_pid} shield samples: {sample_zone(getattr(p, 'shield_zone', []), 3)}")
                                except Exception:
                                    pass
                                try:
                                    try:
                                        dump = EngineCompat.dump_state_debug(self.gs, max_samples=3)
                                        self.callback_log(f"Debug PASS-only -> EngineCompat dump: {dump}")
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        except Exception:
                            pass
            except Exception:
                pass

            if is_human:
                resolve_cmds = []
                for c in cmds:
                    try:
                        d = c.to_dict()
                    except Exception:
                        d = {}
                    if d.get('type') == 'RESOLVE_EFFECT':
                        resolve_cmds.append((c, d))

                if len(resolve_cmds) > 1:
                    pass

                return

            if not cmds:
                self.callback_update_ui()
                return
            
            best_cmd = None
            try:
                for c in cmds:
                    try:
                        d = c.to_dict()
                    except Exception:
                        d = {}
                    t = d.get('type') if isinstance(d, dict) else None
                    if t == 'NONE' and d.get('legacy_warning'):
                        continue
                    if t is None:
                        try:
                            s = str(d)
                            if 'PASS' not in s:
                                best_cmd = c
                                break
                        except Exception:
                            continue
                    else:
                        if t != 'PASS':
                            best_cmd = c
                            break
            except Exception:
                best_cmd = None

            if best_cmd is None:
                best_cmd = cmds[0]

            if self.ai_player:
                try:
                    valid_indices = []
                    cmd_map = {}
                    encoder = self.ai_player.action_encoder

                    for cmd in cmds:
                        idx = encoder.encode_action(cmd, self.gs, active_pid)
                        if idx != -1:
                            valid_indices.append(idx)
                            cmd_map[idx] = cmd
                        else:
                            try:
                                d = None
                                try:
                                    d = cmd.to_dict()
                                except Exception:
                                    d = str(cmd)
                                self.callback_log(f"Debug: encode_action -> -1 for cmd: {d}")
                            except Exception:
                                pass

                    if valid_indices:
                        ai_cmd = self.ai_player.get_action(self.gs, active_pid, valid_indices)
                        ai_idx = encoder.encode_action(ai_cmd, self.gs, active_pid)
                        if ai_idx in cmd_map:
                            best_cmd = cmd_map[ai_idx]
                        else:
                            best_cmd = ai_cmd

                except Exception as e:
                    try:
                        self.callback_log(f"AI Error: {e}")
                    except Exception:
                        pass

            if best_cmd:
                self.execute_action(best_cmd)

            self.callback_update_ui()

        finally:
            self.is_processing = False

    def generate_legal_commands(self) -> List[Any]:
        if not self.gs:
            return []
        from dm_toolkit import commands_v2
        return commands_v2.generate_legal_commands(self.gs, self.card_db)

    def is_game_over(self) -> bool:
        return self.gs.game_over if self.gs else False
