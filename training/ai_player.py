
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.tokenization import StateTokenizer, ActionEncoder
from dm_ai_module import GameCommand
from dm_toolkit.action_to_command import map_action
from dm_toolkit.training.command_compat import normalize_to_command as _cc_normalize
import dm_ai_module as dm

class AIPlayer:
    def __init__(self, model_path: str, device='cpu', config=None):
        self.device = device
        # Default Config (matches train_simple.py)
        self.config = config or {
            'vocab_size': 10000,
            'action_dim': 600,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'max_len': 200
        }

        self.tokenizer = StateTokenizer(max_len=self.config.get('max_len', 200))

        # Ensure action_dim matches CommandEncoder schema when available
        try:
            dm_mod = __import__('dm_ai_module')
            if hasattr(dm_mod, 'CommandEncoder'):
                action_dim = int(getattr(dm_mod, 'CommandEncoder').TOTAL_COMMAND_SIZE)
            else:
                action_dim = int(self.config.get('action_dim', 600))
        except Exception:
            action_dim = int(self.config.get('action_dim', 600))

        self.action_encoder = ActionEncoder(action_dim=action_dim)

        self.model = DuelTransformer(
            vocab_size=self.config['vocab_size'],
            action_dim=self.config['action_dim'] if 'action_dim' in self.config else action_dim,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            max_len=self.config['max_len']
        ).to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
            except RuntimeError as e:
                print(f"Warning: Model architecture mismatch. {e}")
                print("Proceeding with initialized weights (random).")
        else:
            print(f"Warning: Model not found at {model_path}. Using random weights.")

        self.model.eval()

    def get_action(self, game_state, player_id: int, valid_indices: list[int] = None) -> GameCommand:
        # 1. Tokenize
        state_tokens = self.tokenizer.encode_state(game_state, player_id)

        state_tensor = torch.tensor([state_tokens], dtype=torch.long).to(self.device)
        padding_mask = (state_tensor == 0)

        phase = int(getattr(game_state, 'current_phase', 0))
        phase_ids = torch.tensor([phase], dtype=torch.long).to(self.device)

        # 2. Inference
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor, padding_mask=padding_mask, phase_ids=phase_ids)

        # 3. Masking (Optional)
        if valid_indices is not None and len(valid_indices) > 0:
            # Create a mask with -inf for invalid actions
            full_mask = torch.full_like(policy_logits, float('-inf'))

            # Filter valid indices that are within range
            safe_indices = [idx for idx in valid_indices if 0 <= idx < policy_logits.shape[1]]

            if safe_indices:
                full_mask[0, safe_indices] = 0
                # Apply mask
                policy_logits = policy_logits + full_mask
            else:
                pass # No valid indices in range? Fallback to raw model output

        # 4. Decode
        action_idx = torch.argmax(policy_logits, dim=1).item()

        # 5. Map to GameCommand (prefer Command representation)
        act = self.action_encoder.decode_action(action_idx, game_state, player_id)

        def _normalize_to_command(obj):
            try:
                if isinstance(obj, dict):
                    # Wrap dict into an object with attribute access to satisfy callers
                    d = obj
                    class _CmdObj:
                        pass
                    co = _CmdObj()
                    for k, v in d.items():
                        setattr(co, k, v)
                    # Normalize `type` to engine ActionType when possible
                    t = d.get('type')
                    try:
                        if isinstance(t, str):
                            at = getattr(dm.ActionType, t, None)
                            if at is None:
                                at = getattr(dm.ActionType, t.upper(), None)
                            co.type = at if at is not None else t
                        else:
                            co.type = t
                    except Exception:
                        co.type = t
                    # Ensure source_instance_id alias exists for compatibility
                    try:
                        if not hasattr(co, 'source_instance_id') and hasattr(co, 'instance_id'):
                            setattr(co, 'source_instance_id', getattr(co, 'instance_id'))
                    except Exception:
                        pass
                    return co
                if hasattr(obj, 'to_dict'):
                    try:
                        d = obj.to_dict()
                        if isinstance(d, dict):
                            # wrap dict as command object
                            dd = d
                            class _CmdObj2:
                                pass
                            co2 = _CmdObj2()
                            for k, v in dd.items():
                                setattr(co2, k, v)
                            t = dd.get('type')
                            try:
                                if isinstance(t, str):
                                    at = getattr(dm.ActionType, t, None)
                                    if at is None:
                                        at = getattr(dm.ActionType, t.upper(), None)
                                    co2.type = at if at is not None else t
                                else:
                                    co2.type = t
                            except Exception:
                                co2.type = t
                                try:
                                    if not hasattr(co2, 'source_instance_id') and hasattr(co2, 'instance_id'):
                                        setattr(co2, 'source_instance_id', getattr(co2, 'instance_id'))
                                except Exception:
                                    pass
                            return co2
                        return d
                    except Exception:
                        pass
                try:
                    m = _cc_normalize(obj)
                    if isinstance(m, dict):
                        dd = m
                        class _CmdObj3:
                            pass
                        co3 = _CmdObj3()
                        for k, v in dd.items():
                            setattr(co3, k, v)
                        t = dd.get('type')
                        try:
                            if isinstance(t, str):
                                at = getattr(dm.ActionType, t, None)
                                if at is None:
                                    at = getattr(dm.ActionType, t.upper(), None)
                                co3.type = at if at is not None else t
                            else:
                                co3.type = t
                        except Exception:
                            co3.type = t
                        try:
                            if not hasattr(co3, 'source_instance_id') and hasattr(co3, 'instance_id'):
                                setattr(co3, 'source_instance_id', getattr(co3, 'instance_id'))
                        except Exception:
                            pass
                        return co3
                    return m
                except Exception:
                    return None
            except Exception:
                return None

        cmd = _normalize_to_command(act)
        return cmd if cmd is not None else act
