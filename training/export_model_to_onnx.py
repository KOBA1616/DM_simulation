#!/usr/bin/env python3
"""Export latest trained DuelTransformer model to ONNX."""
from pathlib import Path
import torch
import sys
import os

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

def find_latest_model(models_dir: Path):
    files = list(models_dir.glob('duel_transformer_*.pth'))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def export_to_onnx(model_path: Path, onnx_path: Path):
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer

    # Model hyperparams must match training. Prefer canonical CommandEncoder size if available.
    import dm_ai_module
    if hasattr(dm_ai_module, 'CommandEncoder') and dm_ai_module.CommandEncoder is not None:
        action_dim = dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE
        print(f'  Using CommandEncoder.TOTAL_COMMAND_SIZE = {action_dim}')
    else:
        action_dim = 600
    model = DuelTransformer(vocab_size=1000, action_dim=action_dim, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, max_len=200)
    checkpoint = torch.load(str(model_path), map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    # Backwards-compat: older checkpoints stored pos_embedding with a leading
    # batch dim (1, max_len, d_model). If present, squeeze that dim to match
    # the updated model which stores (max_len, d_model).
    if 'pos_embedding' in state:
        v = state['pos_embedding']
        if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
            state = dict(state)
            state['pos_embedding'] = v.squeeze(0)
    # If checkpoint and model action_dim differ, attempt to reconcile policy head shapes
    model_state = model.state_dict()
    reconciled = dict(state)
    # policy_head.1 corresponds to final linear layer weight/bias in older naming
    w_key = None
    b_key = None
    for k in reconciled.keys():
        if k.endswith('policy_head.1.weight'):
            w_key = k
        if k.endswith('policy_head.1.bias'):
            b_key = k

    try:
        if w_key and b_key and w_key in reconciled and b_key in reconciled:
            ck_w = reconciled[w_key]
            ck_b = reconciled[b_key]
            target_w = model_state.get(w_key)
            target_b = model_state.get(b_key)
            if target_w is not None and ck_w.shape != target_w.shape:
                # If checkpoint has larger output dim, truncate; if smaller, pad with zeros
                import torch as _torch
                out_ck, in_ck = ck_w.shape
                out_tgt, in_tgt = target_w.shape
                min_out = min(out_ck, out_tgt)
                min_in = min(in_ck, in_tgt)
                new_w = _torch.zeros_like(target_w)
                new_w[:min_out, :min_in] = ck_w[:min_out, :min_in]
                reconciled[w_key] = new_w
                # bias
                new_b = _torch.zeros_like(target_b)
                new_b[:min_out] = ck_b[:min_out]
                reconciled[b_key] = new_b
    except Exception:
        pass

    model.load_state_dict(reconciled)
    model.eval()

    # Use a small multi-batch dummy input so exported graph captures batched
    # layout (helps avoid baking batch-size==1 into constant reshape tensors).
    dummy_input = torch.zeros((4, 200), dtype=torch.long)
    with torch.no_grad():
        # Use a recent opset and export with a dynamic batch dimension so ONNX model
        # accepts variable batch sizes (e.g., parallel>1).
        dynamic_axes = {
            'input_ids': {0: 'batch_size'},
            'policy_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'},
        }
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=18,
                input_names=['input_ids'],
                output_names=['policy_logits', 'value'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                verbose=False,
            )
        except ModuleNotFoundError as e:
            # Some torch versions require `onnxscript` for newer opsets.
            # Fall back to an older opset if not installed.
            print('Warning: failed export with opset18:', e)
            print('Retrying export with opset_version=13')
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=13,
                input_names=['input_ids'],
                output_names=['policy_logits', 'value'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                verbose=False,
            )

if __name__ == '__main__':
    models_dir = repo_root / 'models'
    models_dir.mkdir(exist_ok=True)
    latest = find_latest_model(models_dir)
    if latest is None:
        print('No model .pth found in models/')
        sys.exit(1)
    out = models_dir / (latest.stem + '.onnx')
    print('Exporting', latest, '->', out)
    export_to_onnx(latest, out)
    print('Export complete')
