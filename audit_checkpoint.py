"""Inspect published FP8 scales without exporting any weights.

python audit_checkpoint.py --model-dir /path/to/pinned/checkpoint
The runtime cached/full diagnostic is independently stored by capture_trace.py.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', required=True)
args = parser.parse_args()
expert_count = expert_elements = other_count = 0
all_one = True
other_min, other_max = float('inf'), float('-inf')
with safe_open(str(Path(args.model_dir) / 'model.safetensors'), framework='pt', device='cpu') as f:
    for key in f.keys():
        if not key.endswith('weight_scale_inv'):
            continue
        scale = f.get_tensor(key)
        if '.mlp.experts.' in key:
            expert_count += 1
            expert_elements += scale.numel()
            all_one = all_one and bool((scale == 1).all())
        else:
            other_count += 1
            other_min = min(other_min, float(scale.min()))
            other_max = max(other_max, float(scale.max()))

# Reproduce the model card's pack operation twice on synthetic data.
def pack(weight):
    blocks = weight.float().reshape(4, 32, 4, 32)
    scales = (blocks.abs().amax(dim=(1, 3)) / 448).clamp_min(1e-12)
    quantized = (blocks / scales[:, None, :, None]).clamp(-448, 448).reshape(128, 128).to(torch.float8_e4m3fn)
    return quantized, scales

torch.manual_seed(42)
q1, s1 = pack(torch.randn(128, 128) * .02)
q2, s2 = pack(q1)
print(json.dumps({
    'routed_expert_scale_tensors': expert_count,
    'routed_expert_scale_elements': expert_elements,
    'all_expert_scales_equal_one': all_one,
    'other_scale_tensor_count': other_count,
    'other_scale_range': [other_min, other_max],
    'synthetic_double_pack': {
        'first_scale_range': [float(s1.min()), float(s1.max())],
        'second_scales_all_one': bool((s2 == 1).all()),
        'quantized_values_unchanged': bool(torch.equal(q1.float(), q2.float())),
    },
}, indent=2))
