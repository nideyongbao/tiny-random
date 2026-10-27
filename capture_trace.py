"""Record a reproducible, CPU/FP32 reference execution. Never save weights.

Run in an environment with torch 2.11.0 and transformers 5.20.2:
python capture_trace.py --model-dir /path/to/pinned/checkpoint

Safetensors FP8 weights are dequantized during loading. Every parameter is then
cast to FP32. These are real executions of that reference backend, not FP8 GPU
benchmark results. ATen records describe dispatcher operations, not hardware
instructions. Tensor values are samples, except explicit token/router/index data.
"""
import argparse
import hashlib
import json
import math
import time
import weakref
from collections import Counter
from pathlib import Path

import torch
import transformers
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from transformers import AutoModelForImageTextToText, AutoTokenizer, FineGrainedFP8Config
import transformers.models.glm5_next.modeling_glm5_next as implementation


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict): return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [json_safe(v) for v in value]
    return value


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    return []


def tensor_sample(t):
    with _disable_current_modes():
        x = t.detach().cpu()
        flat = x.reshape(-1)
        if not flat.numel():
            return {"shape": list(x.shape), "dtype": str(x.dtype).removeprefix('torch.'), "numel": 0, "sample": []}
        # Statistics are over the full tensor, values are a bounded sample.
        f = flat.float()
        result = {"shape": list(x.shape), "dtype": str(x.dtype).removeprefix('torch.'), "numel": x.numel(),
                  "min": f.min().item(), "max": f.max().item(), "mean": f.mean().item(),
                  "std": f.std(unbiased=False).item(), "norm": f.norm().item(),
                  "finite": bool(torch.isfinite(f).all()), "sample": flat[:24].tolist()}
        if x.ndim >= 2:
            tile = x.reshape(-1, x.shape[-1]) if x.shape[-1] else None
            if tile is not None:
                result['tile'] = tile[:8, :12].tolist()
                result['last_row'] = tile[-1, :24].tolist()
        return result


class Recorder(TorchDispatchMode):
    def __init__(self, model):
        super().__init__()
        self.active = False
        self.stack = []
        self.events = []
        self.modules = []
        self.refs = {}
        self.metadata = {}
        self.serial = 0
        self.weights = {id(p): n for n, p in model.named_parameters()}
        self.handles = []
        self.record = {}
        for name, module in model.named_modules():
            self.handles.append(module.register_forward_pre_hook(self.pre(name or 'root'), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.post(name or 'root'), with_kwargs=True))

    def tid(self, t):
        key = id(t)
        prior = self.refs.get(key)
        if prior is not None and prior[0]() is t:
            return prior[1]
        self.serial += 1
        tid = f't{self.serial}'
        self.refs[key] = (weakref.ref(t), tid)
        self.metadata[tid] = {'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.'),
                              'parameter': self.weights.get(key), 'producer': None, 'version': 0}
        return tid

    def pre(self, name):
        def hook(module, args, kwargs):
            if not self.active:
                return
            incoming = tensors(args) + tensors(kwargs)
            rec = {'path': name, 'class': module.__class__.__name__, 'inputs': [self.tid(t) for t in incoming],
                   'input_samples': [tensor_sample(t) for t in incoming[:2]], 'op_start': len(self.events)}
            self.stack.append((name, rec))
        return hook

    def post(self, name):
        def hook(module, args, kwargs, output):
            if not self.active:
                return
            _, rec = self.stack.pop()
            outgoing = tensors(output)
            rec.update(outputs=[self.tid(t) for t in outgoing],
                       output_samples=[tensor_sample(t) for t in outgoing[:3]], op_end=len(self.events))
            self.modules.append(rec)
            with _disable_current_modes():
                if module.__class__.__name__ == 'Glm5NextTextTopkRouter':
                    logits, weights, selected = output
                    self.record.setdefault('routers', {})[name] = {
                        'logits': logits.detach().cpu().tolist(), 'sigmoid_scores': logits.sigmoid().detach().cpu().tolist(),
                        'weights': weights.detach().cpu().tolist(), 'selected': selected.detach().cpu().tolist()}
                elif module.__class__.__name__ == 'Glm5NextTextIndexer':
                    self.record.setdefault('indexers', {})[name] = output.detach().cpu().tolist()
                elif module.__class__.__name__ == 'Glm5NextTextAttention' and output[1] is not None:
                    a = output[1].detach().cpu()
                    self.record.setdefault('attention', {})[name] = {
                        'shape': list(a.shape), 'head0': a[0, 0].tolist(), 'last_query_head0': a[0, 0, -1].tolist()}
        return hook

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.active:
            return func(*args, **kwargs)
        input_ids = [self.tid(t) for t in tensors(args) + tensors(kwargs)]
        input_versions = [self.metadata[k]['version'] for k in input_ids]
        output = func(*args, **kwargs)
        outgoing = tensors(output)
        output_ids = [self.tid(t) for t in outgoing]
        index = len(self.events)
        for tid in output_ids:
            if tid in input_ids:
                self.metadata[tid]['version'] += 1
            self.metadata[tid]['producer'] = index
        def constants(v):
            if isinstance(v, torch.Tensor): return None
            if isinstance(v, (str, int, float, bool)) or v is None: return v
            if isinstance(v, (tuple, list)): return [constants(x) for x in v]
            return str(v)
        self.events.append({'id': index, 'op': str(func), 'scope': self.stack[-1][0] if self.stack else 'forward',
                            'inputs': input_ids, 'input_versions': input_versions, 'outputs': output_ids,
                            'output_versions': [self.metadata[k]['version'] for k in output_ids],
                            'arguments': constants(args), 'kwargs': {k: constants(v) for k, v in kwargs.items()}})
        return output

    def begin(self):
        self.events, self.modules, self.record, self.stack = [], [], {}, []
        self.refs, self.metadata, self.serial = {}, {}, 0
        self.active = True


def cache_snapshot(cache):
    if cache is None:
        return []
    output = []
    for i, layer in enumerate(cache.layers):
        r = {'layer': i, 'class': type(layer).__name__, 'tensors': {}}
        for name in ['keys', 'values', 'indexer_keys', 'conv_states', 'recurrent_states']:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                r['tensors'][name] = tensor_sample(value)
            elif isinstance(value, (list, tuple)):
                for j, t in enumerate(value):
                    if isinstance(t, torch.Tensor): r['tensors'][f'{name}.{j}'] = tensor_sample(t)
            elif isinstance(value, dict):
                for j, t in value.items():
                    if isinstance(t, torch.Tensor): r['tensors'][f'{name}.{j}'] = tensor_sample(t)
        for name in ['cumulative_length', 'is_initialized', 'is_conv_states_initialized', 'is_recurrent_states_initialized']:
            v = getattr(layer, name, None)
            if isinstance(v, (int, bool)): r[name] = v
        output.append(r)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', default='dist/trace')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, loading = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32, device_map='cpu',
        attn_implementation='eager', quantization_config=FineGrainedFP8Config(dequantize=True),
        output_loading_info=True)
    assert not loading.get('missing_keys'), loading
    assert not loading.get('mismatched_keys'), loading
    assert all('layers.8.' in k for k in loading.get('unexpected_keys', [])), loading
    model.to(torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    messages = [{'role': 'user', 'content': '请用一句话介绍你自己。'}]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort='low')
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='low', return_tensors='pt', return_dict=True)
    prompt_ids = inputs['input_ids'][0].tolist()
    def token_info(tid):
        return {'id': tid, 'piece': tokenizer.convert_ids_to_tokens(tid),
                'text': tokenizer.decode([tid], skip_special_tokens=False), 'special': tid in tokenizer.all_special_ids}
    data = {'model': 'tiny-random/glm-5.3-flash', 'revision': 'aeb642e89f4bb6b55b186ca440edff0acb9ee9f5',
            'backend': 'CPU · FP32 · eager attention · PyTorch reference KDA/conv',
            'precision_note': 'FP8 checkpoint dequantized on load, then all parameters cast to FP32. Not native FP8 GPU output.',
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'implementation_sha256': hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
            'messages': messages, 'template': template, 'template_options': {'add_generation_prompt': True, 'reasoning_effort': 'low'},
            'prompt_tokens': [token_info(t) for t in prompt_ids], 'prompt_length': len(prompt_ids),
            'generation': {'method': 'greedy argmax on raw logits', 'temperature': None, 'max_new_tokens': 4,
                           'logits_to_keep': 1, 'use_cache': True, 'seed': 42, 'speculative': False},
            'omitted': ['visual branch (text-only input)', 'MTP (ordinary autoregressive generation)', 'training / backward'],
            'sampling_policy': 'Exact shape/dtype + full-tensor statistics. Values: first 24 scalars, leading 8x12 tile, last row sample. Router/indexer/head0 attention are complete for this short prompt.',
            'loading': {'missing': [], 'ignored_mtp_keys': sorted(loading.get('unexpected_keys', []))}, 'steps': []}
    recorder = Recorder(model)
    # Function arguments can include parameter views (e.g. convolution kernels).
    # Redact their values by storage identity, not only Python tensor identity.
    parameter_storages = {p.untyped_storage().data_ptr(): name for name, p in model.named_parameters()}
    def boundary_sample(t):
        name = parameter_storages.get(t.untyped_storage().data_ptr())
        if name is not None:
            return {'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.'),
                    'numel': t.numel(), 'parameter': name, 'values_omitted': True}
        return tensor_sample(t)
    # Functions are not nn.Modules. Wrap their boundary without changing their math.
    function_records = []
    originals = {}
    for name in ['causal_conv1d_fn', 'causal_conv1d_update', 'chunk_kimi_delta_attention', 'recurrent_kimi_delta_attention']:
        original = getattr(implementation, name); originals[name] = original
        def wrapper(*a, _original=original, _name=name, **kw):
            if not recorder.active: return _original(*a, **kw)
            r = {'function': _name, 'scope': recorder.stack[-1][0], 'inputs': [boundary_sample(t) for t in tensors(a)],
                 'kwargs': {k: boundary_sample(v) if isinstance(v, torch.Tensor) else v for k,v in kw.items() if isinstance(v,(torch.Tensor,bool,int,float,str,type(None)))},
                 'op_start': len(recorder.events)}
            result = _original(*a, **kw)
            r['outputs'] = [tensor_sample(t) for t in tensors(result)]
            r['op_end'] = len(recorder.events)
            function_records.append(r)
            return result
        setattr(implementation, name, wrapper)
    cache = None
    baseline_cache = None
    generated = inputs['input_ids']
    current = generated
    with torch.inference_mode():
        for step in range(4):
            before = cache_snapshot(cache)
            attention_mask = torch.ones(1, generated.shape[1], dtype=torch.long)
            position = torch.arange(generated.shape[1]-current.shape[1], generated.shape[1]).unsqueeze(0)
            function_records.clear()
            recorder.begin()
            with recorder:
                result = model(input_ids=current, attention_mask=attention_mask, position_ids=position,
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
            recorder.active = False
            cache = result.past_key_values
            logits = result.logits[0, -1].float()
            probabilities = logits.softmax(-1)
            values, ids = logits.topk(10)
            chosen = int(ids[0])
            # Independent, uninstrumented cached replay validates trace fidelity.
            baseline = model(input_ids=current, attention_mask=attention_mask, position_ids=position,
                             past_key_values=baseline_cache, use_cache=True, logits_to_keep=1)
            baseline_cache = baseline.past_key_values
            baseline_logits = baseline.logits[0,-1].float()
            trace_difference = (baseline_logits-logits).abs().max().item()
            assert torch.allclose(baseline_logits,logits,rtol=1e-5,atol=1e-6), trace_difference
            # Full-sequence recomputation is a separate diagnostic. Do not hide a
            # reference-backend discrepancy or silently increase its tolerance.
            reference = model(input_ids=generated, attention_mask=attention_mask, use_cache=False, logits_to_keep=1).logits[0,-1].float()
            difference = (reference-logits).abs()
            verify = {'max_abs_logit_difference': difference.max().item(),
                      'same_argmax': int(reference.argmax()) == chosen,
                      'allclose_rtol1e-3_atol1e-4': bool(torch.allclose(reference,logits,rtol=1e-3,atol=1e-4)),
                      'uninstrumented_cached_max_abs': trace_difference,
                      'trace_matches_uninstrumented': True}
            step_record = {'index': step, 'phase': 'prefill' if step == 0 else 'decode',
                           'input_ids': current[0].tolist(), 'input_shape': list(current.shape),
                           'position_ids': position[0].tolist(), 'attention_mask': attention_mask[0].tolist(),
                           'context_length': generated.shape[1], 'cache_before': before, 'cache_after': cache_snapshot(cache),
                           'logits_shape': list(result.logits.shape),
                           'top_tokens': [dict(token_info(int(i)), logit=float(v), probability=float(probabilities[i])) for v,i in zip(values,ids)],
                           'chosen': token_info(chosen), 'verification': verify,
                           'modules': recorder.modules, 'functions': list(function_records),
                           'op_count': len(recorder.events), 'op_types': dict(Counter(e['op'] for e in recorder.events)),
                           **recorder.record}
            (out / f'ops-{step}.json').write_text(json.dumps(json_safe({'events': recorder.events, 'tensors': recorder.metadata}),ensure_ascii=False,separators=(',',':'),allow_nan=False))
            data['steps'].append(step_record)
            print('captured',step,step_record['phase'],chosen,repr(tokenizer.decode([chosen])),len(recorder.events),'ops',verify,flush=True)
            current = torch.tensor([[chosen]], dtype=torch.long)
            generated = torch.cat([generated,current],dim=1)
            if chosen in model.config.text_config.eos_token_id: break
    data['generated_ids'] = generated[0,len(prompt_ids):].tolist()
    data['generated_text'] = tokenizer.decode(data['generated_ids'],skip_special_tokens=False)
    data['stopping_reason'] = 'EOS' if data['generated_ids'][-1] in model.config.text_config.eos_token_id else 'max_new_tokens'
    (out/'session.json').write_text(json.dumps(json_safe(data),ensure_ascii=False,separators=(',',':'),allow_nan=False))
    print('saved',out,flush=True)


if __name__ == '__main__': main()
