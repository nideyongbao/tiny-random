"""Check semantic invariants of the published execution record (no weights needed)."""
import json
import math
from pathlib import Path

root = Path(__file__).parent / 'dist' / 'trace'
session = json.loads((root / 'session.json').read_text())
assert len(session['steps']) == 4
assert len(session['prompt_tokens']) == 18
assert session['generation']['logits_to_keep'] == 1
previous = None
for i, step in enumerate(session['steps']):
    length = 18 + i
    assert step['context_length'] == length
    assert step['input_shape'] == [1, 18 if i == 0 else 1]
    assert step['position_ids'] == (list(range(18)) if i == 0 else [length - 1])
    assert step['attention_mask'] == [1] * length
    assert step['logits_shape'] == [1, 1, 154880]
    assert step['verification']['trace_matches_uninstrumented']
    assert step['verification']['uninstrumented_cached_max_abs'] == 0
    assert step['top_tokens'][0]['id'] == step['chosen']['id']
    if previous:
        assert step['input_ids'] == [previous['chosen']['id']]
        assert step['cache_before'] == previous['cache_after'], 'cache snapshot changed retroactively'
    else:
        assert step['cache_before'] == []
        assert step['input_ids'] == [t['id'] for t in session['prompt_tokens']]
    for c in step['cache_after']:
        tensors = c['tensors']
        if c['layer'] in [3, 7]:
            assert tensors['keys']['shape'] == [1, 1, length, 256]
            assert tensors['values']['shape'] == [1, 1, length, 0]
            assert tensors['indexer_keys']['shape'] == [1, length, 257]
        else:
            assert tensors['conv_states.0']['shape'] == [1, 24576, 4]
            assert tensors['recurrent_states.0']['shape'] == [1, 64, 128, 128]
    for router in step['routers'].values():
        for ids, weights in zip(router['selected'], router['weights']):
            assert len(set(ids)) == 8 and all(0 <= x < 32 for x in ids)
            assert math.isclose(sum(weights), 2.5, abs_tol=1e-5)
    for indexer in step['indexers'].values():
        for q, indices in enumerate(indexer[0]):
            assert len(indices) == 131
            assert all(v == -1 or 0 <= v <= step['position_ids'][q] for v in indices)
    for attention in step['attention'].values():
        for q, row in enumerate(attention['head0']):
            assert len(row) == length
            assert math.isclose(sum(row), 1, abs_tol=1e-5)
            assert all(v == 0 for v in row[step['position_ids'][q] + 1:])
    ops = json.loads((root / f'ops-{i}.json').read_text())
    assert step['op_count'] == len(ops['events'])
    for index, event in enumerate(ops['events']):
        assert event['id'] == index
        assert len(event['inputs']) == len(event['input_versions'])
        assert len(event['outputs']) == len(event['output_versions'])
        assert all(t in ops['tensors'] for t in event['inputs'] + event['outputs'])
    for module in step['modules']:
        assert 0 <= module['op_start'] <= module['op_end'] <= step['op_count']
    previous = step
assert session['generated_ids'] == [s['chosen']['id'] for s in session['steps']]
print('PASS: autoregressive handoff, cache continuity/shapes, router weights, causal indices/attention, operator references and module ranges.')
print('Known cached/full allclose failures:', [s['index'] for s in session['steps'] if not s['verification']['allclose_rtol1e-3_atol1e-4']])
