"""Validate evidence, model-specific invariants and static project-site paths."""
import hashlib
import json
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
records = [json.loads(p.read_text()) for p in (ROOT / 'catalog/models').glob('*.json')]
models = {m['id']: m for m in records}
assert len(models) == 9
assert len({m['family']['id'] for m in records}) == 7
assert sum(bool(m['variants'].get('tiny')) for m in records) == 7
assert sum(m['has_trace'] for m in records) == 1
collection = json.loads((ROOT / 'sources/collection.json').read_text())
expected_rows = {'input','attention','indexer','position','gating','normalization','residual','routing','experts','shared','state','prefill','decode','output','auxiliary'}
for m in records:
    source_ids = {s['id'] for s in m['sources']}
    assert {'base','code'} <= source_ids
    assert expected_rows == set(m['comparison'])
    for source in m['sources']:
        assert source['url'].startswith('https://')
        if 'local' in source:
            path = ROOT / source['local']
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
            assert (DIST / source['local']).read_bytes() == path.read_bytes()
        if source.get('collection_member'):
            assert source['repo'] in collection['model_ids']
    for name, variant in m['variants'].items():
        dims = variant['dimensions']
        assert dims['layers'] == len(variant['attention_schedule']) == len(variant['ffn_schedule'])
        assert set(variant['attention_schedule']) <= set(m['attention'])
        assert 0 < dims['top_k'] <= dims['experts']
        assert dims['hidden'] > 0
    for row in m['comparison'].values():
        assert row['title'] and row['detail'] and row['signature']
        assert set(row['sources']) <= source_ids
    for mechanism in list(m['attention'].values()) + [m['moe'],m['residual']]:
        assert mechanism['steps']
        assert set(mechanism['sources']) <= source_ids
    assert (DIST / 'models' / m['id'] / 'index.html').is_file()
    assert (DIST / 'families' / m['family']['id'] / 'index.html').is_file()

# Catch previously audited high-risk misclassifications, rather than snapshotting UI.
def v(id, variant='base'):
    return models[id]['variants'][variant]
assert v('qwen3-30b-a3b','tiny')['ffn_schedule'] == ['Dense','MoE']
assert v('qwen3-5-35b-a3b')['dimensions']['top_k'] == 8
assert v('qwen3-5-35b-a3b','tiny')['dimensions']['top_k'] == 10
assert v('gemma-4-26b-a4b')['attention_schedule'].count('Global GQA') == 5
assert v('gemma-4-26b-a4b','tiny')['attention_schedule'].count('Global GQA') == 2
assert v('deepseek-v4-pro')['attention_schedule'].count('HCA 128') == 31
assert v('deepseek-v4-pro')['attention_schedule'].count('CSA 4') == 30
assert v('deepseek-v4-pro')['ffn_schedule'][:3] == ['HashMoE'] * 3
assert [i for i,x in enumerate(v('mimo-v2-5-pro')['attention_schedule']) if x=='Global GQA'] == [0,7,15,23,31,39,47,55,62,69]
assert v('minimax-m3')['ffn_schedule'][:3] == ['Dense'] * 3
assert v('kimi-k3')['attention_schedule'].count('KDA') == 69
assert v('kimi-k3')['attention_schedule'][-2:] == ['gated MLA','gated MLA']
assert v('glm-5.3-flash')['attention_schedule'].count('KDA') == 34
assert v('glm-5.3-flash')['attention_schedule'][-1] == 'KDA'
assert v('qwen3-8-flash-next')['attention_schedule'].count('QSA') == 12
assert 'tiny' not in models['minimax-m3']['variants']
assert 'tiny' not in models['mimo-v2-5-pro']['variants']

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        self.links += [value for key,value in attrs if key in ['href','src'] and value]

for page in DIST.rglob('*.html'):
    parser=LinkParser(); parser.feed(page.read_text())
    for link in parser.links:
        url=urlsplit(link)
        if url.scheme or url.netloc or not url.path:continue
        if url.path.startswith('/'):
            assert url.path == '/tiny-random/' and page.name == '404.html', (page,link)
            continue
        target=(page.parent / unquote(url.path)).resolve()
        assert target.is_relative_to(DIST.resolve()), (page,link)
        if target.is_dir():target /= 'index.html'
        assert target.is_file(), (page,link)
assert not list(DIST.rglob('*.safetensors'))
assert not any(path.is_symlink() for path in DIST.rglob('*'))
assert (DIST/'.nojekyll').exists()
assert json.loads((DIST/'assets/catalog.json').read_text())['models'] == sorted(records,key=lambda m:m['order'])
print('PASS: 9 profiles / 7 families, pinned source hashes, exact schedules, tiny relationships, comparison rows, generated routes and relative asset links.')
