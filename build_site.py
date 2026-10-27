"""Build an offline, dependency-free atlas. No upstream fetch or model execution."""
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dist'
MODELS = sorted((ROOT / 'catalog/models').glob('*.json'))

def page(relative, title, view, selected='', depth=0):
    prefix = '../' * depth or './'
    destination = OUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = (ROOT / 'site/shell.html').read_text()
    for key, value in {'TITLE': title, 'BASE': prefix, 'VIEW': view, 'SELECTED': selected}.items():
        body = body.replace('{{' + key + '}}', html.escape(value, quote=True))
    destination.write_text(body)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'assets').mkdir(exist_ok=True)
    # Only overwrite the generated atlas, preserving the separate recorded GLM replay.
    for directory in ['models', 'families', 'compare', 'sources']:
        if (OUT / directory).exists():
            shutil.rmtree(OUT / directory)
    records = [json.loads(path.read_text()) for path in MODELS]
    if len(records) != 9:
        raise ValueError(f'Expected 9 reviewed model profiles, got {len(records)}')
    records.sort(key=lambda m: m['order'])
    (OUT / 'assets/catalog.json').write_text(json.dumps({'schema_version': 1, 'reviewed_on': '2026-10-27', 'models': records}, ensure_ascii=False, indent=2))
    for asset in ['atlas.js', 'atlas.css']:
        shutil.copy2(ROOT / 'site' / asset, OUT / 'assets' / asset)
    shutil.copytree(ROOT / 'sources', OUT / 'sources')
    page('index.html', 'tiny-random · MoE Architecture Atlas', 'home')
    page('compare/index.html', '模型正向对比 · tiny-random', 'compare', depth=1)
    families = {}
    for model in records:
        page(f"models/{model['id']}/index.html", model['name'] + ' · tiny-random', 'model', model['id'], 2)
        families[model['family']['id']] = model['family']['name']
    for slug, name in families.items():
        page(f'families/{slug}/index.html', name + ' 家族 · tiny-random', 'family', slug, 2)
    (OUT / '.nojekyll').touch()
    (OUT / '404.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>页面未找到</title><body><h1>页面未找到</h1><p>请返回 <a href="/tiny-random/">tiny-random 首页</a>，使用模型目录进入子页面。</p></body></html>')
    print(f'Built {len(records)} models, {len(families)} families, compare and home into dist/')

if __name__ == '__main__':
    main()
