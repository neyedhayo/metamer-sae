#!/usr/bin/env python3
"""Build the standalone presentation from local templates, assets and data."""
from argparse import ArgumentParser
from base64 import b64encode
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from src import figures

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'src'
ASSETS = ROOT / 'assets'
IMAGE_TYPES = {
    '.webp': 'image/webp', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg',
}
PLACEHOLDER = re.compile(r'\{\{([A-Z][A-Z_0-9]*)\}\}')


class SlideIndex(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.assets = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section' and 'slide' in (attrs.get('class') or '').split():
            self.ids.append(attrs.get('id'))
        if 'data-photo' in attrs:
            self.assets.add(attrs['data-photo'])


def read(name):
    return (SOURCE / name).read_text(encoding='utf-8')


def render_slides(settings):
    values = {
        'COVER_TITLE': lambda: settings['cover_title_html'],
        'TITLE_ATTRIBUTE': lambda: f'data-title="{escape(settings["title"], quote=True)}"',
        'TITLE_TEXT': lambda: escape(settings['title']),
        **figures.RENDERERS,
    }

    def expand(match):
        key = match[1]
        if key not in values:
            raise ValueError(f'Unknown placeholder {match[0]} in src/slides.html')
        return values[key]()

    return PLACEHOLDER.sub(expand, read('slides.html'))


def embed_assets(required):
    paths = {}
    for path in sorted(ASSETS.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_TYPES:
            continue
        if path.stem in paths:
            raise ValueError(f'Duplicate image ID "{path.stem}" in {paths[path.stem]} and {path}')
        paths[path.stem] = path

    missing = required - paths.keys()
    if missing:
        raise ValueError('Missing local images: ' + ', '.join(sorted(missing)))

    payload = {}
    for key, path in paths.items():
        content = b64encode(path.read_bytes()).decode('ascii')
        payload[key] = f'data:{IMAGE_TYPES[path.suffix.lower()]};base64,{content}'
    return json.dumps(payload, indent=2).replace('<', '\\u003c')


def build(target):
    settings = json.loads(read('settings.json'))
    figures.asset_references.clear()
    slides = render_slides(settings)
    index = SlideIndex()
    index.feed(slides)
    if not index.ids or None in index.ids or '' in index.ids or len(set(index.ids)) != len(index.ids):
        raise ValueError('The deck must have at least one slide and every slide must have a unique ID.')

    photos = embed_assets(index.assets | figures.asset_references)
    pieces = {
        'TITLE': escape(settings['title'], quote=True),
        'CSS': read('deck.css'),
        'SLIDES': slides,
        'PHOTOS': photos,
        'JS': read('deck.js'),
    }
    document = re.sub(r'%%([A-Z]+)%%', lambda match: pieces[match[1]], read('shell.html'))
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=target.parent,
                            prefix=f'.{target.name}.', suffix='.tmp', delete=False) as stream:
        stream.write(document)
    staged = Path(stream.name)
    try:
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    print(f'Built {len(index.ids)} slides → {target}')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', type=Path, default=ROOT / 'index.html',
                        help='Output path (default: index.html in this folder)')
    args = parser.parse_args()
    build(args.output.resolve())
