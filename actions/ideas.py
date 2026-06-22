# -*- coding: utf-8 -*-
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from threading import RLock

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'ideas.json'
_LOCK = RLock()

_NEXT_FIELDS = {
    'pre_implementation_checks': list,
    'risks': list,
    'priority_note': str,
    'first_test_plan': str,
}


def load_ideas():
    with _DATA_PATH.open(encoding='utf-8') as f:
        return json.load(f).get('ideas', [])


def get_idea(idea_id):
    for item in load_ideas():
        if item['id'] == idea_id:
            return item
    return None


def _load_doc():
    with _DATA_PATH.open(encoding='utf-8') as f:
        return json.load(f)


def _serialize(doc):
    return json.dumps(doc, ensure_ascii=False, indent=2) + '\n'


def _save_local(text):
    tmp = _DATA_PATH.with_suffix('.json.tmp')
    tmp.write_text(text, encoding='utf-8', newline='\n')
    tmp.replace(_DATA_PATH)


def _github_request(method, url, headers, body=None, timeout=15):
    data = body.encode('utf-8') if isinstance(body, str) else body
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def _push_to_github(text, idea_id, fields):
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        return {'pushed': False, 'reason': 'GITHUB_TOKEN 미설정 — 컨테이너 파일만 갱신됨'}
    repo = os.environ.get('GITHUB_REPO', 'caifyhelp-cmyk/button-man')
    branch = os.environ.get('GITHUB_BRANCH', 'main')
    api = f'https://api.github.com/repos/{repo}/contents/data/ideas.json'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'button-man',
    }
    try:
        cur = _github_request('GET', f'{api}?ref={branch}', headers)
        body = json.dumps({
            'message': f'chore(ideas): web edit · {idea_id} · {", ".join(sorted(fields))}',
            'content': base64.b64encode(text.encode('utf-8')).decode('ascii'),
            'sha': cur.get('sha'),
            'branch': branch,
        })
        put_headers = {**headers, 'Content-Type': 'application/json'}
        res = _github_request('PUT', api, put_headers, body)
        return {'pushed': True, 'commit': (res.get('commit') or {}).get('sha')}
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')[:200] if hasattr(e, 'read') else ''
        return {'pushed': False, 'reason': f'GitHub API {e.code}: {detail}'}
    except Exception as e:
        return {'pushed': False, 'reason': f'{type(e).__name__}: {e}'}


def update_next_block(idea_id, payload):
    """Update one or more of the next-step fields for an idea.

    Returns dict: { ok, status?, error?, idea?, applied?, github? }.
    """
    if not isinstance(payload, dict):
        return {'ok': False, 'status': 'bad_request', 'error': 'payload must be an object'}
    with _LOCK:
        doc = _load_doc()
        target = None
        for item in doc.get('ideas', []):
            if item.get('id') == idea_id:
                target = item
                break
        if target is None:
            return {'ok': False, 'status': 'not_found'}
        applied = {}
        for key, expected in _NEXT_FIELDS.items():
            if key not in payload:
                continue
            value = payload[key]
            if expected is list:
                if not isinstance(value, list):
                    return {'ok': False, 'status': 'bad_request', 'error': f'{key} must be a list'}
                value = [str(x).strip() for x in value if str(x).strip()]
            else:
                if value is None:
                    value = ''
                if not isinstance(value, str):
                    return {'ok': False, 'status': 'bad_request', 'error': f'{key} must be a string'}
                value = value.strip()
            target[key] = value
            applied[key] = value
        if not applied:
            return {'ok': False, 'status': 'bad_request', 'error': '편집 가능한 필드가 없습니다.'}
        text = _serialize(doc)
        _save_local(text)
        push = _push_to_github(text, idea_id, applied.keys())
        return {'ok': True, 'idea': target, 'applied': applied, 'github': push}
