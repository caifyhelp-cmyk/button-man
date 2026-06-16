# -*- coding: utf-8 -*-
import os
import traceback

from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS

import actions

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.environ.get('SECRET_KEY', 'button-man-dev-secret')
CORS(app)


@app.errorhandler(Exception)
def _handle_all_exceptions(e):
    tb = traceback.format_exc()
    print(f'[UNHANDLED] {type(e).__name__}: {e}\n{tb}')
    return jsonify({'error': str(e), 'type': type(e).__name__}), 500


@app.route('/health')
def health_check():
    return jsonify({'ok': True})


@app.route('/')
def dashboard():
    ideas = actions.load_ideas()
    return render_template('dashboard.html', ideas=ideas, active='ideas')


@app.route('/ideas/<idea_id>')
def idea_detail(idea_id):
    idea = actions.get_idea(idea_id)
    if not idea:
        return render_template('not_found.html', idea_id=idea_id, active='ideas'), 404
    return render_template('idea_detail.html', idea=idea, active='ideas')


@app.route('/api/ideas/<idea_id>/run', methods=['POST'])
def api_run(idea_id):
    payload = request.get_json(silent=True) or {}
    result = actions.run_idea(idea_id, payload)
    status = 404 if result.get('status') == 'not_found' else 200
    return jsonify(result), status


_anthropic_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        key = os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            return None
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


_REFINE_SYSTEM = (
    "당신은 한국어 카톡 메시지를 표준 경어체로 자연스럽게 다듬는 역할입니다.\n"
    "규칙:\n"
    "- 음슴체(~함, ~임, ~음, ~었음 등)는 모두 표준 경어체(~합니다, ~입니다, ~었습니다)로 변환합니다.\n"
    "- 명사 + '였습니다/이었습니다'는 받침 유무에 따라 정확히 구분합니다.\n"
    "  (받침 없음: ~였습니다 / 받침 있음: ~이었습니다)\n"
    "- 숫자, 원화 표기, %, +/- 기호, 줄바꿈 구조는 그대로 유지합니다.\n"
    "- 블록 사이의 빈 줄, 한 블록 내부의 줄바꿈도 유지합니다.\n"
    "- 새 정보를 추가하거나 의역하지 않습니다. 과한 조언이나 액션 제안도 넣지 않습니다.\n"
    "- 응답에 다른 설명 없이 다듬어진 메시지 본문만 그대로 출력합니다."
)


@app.route('/api/refine-message', methods=['POST'])
def refine_message():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'refined': '', 'used': False, 'error': 'empty'}), 400
    client = _get_anthropic()
    if client is None:
        return jsonify({'refined': text, 'used': False, 'error': 'no_api_key'}), 503
    try:
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=1024,
            system=_REFINE_SYSTEM,
            messages=[{'role': 'user', 'content': text}],
        )
        out = ''
        for block in (resp.content or []):
            if getattr(block, 'type', '') == 'text':
                out += block.text
        return jsonify({'refined': out.strip(), 'used': True})
    except Exception as e:
        return jsonify({'refined': text, 'used': False, 'error': str(e)}), 500


@app.route('/history')
def history():
    return render_template('history.html', active='history')


@app.route('/settings')
def settings():
    return render_template('settings.html', active='settings')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), debug=True)
