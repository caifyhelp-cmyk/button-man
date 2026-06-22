# -*- coding: utf-8 -*-
import os
import traceback

from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS

import actions

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB upload cap for QA agent
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


@app.route('/api/ideas/<idea_id>/next', methods=['POST'])
def api_update_next(idea_id):
    """Update the '다음 작업' block fields for an idea.

    Body: any subset of { pre_implementation_checks, risks, priority_note, first_test_plan }.
    """
    payload = request.get_json(silent=True) or {}
    result = actions.update_next_block(idea_id, payload)
    if not result.get('ok'):
        st = result.get('status')
        code = 404 if st == 'not_found' else (400 if st == 'bad_request' else 500)
        return jsonify(result), code
    return jsonify(result)


@app.route('/api/qa/run', methods=['POST'])
def api_qa_run():
    """Multipart endpoint for the video QA agent (qa-video idea).

    Runs the real pipeline: ffmpeg → Whisper → dHash similarity → GPT-4o vision →
    aggregator. Requires OPENAI_API_KEY and ffmpeg installed in the container.
    """
    from qa.ffmpeg_utils import FfmpegMissing
    from qa.web_runner import run_qa_on_upload

    video = request.files.get('video')
    if not video or not video.filename:
        return jsonify({'error': 'video file required (field name: video)'}), 400

    if not os.environ.get('OPENAI_API_KEY'):
        return jsonify({
            'error': "OPENAI_API_KEY가 설정되지 않았습니다. Fly 배포에서는 `fly secrets set OPENAI_API_KEY=sk-...` 후 재배포하세요.",
            'type': 'ConfigError',
        }), 503

    def _form(name, default=''):
        return (request.form.get(name) or default).strip()

    def _form_list(name):
        raw = _form(name)
        if not raw:
            return []
        return [s.strip() for s in raw.split(',') if s.strip()]

    client_info = {
        'clientName': _form('clientName'),
        'industry': _form('industry'),
        'services': _form_list('services'),
        'promotionPoints': _form_list('promotionPoints'),
        'forbiddenClaims': _form_list('forbiddenClaims'),
        'brandTone': _form('brandTone'),
        'notes': _form('notes'),
    }
    qa_context = {
        'videoType': _form('videoType'),
        'sceneIntent': _form('sceneIntent'),
        'expectedSubjects': _form_list('expectedSubjects'),
        'forbiddenObjects': _form_list('forbiddenObjects'),
    }

    raw_bytes = video.read()

    try:
        result = run_qa_on_upload(
            video_bytes=raw_bytes,
            video_filename=video.filename,
            client_info=client_info,
            qa_context=qa_context,
            script=_form('script') or None,
            scenes_text=_form('scenes') or None,
            generation_prompt=_form('generationPrompt') or None,
            references=_form('references') or None,
        )
        return jsonify(result)
    except FfmpegMissing as e:
        return jsonify({'error': str(e), 'type': 'FfmpegMissing'}), 500


@app.route('/history')
def history():
    return render_template('history.html', active='history')


@app.route('/settings')
def settings():
    return render_template('settings.html', active='settings')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), debug=True)
