# -*- coding: utf-8 -*-
import os
import traceback

from flask import Flask, render_template, jsonify, request
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


@app.route('/api/ideas/<idea_id>/run', methods=['POST'])
def api_run(idea_id):
    payload = request.get_json(silent=True) or {}
    result = actions.run_idea(idea_id, payload)
    status = 404 if result.get('status') == 'not_found' else 200
    return jsonify(result), status


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), debug=True)
