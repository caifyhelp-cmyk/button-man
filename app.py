# -*- coding: utf-8 -*-
import os
import traceback

from flask import Flask, render_template, jsonify
from flask_cors import CORS

import db

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.environ.get('SECRET_KEY', 'button-man-dev-secret')
CORS(app)


@app.errorhandler(Exception)
def _handle_all_exceptions(e):
    tb = traceback.format_exc()
    print(f'[UNHANDLED] {type(e).__name__}: {e}\n{tb}')
    return jsonify({'error': str(e), 'type': type(e).__name__}), 500


try:
    db.init_db()
except Exception as e:
    print(f'[DB] init skipped: {e}')


@app.route('/health')
def health_check():
    return jsonify({'ok': True})


@app.route('/health/db')
def health_db():
    return jsonify(db.db_status())


@app.route('/')
def dashboard():
    return render_template('dashboard.html', active='dashboard')


@app.route('/data')
def page_data():
    return render_template('data.html', active='data')


@app.route('/activity')
def page_activity():
    return render_template('activity.html', active='activity')


@app.route('/analytics')
def page_analytics():
    return render_template('analytics.html', active='analytics')


@app.route('/settings')
def page_settings():
    return render_template('settings.html', active='settings')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), debug=True)
