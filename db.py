# -*- coding: utf-8 -*-
import os

import pymysql
import pymysql.cursors


_HOST = os.environ.get('MYSQL_HOST', '')
_PORT = int(os.environ.get('MYSQL_PORT', '3306'))
_USER = os.environ.get('MYSQL_USER', '')
_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
_DB = os.environ.get('MYSQL_DB', 'button_man')


def db_configured():
    return all([_HOST, _USER, _PASSWORD, _DB])


def _connect(database=None):
    return pymysql.connect(
        host=_HOST,
        port=_PORT,
        user=_USER,
        password=_PASSWORD,
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        connect_timeout=10,
    )


def get_conn():
    return _connect(_DB)


def _ph():
    return '%s'


def _exec(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or [])


def _insert(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or [])
        return cur.lastrowid


def _fetchall(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or [])
        return list(cur.fetchall())


def _fetchone(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or [])
        row = cur.fetchone()
        return dict(row) if row else None


def _ensure_database():
    conn = _connect(database=None)
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.close()


def init_db():
    if not db_configured():
        print('[DB] MYSQL_* env vars not set, skipping init_db')
        return False
    _ensure_database()
    conn = get_conn()
    _exec(conn, '''CREATE TABLE IF NOT EXISTS app_meta (
        id INT AUTO_INCREMENT PRIMARY KEY,
        key_name VARCHAR(100) UNIQUE NOT NULL,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )''')
    _exec(conn, "INSERT IGNORE INTO app_meta (key_name, value) VALUES ('schema_version', '1')")
    conn.close()
    print('[DB] init ok')
    return True


def db_status():
    if not db_configured():
        return {'configured': False}
    try:
        conn = get_conn()
        row = _fetchone(conn, "SELECT VERSION() AS v, DATABASE() AS db")
        conn.close()
        return {'configured': True, 'connected': True, 'version': row['v'], 'database': row['db']}
    except Exception as e:
        return {'configured': True, 'connected': False, 'error': str(e)}
