# -*- coding: utf-8 -*-
import os

import pymysql
import pymysql.cursors


_HOST = os.environ.get('MYSQL_HOST', '')
_PORT = int(os.environ.get('MYSQL_PORT', '3306'))
_USER = os.environ.get('MYSQL_USER', '')
_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
_DB = os.environ.get('MYSQL_DB', 'button_man')


def get_conn():
    return pymysql.connect(
        host=_HOST,
        port=_PORT,
        user=_USER,
        password=_PASSWORD,
        database=_DB,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        connect_timeout=30,
    )


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


def db_configured():
    return all([_HOST, _USER, _PASSWORD, _DB])


def init_db():
    if not db_configured():
        print('[DB] MYSQL_* env vars not set, skipping init_db')
        return
    conn = get_conn()
    conn.close()
    print('[DB] connection ok')
