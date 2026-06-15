# -*- coding: utf-8 -*-
import json
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'ideas.json'


def load_ideas():
    with _DATA_PATH.open(encoding='utf-8') as f:
        return json.load(f).get('ideas', [])


def get_idea(idea_id):
    for item in load_ideas():
        if item['id'] == idea_id:
            return item
    return None
