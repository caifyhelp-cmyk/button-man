# -*- coding: utf-8 -*-
"""아이디어 실행 엔진.

현재 MVP 단계에서는 mock 결과를 반환한다.
추후 실제 API/워커 연동이 추가될 때 본 모듈의 run_idea(idea_id, payload) 시그니처는
유지하고 내부 구현만 교체한다.
"""
from datetime import datetime, timezone

from .ideas import get_idea


def _mock_result(idea):
    return {
        'idea_id': idea['id'],
        'title': idea['title'],
        'status': 'mocked',
        'message': '실제 연동 전입니다. 본 결과는 mock 데이터입니다.',
        'sample_output': {
            'steps': [
                {'step': 1, 'label': '입력 수집', 'state': 'done'},
                {'step': 2, 'label': '아이디어 실행', 'state': 'done'},
                {'step': 3, 'label': '결과 정리', 'state': 'done'},
            ],
            'summary': f"{idea['title']} 실행이 완료되었습니다 (mock).",
        },
        'finished_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def run_idea(idea_id, payload=None):
    idea = get_idea(idea_id)
    if not idea:
        return {'status': 'not_found', 'idea_id': idea_id}
    return _mock_result(idea)
