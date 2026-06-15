# -*- coding: utf-8 -*-
"""아이디어 실행 엔진.

MVP 단계에서는 ideas.json에 저장된 기획 데이터를 그대로 받아 mock 실행 결과로 포장한다.
추후 실제 연동이 붙으면 run_idea() 시그니처는 유지하고 내부 구현만 교체한다.
"""
from datetime import datetime, timezone

from .ideas import get_idea


def _mock_execution(idea):
    return {
        'label': f"{idea['title']} 실행 시뮬레이션",
        'steps': [
            {'step': 1, 'label': '입력 데이터 수집', 'state': 'done'},
            {'step': 2, 'label': f"'{idea['title']}' 로직 실행", 'state': 'done'},
            {'step': 3, 'label': '사장님 알림 메시지 조립', 'state': 'done'},
        ],
        'summary': f"{idea['title']} mock 실행이 완료되었습니다. 실제 데이터 없이 시나리오만 시연합니다.",
    }


def run_idea(idea_id, payload=None):
    idea = get_idea(idea_id)
    if not idea:
        return {'status': 'not_found', 'idea_id': idea_id}
    return {
        'status': 'mocked',
        'idea_id': idea['id'],
        'title': idea['title'],
        'execution_result': _mock_execution(idea),
        'owner_message_sample': idea.get('owner_message_sample', ''),
        'next_integration_requirements': idea.get('integration_requirements', []),
        'finished_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
