"""Compatible one-shot entrypoint using the v4.5 graph-native AG1 adapter.
The actual theorem matcher and AR structures are AG1's unmodified code.
"""
import json
from geomtool_v45_session import Session


def execute(payload):
    session = Session(payload)
    for _ in range(10000):
        p = session.step()
        if p['done']:
            break
    return {'engine':'AlphaGeometry1-official','queries':p['queries'],
            'elapsedMs':p['elapsedMs'],'pointCount':p['pointCount'],
            'premiseCount':p['premiseCount'],'steps':p['depth']}


def solve_json(payload):
    return json.dumps(execute(json.loads(payload)), ensure_ascii=False)
