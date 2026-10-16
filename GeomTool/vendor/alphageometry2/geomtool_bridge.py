"""GeomTool v4.4 <-> the UNMODIFIED official AlphaGeometry2 DDAR engine.

No mathematical inference is implemented here. Each positive goal is checked
by DDAR.check_pred after force_pred and deduction_closure. Numerical candidate
preselection NEVER establishes a goal. Keep this module usable in ordinary
CPython for differential tests as well as Pyodide.
"""
import json
import time
import numpy as np
from ddar import DDAR
from parse import AGPoint, AGPredicate


def execute(payload):
    start = time.perf_counter()
    coords = payload['compiled']['points']
    by_name = {p['name']: AGPoint(p['name'], np.array(p['value'], dtype=float)) for p in coords}
    if len(by_name) != len(coords):
        raise ValueError('Duplicate points in compiler input')
    engine = DDAR(list(by_name.values()))
    compiled = payload['compiled']
    # All premises are sourced from graph construction *definitions*; the
    # bridge deliberately does not infer premises from floating-point numbers.
    for item in compiled['predicates']:
        p = AGPredicate(item['name'], [by_name[x] for x in item['points']], item['constants'])
        try:
            engine.force_pred(p)
        except Exception as exc:
            raise ValueError('Cannot force construction premise %s %s (%s): %s'
                             % (p.name, item['points'], item.get('source', '?'), exc)) from exc
    engine.deduction_closure(verbose=False, progress_dot=False)
    result = []
    for item in payload.get('queries', []):
        status = 'proved'
        reason = 'AlphaGeometry2 官方 DDAR 的闭包检验；官方实现不提供完整证明依赖回溯'
        for pred in item.get('predicates', []):
            try:
                p = AGPredicate(pred['name'], [by_name[x] for x in pred['points']], pred.get('constants', []))
                if not engine.check_pred(p):
                    status = 'unknown'
                    reason = 'AlphaGeometry2 DDAR 闭包尚未证明，不表示命题为假'
                    break
            except (KeyError, IndexError, AssertionError, ValueError, ZeroDivisionError, StopIteration) as exc:
                status = 'unsupported'
                reason = '当前构造 / 谓词不能安全表达: ' + type(exc).__name__ + ': ' + str(exc)
                break
        if not item.get('predicates'):
            status, reason = 'unsupported', '不存在可安全转换的 AlphaGeometry2 查询'
        result.append({'id': item['id'], 'status': status, 'reason': reason})
    return {
        'engine': 'AlphaGeometry2-official',
        'queries': result,
        'elapsedMs': (time.perf_counter() - start) * 1000,
        'pointCount': len(by_name),
        'premiseCount': len(compiled['predicates']),
    }


def solve_json(text):
    return json.dumps(execute(json.loads(text)), ensure_ascii=False)
