"""Read-only AG2 proof extraction from the session's append-only DD / AR audit.

The recorder maintains its own equation DAG. This module never calls a DDAR
search/force routine, edits the engine, or appends to the audit. An exact,
rational, small-support linear replay is done in separate Python dictionaries.
"""
from collections import deque
from fractions import Fraction
import re

AR_KINDS = {'para': 'ARang', 'perp': 'ARang', 'eqangle': 'ARang',
            'angeq': 'ARang', 'aconst': 'ARang',
            'cong': 'ARmul', 'eqratio': 'ARmul', 'rconst': 'ARmul'}


def _vector(expr):
    """Decode LinComb.__str__ (the actual pre-elimination coefficients).

    Variable names are opaque, e.g. d(A B), log(|A B|), log(2), pi.
    Reject malformed terms rather than fabricate a dependency.
    """
    if expr == '0':
        return {}
    result = {}
    for term in expr.split(' + '):
        match = re.match(r'^(-?\d+(?:/\d+)?)\*(.+)$', term)
        if match:
            coef, var = Fraction(match[1]), match[2]
        else:
            coef, var = Fraction(1), term
        if not var:
            raise ValueError('bad linear variable')
        result[var] = result.get(var, Fraction(0)) + coef
        if not result[var]:
            del result[var]
    return result


def _comb_vector(comb):
    return {str(var): Fraction(coef) for var, coef in comb.d.items() if coef}


def _simplified_zero(core, target, kind):
    from elimination import LinComb, angle_unit
    # A fresh LinComb: simplify mutates only the caller's copy.
    variable_map = {}
    for obj in core.instantiated:
        variable_map[str(obj)] = obj
        for other in core.instantiated[obj].d:
            variable_map[str(other)] = other
    original = {}
    for var, coef in target.items():
        if var == 'pi':
            original[angle_unit] = coef
        elif var in variable_map:
            original[variable_map[var]] = coef
        elif coef:
            return False
    comb = LinComb(original)
    core.simplify(comb)
    if not comb.d:
        return True
    return kind == 'ARang' and set(comb.d) == {angle_unit} and comb.d[angle_unit].denominator == 1


def _subtract(target, row, factor):
    for var, coef in row.items():
        value = target.get(var, Fraction(0)) - factor * coef
        if value:
            target[var] = value
        else:
            target.pop(var, None)


def _replay(target, equations, kind):
    """Exact row elimination; return actually used equation IDs, or None.

    Angle reasoning is over R/Z: the final integer coefficient of pi is zero
    mod pi. We do not treat an arbitrary fractional pi as free.
    """
    basis = {}
    for num, vec in equations:
        row = dict(vec)
        used = {num: Fraction(1)}
        for pivot in basis:
            if pivot in row:
                bvec, bdeps = basis[pivot]
                factor = row[pivot] / bvec[pivot]
                _subtract(row, bvec, factor)
                _subtract(used, bdeps, factor)
        free = sorted(v for v in row if v != 'pi')
        # A pure constant equation should not become a free normalization axiom.
        if free:
            basis[free[0]] = (row, used)
    row = dict(target)
    used = {}
    for pivot in basis:
        if pivot in row:
            bvec, bdeps = basis[pivot]
            factor = row[pivot] / bvec[pivot]
            _subtract(row, bvec, factor)
            _subtract(used, bdeps, -factor)
    if not row or (kind == 'ARang' and set(row) == {'pi'} and row['pi'].denominator == 1):
        return set(used)
    return None


def _ancestor_equations(session, roots, kind):
    pending = list(roots)
    seen = set()
    while pending:
        num = pending.pop()
        if not isinstance(num, int) or num >= 0 or num in seen:
            continue
        node = session.eq_nodes.get(num)
        if node is None:
            continue
        seen.add(num)
        pending.extend(node['parents'])
    return [(n, _vector(session.eq_nodes[n]['expression']))
            for n in sorted(seen, reverse=True) if session.eq_nodes[n]['type'] == kind]


def _equation_refs(session, eqids):
    refs = set()
    for eqid in eqids:
        node = session.eq_nodes[eqid]
        direct = node.get('direct', ())
        # Source collected from the DD/input event, *before* elimination added
        # earlier pivot IDs to the bookkeeping. In absence of a direct source,
        # conservatively retain the old pivot ancestry (never guess an axiom).
        roots = set(direct) if direct else session._roots(eqid)
        if not roots or any(not isinstance(k, int) or k not in session.audit_index for k in roots):
            return None
        refs.update(roots)
    return refs


def _algebra(session, data):
    """Read the original named distances/directions, not substituted points."""
    engine = session.engine
    points = [session.points[p] for p in data['points']]
    kind = AR_KINDS[data['name']]
    if kind == 'ARang':
        from parse import AGPredicate
        obj = engine.pred_to_angle(AGPredicate(data['name'], points,
                     [Fraction(str(x)) for x in data.get('constants', [])]))
        core = engine.elim_angle.core
    else:
        distances = engine.pair_to_dist_mul
        typ = data['name']
        if typ == 'cong':
            obj = distances[points[0], points[1]] / distances[points[2], points[3]]
        elif typ == 'eqratio':
            obj = (distances[points[0], points[1]] / distances[points[2], points[3]] /
                   (distances[points[4], points[5]] / distances[points[6], points[7]]))
        else:
            obj = distances[points[0], points[1]] / distances[points[2], points[3]] / Fraction(str(data['constants'][0]))
        core = engine.elim_dist_mul.core
    return kind, core, _comb_vector(obj.comb), [x for x in core.explain(obj.comb) if x < 0]


def _min_ar(session, data, before_event=None, preferred_sources=None):
    kind, core, target, roots = _algebra(session, data)
    if before_event is None and not _simplified_zero(core, target, kind):
        return None
    # Constants / reflexive identities have no algebraic antecedents.
    if not target or (kind == 'ARang' and set(target) == {'pi'} and target['pi'].denominator == 1):
        return set()
    if before_event is None:
        candidates = _ancestor_equations(session, roots, kind)
    else:
        candidates = []
        for n in sorted(session.eq_nodes, reverse=True):
            row = session.eq_nodes[n]
            if row['type'] != kind or row.get('audit_at', before_event) >= before_event:
                continue
            direct = set(row.get('direct', ()))
            if preferred_sources is not None and (not direct or not direct <= preferred_sources):
                continue
            candidates.append((n, _vector(row['expression'])))
    if not candidates:
        return None
    chosen = _replay(target, candidates, kind)
    if chosen is None:
        return None
    # Exact deletion minimization on the selected sparse representation.
    # This is inclusion-minimal, not the globally NP-hard sparsest solution.
    used = [(n, vec) for n, vec in candidates if n in chosen]
    for n, _ in sorted(used, key=lambda pair: len(session.eq_nodes[pair[0]].get('parents', ())), reverse=True):
        attempt = [(k, v) for k, v in used if k != n]
        if _replay(target, attempt, kind) is not None:
            used = attempt
    if _replay(target, used, kind) is None:
        return None
    return _equation_refs(session, [n for n, _ in used])


def _overlap_path(session, start, finish):
    if start == finish:
        return set()
    graph = {}
    for event in session.audit_index.values():
        for statement in event.get('conclusions', ()):
            match = re.fullmatch(r'overlap (\S+) (\S+)', statement)
            if match:
                a, b = match.groups()
                graph.setdefault(a, []).append((b, event['id']))
                graph.setdefault(b, []).append((a, event['id']))
    queue = deque([(start, frozenset())]); seen = {start}
    while queue:
        name, refs = queue.popleft()
        for adjacent, ident in graph.get(name, ()):
            if adjacent in seen:
                continue
            nxt = refs | {ident}
            if adjacent == finish:
                return set(nxt)
            seen.add(adjacent)
            queue.append((adjacent, nxt))
    return None


def _structural(session, data):
    kind = data['name']
    target = set(data['points'])
    choices = []
    for ref, points in session.sources.get(kind, []):
        if target <= points:
            choices.append(ref)
    if kind == 'cyclic':
        # Some AG2 circle constructions use a center-aware native predicate.
        # Its point set, excluding the listed center(s), directly entails
        # ordinary concyclicity without any extra inference.
        for event in session.audit_index.values():
            if event['type'] != 'PREMISE':
                continue
            for statement in event.get('conclusions', ()):
                tokens = statement.split()
                if tokens and tokens[0] == 'cyclic_with_centers':
                    try:
                        count = int(tokens[-1]); members = tokens[1:-1][count:]
                        if len(members) >= 4 and target <= set(members):
                            choices.append(event['id'])
                    except (ValueError, IndexError):
                        pass
    if not choices:
        return None
    # An actual logged assertion must cover the requested points.
    return {min(choices, key=lambda ident: (len(session.audit_index[ident].get('premises', ())), ident))}


def _goal_sources(session, data):
    kind = data['name']
    original = list(data['points'])
    if kind in AR_KINDS:
        # If the requested conclusion is literally an input premise, no
        # algebraic reconstruction is needed (nor should it add dependencies).
        premise = session.input_by_label.get(session._label(data))
        if premise is not None:
            return {premise}, False
        try:
            answer = _min_ar(session, data)
            if answer is not None:
                return answer, False
        except (AssertionError, IndexError, KeyError, ValueError, ZeroDivisionError):
            pass
    else:
        answer = _structural(session, data)
        if answer is not None:
            return answer, False
        if kind == 'overlap' and len(original) == 2:
            path = _overlap_path(session, *original)
            if path is not None:
                return path, True
        if kind == 'coll' and len(original) == 3:
            # Exact angle elimination can entail an incidence fact even when
            # the native incidence table has no separate logged coll event.
            # Distinct-line directions AB and AC imply A,B,C are collinear.
            try:
                result = _min_ar(session, {'name':'para',
                                           'points':[original[0],original[1],original[0],original[2]],
                                           'constants':[]})
                if result is not None:
                    return result, False
            except (KeyError, ValueError, AssertionError, IndexError, ZeroDivisionError):
                pass
    canonical = [str(session.engine.point_subst[session.points[x]]) for x in original]
    if canonical == original:
        return None, False
    replacement = dict(data, points=canonical)
    try:
        base = _min_ar(session, replacement) if kind in AR_KINDS else _structural(session, replacement)
    except (AssertionError, IndexError, KeyError, ValueError, ZeroDivisionError):
        return None, False
    if base is None:
        return None, False
    all_refs = set(base)
    for a, b in zip(original, canonical):
        path = _overlap_path(session, a, b)
        if path is None:
            return None, False
        all_refs.update(path)
    return all_refs, True


def _point_tokens(session, statements):
    names = set(session.points)
    found = set()
    for item in statements:
        for word in str(item).split()[1:]:
            if word in names:
                found.add(word)
    return found


def _backtrace(session, seeds):
    needed = set()
    visiting = set()
    parents_for = {}
    def visit(ident):
        if ident in needed:
            return
        if ident in visiting or ident not in session.audit_index:
            raise ValueError('推导日志的前提缺失或出现循环：#' + str(ident))
        visiting.add(ident)
        event = session.audit_index[ident]
        parents = list(event.get('premises', ()))
        if event['type'] == 'AR' and event.get('conclusions'):
            data = session._as_pred(event['conclusions'][0])
            if data is not None and data['name'] in AR_KINDS:
                try:
                    minimal = _min_ar(session, data, before_event=ident,
                                      preferred_sources=set(parents))
                except (ValueError, AssertionError, KeyError, IndexError, ZeroDivisionError):
                    minimal = None
                if minimal is not None and all(x < ident for x in minimal):
                    parents = sorted(minimal)
        parents_for[ident] = parents
        for parent in parents:
            visit(parent)
        visiting.remove(ident)
        needed.add(ident)
    for ident in seeds:
        visit(ident)
    return sorted(needed), parents_for


def _format_predicate(label):
    parts = label.split()
    if not parts:
        return label
    typ, a = parts[0], parts[1:]
    if typ == 'coll' and len(a) >= 3:
        return ','.join(a) + ' 共线'
    if typ == 'cyclic' and len(a) >= 4:
        return ','.join(a) + ' 共圆'
    if typ == 'overlap' and len(a) == 2:
        return a[0] + ' = ' + a[1] + '（点重合）'
    if typ in ('perp', 'para', 'cong') and len(a) == 4:
        return a[0] + a[1] + {'perp':' ⟂ ', 'para':' ∥ ', 'cong':' = '}[typ] + a[2] + a[3]
    if typ == 'eqratio' and len(a) == 8:
        return f'{a[0]}{a[1]}:{a[2]}{a[3]} = {a[4]}{a[5]}:{a[6]}{a[7]}'
    if typ == 'rconst' and len(a) == 5:
        return f'{a[0]}{a[1]}:{a[2]}{a[3]} = {a[4]}'
    if typ == 'eqangle' and len(a) == 8:
        return f'∠({a[0]}{a[1]}-{a[2]}{a[3]}) = ∠({a[4]}{a[5]}-{a[6]}{a[7]})'
    if typ == 'aconst' and len(a) == 5:
        return f'∠({a[0]}{a[1]}-{a[2]}{a[3]}) = {a[4]}°'
    if typ == 'angeq' and len(a) == 9 and a[-1] == '1,-1,-1,1,0':
        return f'∠({a[0]}{a[1]}-{a[2]}{a[3]}) = ∠({a[4]}{a[5]}-{a[6]}{a[7]})'
    if typ == 'similar' and len(a) == 6:
        return '△'+''.join(a[:3])+' ∼ △'+''.join(a[3:])
    if typ == 'representative' and len(a) == 3 and a[1] == '->':
        return f'{a[0]} → {a[2]}（点合并）'
    match = re.fullmatch(r'd\((\S+) (\S+)\) \+ -1\*d\((\S+) (\S+)\) = 0', label)
    if match:
        u,v,x,y = match.groups()
        return f'{u}{v} ∥ {x}{y}'
    return label


def render(session):
    if not session.queries or not session.goal_mode:
        return {'available': False, 'reason': '请先使用「尝试证明选中命题」运行 AG2；闭包没有指定待整理的结论。'}
    query = session.queries[0]
    if session.found.get(query['id'], {}).get('status') != 'proved':
        return {'available': False, 'reason': 'AG2 尚未证明本次提交的命题，无法整理出完整证明。'}
    goals = list(query.get('predicates', []))
    if not goals:
        return {'available': False, 'reason': '没有可证明的 AG2 原生命题。'}
    refs = set()
    substitutions = False
    for data in goals:
        result, replaced = _goal_sources(session, data)
        if result is None:
            return {'available': False,
                    'reason': 'AG2 已判断命题成立，但当前日志无法还原其全部严格依赖（可能涉及点重合、共线或共圆状态）。未输出不完整的证明。'}
        refs.update(result)
        substitutions |= replaced
    try:
        ids, parents_for = _backtrace(session, refs)
    except ValueError as exc:
        return {'available': False, 'reason': str(exc)}
    events = session.audit_index
    # Collinear length / direction encodings are transparent DD bookkeeping:
    # the collinearity source already implies the equation. Present the actual
    # source instead of four distracting conversions in the readable proof.
    transparent = {i for i in ids if events[i].get('type') == 'DD' and
                   events[i].get('rule') in ('collinear length addition',
                                             'collinear direction equivalence') and
                   parents_for.get(i)}
    def visible_refs(ref_list):
        expanded = set()
        def add(ident):
            if ident in transparent:
                for p in parents_for[ident]:
                    add(p)
            else:
                expanded.add(ident)
        for ident in ref_list:
            add(ident)
        return sorted(expanded)

    shown = [i for i in ids if i not in transparent]
    inputs = [events[i] for i in shown if events[i]['type'] == 'PREMISE']
    derived = [events[i] for i in shown if events[i]['type'] != 'PREMISE']
    goal_labels = [session._label(row) for row in goals]
    statements = goal_labels + [c for i in shown for c in events[i].get('conclusions', [])]
    required_points = _point_tokens(session, statements)
    virtual = {p['name'] for p in session.payload['compiled']['points'] if p.get('virtual')}
    virtual.update(session.payload.get('auxiliaryPoints', []))
    regular = sorted(required_points - virtual)
    aux = sorted(required_points & virtual)
    premise_inputs = [e for e in inputs if not (_point_tokens(session,e.get('conclusions',())) & set(aux))]
    aux_inputs = [e for e in inputs if e not in premise_inputs]
    label_map = {event['id']:f'{i:02d}' for i,event in enumerate(inputs)}
    lines = ['==========================', ' * From theorem premises:',
             (' '.join(regular) + ' : Points') if regular else '(无普通点)']
    for event in premise_inputs:
        for conclusion in event.get('conclusions', []):
            lines.append(f'{_format_predicate(conclusion)} [{label_map[event["id"]]}]')
    lines.extend(['', ' * Auxiliary Constructions:',
                  (' '.join(aux) + ' : Points') if aux else '(无新增辅助点)'])
    for event in aux_inputs:
        for conclusion in event.get('conclusions', []):
            lines.append(f'{_format_predicate(conclusion)} [{label_map[event["id"]]}]')
    lines.extend(['', ' * Proof steps:'])
    def cited_fact(ident):
        item = events[ident]
        descriptions = '；'.join(_format_predicate(con) for con in item.get('conclusions', ()))
        return descriptions + f' [{label_map[ident]}]'
    for idx, event in enumerate(derived, 1):
        new_label = f'{len(inputs)+idx-1:02d}'
        parents = ' & '.join(cited_fact(p) for p in visible_refs(parents_for.get(event['id'], ()))) or '恒等关系'
        if not event.get('conclusions'):
            return {'available':False,'reason':f'日志 #{event["id"]} 没有可展示的结论。'}
        conclusions = '；'.join(_format_predicate(c) for c in event['conclusions'])
        lines.append(f'{idx:03d}. {parents} ⇒ {conclusions} [{new_label}]')
        label_map[event['id']] = new_label
    ending = ' & '.join(cited_fact(r) for r in visible_refs(refs)) or '恒等关系'
    target = ' & '.join(_format_predicate(s) for s in goal_labels)
    lines.append(f'{len(derived)+1:03d}. {ending} ⇒ {target}' + ('（点重合替换）' if substitutions else ''))
    lines.append('==========================')
    return {'available': True, 'content': '\n'.join(lines), 'eventCount':len(ids),
            'proofSteps':len(derived)+1, 'goal':goal_labels}
