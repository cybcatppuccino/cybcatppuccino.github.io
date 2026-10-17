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
    return [(n, {row['variable']:Fraction(row['numerator'],row['denominator']) for row in session.eq_nodes[n]['terms']})
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


def _historic_equations(session, kind, before_event=None, max_equation_revision=None,
                        preferred_sources=None):
    """Original AR inputs visible at the *read* revision, not pivot ancestry.

    `core.explain` returns the final reduced pivot identifiers.  It does NOT
    list every original input row needed to reproduce that pivot: an equation
    may have been reduced by other pivots, whose history is not necessarily
    transitively recorded in `instantiated_sources`.  Searching only their
    ancestors can therefore fail for a perfectly valid AR proof (2019_p6).
    """
    result=[]
    for n in sorted(session.eq_nodes, reverse=True):
        row=session.eq_nodes[n]
        if row['type'] != kind:
            continue
        if max_equation_revision is not None and -n > max_equation_revision:
            continue
        if before_event is not None and row.get('audit_at', before_event) >= before_event:
            continue
        direct=set(row.get('direct',()))
        if preferred_sources is not None and (not direct or not direct <= preferred_sources):
            continue
        result.append((n, {term['variable']:Fraction(term['numerator'],term['denominator'])
                           for term in row['terms']}))
    return result


def _min_ar(session, data, before_event=None, preferred_sources=None,
            max_equation_revision=None, target_terms=None):
    kind, core, target, roots = _algebra(session, data)
    if target_terms is not None:
        # Preserve the original geometric variable names at goal-observation
        # time; a later point merge must not silently alter the question.
        target={r['variable']:Fraction(r['numerator'],r['denominator'])
                for r in target_terms}
    if before_event is None and not _simplified_zero(core, target, kind):
        return None
    if not target or (kind == 'ARang' and set(target) == {'pi'} and
                      target['pi'].denominator == 1):
        return set()

    # Try the small set normally recovered from the pivot DAG first.  A DD
    # antecedent restricted to its own earlier parents must *never* borrow
    # later equations.  A goal witness similarly fixes the equation revision.
    if before_event is None and preferred_sources is None:
        candidates=_ancestor_equations(session, roots, kind)
        if max_equation_revision is not None:
            candidates=[(n,v) for n,v in candidates if -n <= max_equation_revision]
    else:
        candidates=[]
    chosen=_replay(target,candidates,kind) if candidates else None
    if chosen is None:
        candidates=_historic_equations(session,kind,before_event,
                                       max_equation_revision,preferred_sources)
        chosen=_replay(target,candidates,kind) if candidates else None
    if chosen is None:
        return None
    # Inclusion-minimal subset of the *selected original exact equations*.
    # Never treat the entire row span or numerical tests as premises.
    used=[(n,vec) for n,vec in candidates if n in chosen]
    for n,_ in sorted(used,key=lambda item:len(session.eq_nodes[item[0]].get('parents',())),reverse=True):
        trial=[(k,v) for k,v in used if k!=n]
        if _replay(target,trial,kind) is not None:
            used=trial
    if _replay(target,used,kind) is None:
        return None
    return _equation_refs(session,[n for n,_ in used])


# DDAR theorem identifiers are preserved in the log; the reader displays a
# short human mnemonic, not the source file's implementation method name.
DD_SLOGANS = {
    'T1':'AA相似', 'T2':'SSS相似', 'T3':'SAS相似', 'T4':'SSA相似',
    'T5':'相似角对应', 'T6':'相似边成比例',
    'C1':'零角判共线', 'C2':'圆周角定理', 'C3':'同圆圆周角',
    'C4':'圆心角性质', 'C5':'共圆唯一性', 'C6':'同圆半径相等',
    'C7':'共线合并', 'C8':'共线段长加法', 'C9':'共线方向一致',
    'C10':'同圆继承', 'R1':'等半径判圆', 'R2':'圆心确定',
    'R3':'小圆候选', 'P1':'圆心唯一性', 'P2':'交点唯一性',
    'P3':'重合点合并', 'P4':'共线点继承', 'P5':'共圆点继承',
    'P6':'重合点距离替换', 'P7':'点名等价替换',
    'D1':'乘法转加法', 'D2':'加法转乘法',
    'B1':'梅涅劳斯定理', 'B2':'梅涅劳斯逆定理',
    'B3':'塞瓦定理', 'B4':'塞瓦逆定理',
    'B5':'四线段比例', 'B5ab':'正弦比桥接',
    'B5ac':'正弦比桥接', 'B5bb':'正弦比桥接',
    'B5bc':'正弦比桥接', 'B5cc':'正弦比桥接',
    'B6':'比例共线逆推', 'B7':'比例交点逆推',
    'B8':'圆内交弦定理', 'B9':'交弦定理逆定理',
    'B10':'角平分线定理', 'B11':'切割线定理',
    'A1':'弧长转弦长', 'A2':'弦长转弧长',
}


def _overlap_path(session, start, finish):
    # Prefer exact merge edges: unlike broad overlap labels, these preserve
    # the historical equality forest and their individual DD origins.
    if hasattr(session, 'point_merges'):
        result = session._point_path(start, finish)
        if result is not None:
            return result
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
    if kind in ('coll','cyclic') and hasattr(session,'_structural_candidates'):
        choices = session._structural_candidates(kind,data['points'])
        if choices:
            return min(choices, key=session._support_cost)
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


def _goal_sources(session, data, max_equation_revision=None, target_terms=None):
    kind = data['name']
    original = list(data['points'])
    if kind == 'coll' and len(original) >= 3:
        # AG2 also accepts the degenerate incidence (A,B,A): any two points
        # define a line. Once A=B has been proved, (A,B,C) likewise becomes
        # a two-point incidence, whose ONLY hypothesis is that equality.
        by_representative={}
        for name in original:
            root=str(session.engine.point_subst[session.points[name]])
            by_representative.setdefault(root,set()).add(name)
        if len(by_representative) <= 2:
            required=set()
            for names in by_representative.values():
                if len(names)>1:
                    anchor=min(names)
                    for other in names-{anchor}:
                        path=_overlap_path(session,anchor,other)
                        if path is None:return None, False
                        required.update(path)
            return required, bool(required)
    if kind in AR_KINDS:
        # An original named relation + minimal point-equivalence paths is
        # usually substantially smaller than all AR merge consequences.
        direct = session._named_relation_candidates(data) if hasattr(session,'_named_relation_candidates') else []
        if direct:
            best=min(direct,key=session._support_cost)
            return best, any(ref in {p[2] for p in session.point_merges} for ref in best)
        # If the requested conclusion is literally an input premise, no
        # algebraic reconstruction is needed (nor should it add dependencies).
        premise = session.input_by_label.get(session._label(data))
        if premise is not None:
            return {premise}, False
        try:
            answer = _min_ar(session, data, max_equation_revision=max_equation_revision,
                             target_terms=target_terms)
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
        unresolved = event.get('unresolvedReads', ())
        if unresolved:
            # There must be no silent omission of an antecedent that the
            # native DD operation inspected. Algebraic identities have been
            # identified separately by the session recorder.
            raise ValueError('日志 #' + str(ident) + ' 的 DD 前提缺少历史来源：' +
                             '；'.join(unresolved[:3]) + '。未输出不完整的证明。')
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


class _PointAliases(dict):
    """Display-only aliases plus the actual point lexicon of this proof."""
    def __init__(self, mapping, point_names):
        super().__init__(mapping)
        self.point_names = frozenset(map(str, point_names))


def _triple_angle(token, aliases):
    """Decode sin-bridge angle(ABC), including multi-letter/virtual points.

    The historical log intentionally concatenates the three ORIGINAL point
    names without separators.  Reconstruct them from the exact, read-only
    session point lexicon; do not guess a point name from character positions.
    """
    names = getattr(aliases, 'point_names', ())
    # Prefer longer names (e.g. ma over m+a), but reject ambiguity only after
    # examining all complete three-point parses.
    by_initial = {}
    for name in names:
        if name:
            by_initial.setdefault(name[0], []).append(name)
    for options in by_initial.values():
        options.sort(key=lambda x: (-len(x), x))
    parsed = []
    def split(offset, parts):
        if len(parts) == 3:
            if offset == len(token):
                parsed.append(tuple(parts))
            return
        for name in by_initial.get(token[offset:offset+1], ()): 
            if token.startswith(name, offset):
                split(offset + len(name), parts + [name])
    split(0, [])
    if not parsed:
        return None
    # A coincident point may legitimately occur twice; the first valid parse
    # is deterministic, preserving the original ordered angle arguments.
    points = parsed[0]
    displayed = [_name(point, aliases) for point in points]
    if all(len(point) == 1 and point.isalpha() for point in displayed):
        return '∠' + ''.join(displayed)
    return '∠(' + ', '.join(displayed) + ')'


def _sine_angle_relation(label, aliases):
    """Present historical ARang sine witnesses as readable angle relations."""
    match = re.fullmatch(r'angle\((.+?)\)\s*≡\s*(-?\s*)angle\((.+?)\)\s*mod\s*π', label)
    if not match:
        return None
    left = _triple_angle(match.group(1), aliases)
    right = _triple_angle(match.group(3), aliases)
    if left is None or right is None:
        return None
    return left + ' = ' + ('-' if '-' in match.group(2) else '') + right


def _name(point, aliases):
    """Presentation only: never replace historical point names in the audit."""
    return aliases.get(point, point) if aliases else point


def _segment(a, b, aliases):
    first, last = _name(a, aliases), _name(b, aliases)
    # Two named points are conventionally written AB; descriptive aliases
    # need a separator so that a constructed midpoint is not misread.
    return first + ('–' if (a.startswith('@') or b.startswith('@')) else '') + last


def _angle(first, second, aliases):
    return '∠(' + _segment(*first, aliases) + ' - ' + _segment(*second, aliases) + ')'


def _signed_terms(expression):
    """Parse the LinComb *display*, without any dependence on its internals.

    Coefficients remain exact Fraction values.  This is solely a renderer;
    it never uses the displayed text to establish a proof dependency.
    """
    terms = []
    for part in expression.split(' + '):
        match = re.fullmatch(r'(-?\d+(?:/\d+)?)\*(.+)', part)
        if match:
            value, term = Fraction(match.group(1)), match.group(2)
        elif part.startswith('-'):
            value, term = Fraction(-1), part[1:]
        else:
            value, term = Fraction(1), part
        terms.append((value, term))
    return terms


def _coefficient_product(coef, value):
    if coef == 1:
        return value
    if coef.denominator == 1:
        return str(coef.numerator) + ' * ' + value
    return '(' + str(coef) + ') * ' + value


def _positive_equation(terms, description=''):
    left = [_coefficient_product(c, var) for c, var in terms if c > 0]
    right = [_coefficient_product(-c, var) for c, var in terms if c < 0]
    result = ' + '.join(left) if left else '0'
    result += ' = ' + (' + '.join(right) if right else '0')
    return description + result


def _direction_relation(label, aliases):
    match = re.fullmatch(r'(.+) = 0', label)
    if not match:
        return None
    terms = _signed_terms(match.group(1))
    directions = []
    for coef, variable in terms:
        parsed = re.fullmatch(r'd\((\S+) (\S+)\)', variable)
        if not parsed:
            return None
        directions.append((coef, parsed.groups()))
    coefs = [coef for coef, _ in directions]
    if coefs == [Fraction(1), Fraction(-1)]:
        return (_segment(*directions[0][1], aliases) + ' ∥ ' +
                _segment(*directions[1][1], aliases))
    if coefs == [Fraction(1), Fraction(-1), Fraction(-1), Fraction(1)]:
        return (_angle(directions[0][1], directions[1][1], aliases) + ' = ' +
                _angle(directions[2][1], directions[3][1], aliases) + '')
    items = [(coef, '方向(' + _segment(*pair, aliases) + ')')
             for coef, pair in directions]
    return '角度关系：' + _positive_equation(items)


def _additive_length_relation(label, aliases):
    match = re.fullmatch(r'(.+) = 0', label)
    if not match:
        return None
    terms = []
    for coef, variable in _signed_terms(match.group(1)):
        parsed = re.fullmatch(r'\|(\S+) (\S+)\|', variable)
        if not parsed:
            return None
        terms.append((coef, _segment(*parsed.groups(), aliases)))
    return '线段的代数长度满足：' + _positive_equation(terms)


def _multiplicative_length_relation(label, aliases):
    """ARmul uses '= 1' for a product identity, NOT a logarithm equal to 1.

    In particular, log(|AB|)-log(|CD|)=1 in the engine display means
    AB/CD=1, and it would be mathematically wrong to print ln(AB/CD)=1.
    """
    match = re.fullmatch(r'(.+) = 1', label)
    if not match:
        return None
    lhs, rhs = [], []
    for coef, variable in _signed_terms(match.group(1)):
        if variable.startswith('log(|') and variable.endswith('|)'):
            parsed = re.fullmatch(r'log\(\|(\S+) (\S+)\|\)', variable)
            if not parsed:
                return None
            value = _segment(*parsed.groups(), aliases)
        else:
            parsed = re.fullmatch(r'log\((\d+(?:/\d+)?(?:\.\d+)?)\)', variable)
            if not parsed:
                return None
            value = parsed.group(1)
        if coef == 0:
            continue
        if abs(coef) != 1:
            value += '^(' + str(abs(coef)) + ')'
        (lhs if coef > 0 else rhs).append(value)
    return ('线段长度的乘积关系：' + (' * '.join(lhs) if lhs else '1') +
            ' = ' + (' * '.join(rhs) if rhs else '1'))


def _length_product_relation(label, aliases):
    """Present Menelaus/Ceva/chord-product statements as length products."""
    if ' * ' not in label or ' = ' not in label:
        return None
    left, right = label.split(' = ', 1)
    sides = (left.split(' * '), right.split(' * '))
    if not all(sides) or not all(re.fullmatch(r'[\w@|:.-]+', term) for side in sides for term in side):
        return None
    def present(term):
        # DD can concatenate a virtual midpoint and an ordinary endpoint
        # without separators: `g@mid:g|h` means segment G--midpoint(GH).
        for raw in sorted(aliases, key=len, reverse=True):
            term = term.replace(raw, '–' + aliases[raw] + '–')
        return re.sub('–{2,}', '–', term).strip('–')
    return ('线段长度的乘积关系：' + ' * '.join(present(v) for v in sides[0]) +
            ' = ' + ' * '.join(present(v) for v in sides[1]))


def _angle_offset(angle):
    if not angle:
        return ''
    return (' + ' if angle > 0 else ' - ') + str(abs(angle)) + '°'


def _format_predicate(label, aliases=None):
    aliases = aliases if aliases is not None else {}
    sine_relation = _sine_angle_relation(label, aliases)
    if sine_relation is not None:
        return sine_relation
    parts = label.split()
    if not parts:
        return label
    typ, a = parts[0], parts[1:]
    if typ == 'coll' and len(a) >= 3:
        return ','.join(_name(p, aliases) for p in a) + ' 共线'
    if typ == 'cyclic' and len(a) >= 3:
        # Some native circle assertions have three points and a center read.
        return ','.join(_name(p, aliases) for p in a) + ' 位于同一圆上'
    if typ == 'cyclic_with_centers' and len(a) >= 3:
        try:
            count = int(a[-1]); centers = a[:count]; points = a[count:-1]
            if count >= 0 and len(points) >= 2:
                center_text = ','.join(_name(p, aliases) for p in centers)
                return (','.join(_name(p, aliases) for p in points) +
                        ' 位于同一圆上' + ('（圆心 ' + center_text + '）' if centers else ''))
        except (ValueError, IndexError):
            pass
    if typ == 'overlap' and len(a) == 2:
        return _name(a[0], aliases) + ' = ' + _name(a[1], aliases) + '（点重合）'
    if typ in ('perp', 'para', 'cong') and len(a) == 4:
        return (_segment(a[0], a[1], aliases) +
                {'perp':' ⟂ ', 'para':' ∥ ', 'cong':' = '}[typ] +
                _segment(a[2], a[3], aliases))
    if typ == 'eqratio' and len(a) == 8:
        return (f'{_segment(a[0],a[1],aliases)} : {_segment(a[2],a[3],aliases)} = '
                f'{_segment(a[4],a[5],aliases)} : {_segment(a[6],a[7],aliases)}')
    if typ == 'rconst' and len(a) == 5:
        return f'{_segment(a[0],a[1],aliases)} : {_segment(a[2],a[3],aliases)} = {a[4]}'
    if typ == 'eqangle' and len(a) == 8:
        return _angle((a[0],a[1]),(a[2],a[3]),aliases) + ' = ' + _angle((a[4],a[5]),(a[6],a[7]),aliases)
    if typ == 'aconst' and len(a) == 5:
        return _angle((a[0],a[1]),(a[2],a[3]),aliases) + ' = ' + a[4] + '°'
    if typ == 'angeq' and len(a) in (8,9):
        try:
            coeffs = ([Fraction(v) for v in a[8].split(',')] if len(a) == 9
                      else [Fraction(1),Fraction(-1),Fraction(-1),Fraction(1),Fraction(0)])
        except ValueError:
            coeffs = []
        if len(coeffs) == 5 and coeffs[:2] == [1,-1] and coeffs[2:4] in ([-1,1],[1,-1]):
            first = _angle((a[0],a[1]),(a[2],a[3]),aliases)
            second = (_angle((a[4],a[5]),(a[6],a[7]),aliases)
                      if coeffs[2:4] == [-1,1] else
                      _angle((a[6],a[7]),(a[4],a[5]),aliases))
            return first + ' = ' + second + _angle_offset(-coeffs[4]) + ''
        if len(coeffs) == 5:
            terms = [(coef, '方向(' + _segment(a[2*i],a[2*i+1],aliases) + ')')
                     for i,coef in enumerate(coeffs[:4]) if coef]
            terms.append((coeffs[4], '1°'))
            return '角度关系：' + _positive_equation(terms)
    if typ == 'similar' and len(a) == 6:
        return '△'+''.join(_name(x,aliases) for x in a[:3])+' ∼ △'+''.join(_name(x,aliases) for x in a[3:])
    if typ == 'representative' and len(a) == 3 and a[1] == '->':
        return f'{_name(a[0],aliases)} → {_name(a[2],aliases)}（点合并）'
    # Internal AR equations (direction, additive length, and logarithmic
    # multiplicative length) are expressions, not named AG2 predicates.
    for formatter in (_direction_relation, _additive_length_relation,
                      _multiplicative_length_relation, _length_product_relation):
        presented = formatter(label, aliases)
        if presented is not None:
            return presented
    return label


def _event_conclusion(event, label, aliases=None):
    """Preserve the center witness when presenting a native cyclic event."""
    shown = _format_predicate(label, aliases)
    if label.startswith('cyclic ') and event.get('type') == 'DD':
        centers = (event.get('detail') or {}).get('centers') or ()
        if centers:
            points = label.split()[1:]
            if len(points) == 3 and len(centers) == 1:
                shown = (','.join(_name(p, aliases) for p in points) + ' 位于以 ' +
                         _name(centers[0], aliases) + ' 为圆心的圆上')
            else:
                shown += '（圆心 ' + ','.join(_name(p,aliases) for p in centers) + '）'
    return shown


def _display_aliases(session, statements):
    """Give synthetic construction tokens consistent, human-readable names.

    Internal names never change in the engine or dependencies.  Midpoint,
    direction-anchor and circumcenter tokens get an explanatory glossary.
    """
    items = sorted((p for p in session.points if p.startswith('@') and
                    any(p in statement for statement in statements)), key=str)
    aliases = {}; descriptions = []; counts = {'@mid:':0,'@line:':0,'@circumcenter:':0,'@other:':0}
    for point in items:
        kind = next((prefix for prefix in counts if point.startswith(prefix)), None)
        if kind is None:
            kind = '@other:'
        counts[kind] += 1
        index = counts[kind]
        if kind == '@mid:':
            alias = f'M₍{index}₎'
            pair = point[len('@mid:'):].split('|')
            desc = ('线段 ' + '–'.join(_name(p, aliases) for p in pair) + ' 的中点'
                    if len(pair)==2 else '构造得到的中点')
        elif kind == '@line:':
            alias = f'U₍{index}₎'
            desc = '构造直线使用的方向参考点'
        elif kind == '@circumcenter:':
            alias = f'O₍{index}₎'
            desc = '构造得到的圆心'
        else:
            alias = f'X₍{index}₎'
            desc = '构造得到的辅助点'
        aliases[point] = alias
        descriptions.append(alias + '：' + desc)
    return _PointAliases(aliases, session.points), descriptions


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
    witnesses = getattr(session,'goal_witness',{}).get(query['id'],[])
    for k, data in enumerate(goals):
        # A witness fixes the state version at the moment check_pred succeeded;
        # later inferences are not silently substituted into an earlier goal.
        observed = witnesses[k] if k < len(witnesses) else {}
        sources = observed.get('sources',()) if observed.get('predicate')==session._label(data) else ()
        if sources and data['name'] in ('coll','cyclic','overlap',*AR_KINDS):
            result, replaced = set(sources), False
        else:
            result, replaced = _goal_sources(session, data, observed.get('arEquationRevision'),
                                             observed.get('arReadTerms'))
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
    transparent = {i for i in ids if
                   ((events[i].get('type') in ('LINE_STATE','CIRCLE_STATE','POINT_MERGE')
                     and parents_for.get(i)) or
                    (events[i].get('type') == 'DD' and
                     events[i].get('rule') in ('collinear length addition',
                                               'collinear direction equivalence') and
                     parents_for.get(i)))}
    missing_state = [i for i in ids if events[i].get('type') in
                     ('LINE_STATE','CIRCLE_STATE','POINT_MERGE') and not parents_for.get(i)]
    if missing_state:
        return {'available':False,'reason':'几何状态日志缺少必要来源：'+str(missing_state[:5])}
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
    aliases, alias_notes = _display_aliases(session, statements)
    required_points = _point_tokens(session, statements) | set(aliases)
    virtual = {p['name'] for p in session.payload['compiled']['points'] if p.get('virtual')}
    virtual.update(session.payload.get('auxiliaryPoints', []))
    regular = sorted(required_points - virtual - set(aliases))
    aux_raw = required_points & virtual
    aux = sorted(aux_raw - set(aliases))
    aux.extend(sorted(aliases.values()))
    premise_inputs = [e for e in inputs if not (_point_tokens(session,e.get('conclusions',())) & aux_raw)]
    aux_inputs = [e for e in inputs if e not in premise_inputs]
    label_map = {event['id']:f'{i:02d}' for i,event in enumerate(inputs)}
    lines = ['==========================', ' * From theorem premises:',
             (' '.join(regular) + ' : Points') if regular else '(无普通点)']
    for event in premise_inputs:
        for conclusion in event.get('conclusions', []):
            lines.append(f'{_format_predicate(conclusion, aliases)} [{label_map[event["id"]]}]')
    lines.extend(['', ' * Auxiliary Constructions:',
                  (' '.join(aux) + ' : Points') if aux else '(无新增辅助点)'])
    lines.extend(alias_notes)
    for event in aux_inputs:
        for conclusion in event.get('conclusions', []):
            lines.append(f'{_format_predicate(conclusion, aliases)} [{label_map[event["id"]]}]')
    lines.extend(['', ' * Proof steps:'])
    def cited_fact(ident):
        item = events[ident]
        descriptions = '；'.join(_event_conclusion(item, con, aliases) for con in item.get('conclusions', ()))
        return descriptions + f' [{label_map[ident]}]'
    for idx, event in enumerate(derived, 1):
        new_label = f'{len(inputs)+idx-1:02d}'
        parents = ' & '.join(cited_fact(p) for p in visible_refs(parents_for.get(event['id'], ()))) or '恒等关系'
        if not event.get('conclusions'):
            return {'available':False,'reason':f'日志 #{event["id"]} 没有可展示的结论。'}
        conclusions = '；'.join(_event_conclusion(event, c, aliases) for c in event['conclusions'])
        slogan = ''
        if event['type']=='DD':
            slogan = ' (' + DD_SLOGANS.get(event.get('tag'), event.get('rule') or 'DD推导') + ')'
            if (event.get('rule') or '').startswith('B11 converse'):
                slogan = ' (切割线逆定理)'
            guard = (event.get('detail') or {}).get('exteriorSecant')
            if event.get('tag') == 'B11' and guard and len(guard) == 3:
                p, a, b = (aliases.get(str(x), str(x)) for x in guard)
                slogan += f'（位置分支：{p} 在 {a}{b} 线段外，{a}、{b} 位于 {p} 同侧）'
        lines.append(f'{idx:03d}. {parents}{slogan} ⇒ {conclusions} [{new_label}]')
        label_map[event['id']] = new_label
    ending = ' & '.join(cited_fact(r) for r in visible_refs(refs)) or '恒等关系'
    target = ' & '.join(_format_predicate(s, aliases) for s in goal_labels)
    lines.append(f'{len(derived)+1:03d}. {ending} ⇒ {target}' + ('（点重合替换）' if substitutions else ''))
    lines.append('==========================')
    return {'available': True, 'content': '\n'.join(lines), 'eventCount':len(ids),
            'proofSteps':len(derived)+1, 'goal':goal_labels}
