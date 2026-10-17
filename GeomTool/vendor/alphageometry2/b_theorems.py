"""Incremental B1--B11 incidence/ratio bridge for GeomTool.

Only wakes for newly proved (non-trivial) collinear lines or a pending
collinearity proof whose supporting multiplicative AR constraints changed.
No eqangle enumeration, numeric fact injection, or periodic theorem scans.
"""

import bisect
import itertools
import math
from collections import Counter, defaultdict

import numericals as ng


class BTheorems:
  def __init__(self, engine):
    self.e = engine
    self.new_lines = set()
    self.new_circles = set()
    # Indexed by an actually proved line, not by arbitrary coordinate triples.
    self.lines_at_point = defaultdict(set)
    self.intersection_templates = {}  # keyed (A,C,B,D,P)
    self.intersections_at_point = defaultdict(set)
    self.new_intersections = set()
    self.intersection_pairs_seen = set()
    self.b8_forward_checked = set()
    self.b8_checked_at = {}
    self.b9_inverse_checked_at = {}
    self.last_angle_epoch = -1
    # B10/B11: templates are created only on a proved line/circle change.
    # A stable geometry is revisited only after the relevant elimination rank
    # changes; numerical tests are *filters*, never sources of premises.
    self.b10_templates = {}  # (A,B,D,C) -> internal/external
    self.b10_forward_at = {}
    self.b10_inverse_at = {}
    self.b11_templates = {}  # (circle, center, P,A,B,T)
    self.b11_checked_at = {}
    self.b11_inverse_at = {}  # per candidate, last ARmul revision checked
    self.b10_last_angle_epoch = -1
    self.b10_last_mul_epoch = -1
    self.b11_last_angle_epoch = -1
    self.b11_last_mul_epoch = -1
    self.circle_epoch = 0
    self.triangles = {}  # canonical side-line triples -> incidence templates
    self.inverses = {}   # (rule, ordered names) -> (query, preconditions)
    self.forward_used = set()
    self.inverse_checked_at = {}
    self.line_epoch = 0
    self._busy = False
    self.extended_sin = None  # Off by default: original B5 unchanged.
    # On initialization, only proven multi-point lines count as incidences.
    for line in engine.lines:
      if len(line.points) >= 3:
        self.new_lines.add(line)
        for pt in line.points:self.lines_at_point[pt].add(line)
    for circle in engine.circles:
      self.new_circles.add(circle)

  def set_extended_sin(self, enabled):
    if enabled:
      if self.extended_sin is None:
        from sine_ratios import SineRatios
        self.extended_sin=SineRatios(self)
    else:
      self.extended_sin=None

  def circle_added(self, circle):
    if self.extended_sin is not None:self.extended_sin.circle_added(circle)
    # B8 needs four points; B11 also recognizes a centered three-point circle.
    if len(circle.points) >= 4 or (len(circle.points) >= 3 and circle.centers):
      self.new_circles.add(circle)
      self.circle_epoch += 1

  @property
  def angle_epoch(self):
    return len(self.e.elim_angle.core.instantiated)

  @property
  def needs_search(self):
    return ((self.extended_sin is not None and self.extended_sin.needs_search) or
            bool(self.new_lines) or bool(self.new_circles) or
            (bool(self.b10_templates) and (
                self.angle_epoch != self.b10_last_angle_epoch or
                self.mul_epoch != self.b10_last_mul_epoch)) or
            (bool(self.b11_templates) and
             (self.angle_epoch != self.b11_last_angle_epoch or
              self.mul_epoch != self.b11_last_mul_epoch)) or
            (bool(self.inverses) and self.mul_epoch != getattr(self, '_last_mul_epoch', -1)) or
            (bool(self.intersection_templates) and
             (self.mul_epoch != getattr(self, '_last_b9_mul_epoch', -1) or
              self.angle_epoch != self.last_angle_epoch)))

  def line_added(self, line):
    if self.extended_sin is not None:self.extended_sin.line_added(line)
    if len(line.points) >= 3:
      self.new_lines.add(line)
      for pt in line.points:self.lines_at_point[pt].add(line)
      self.line_epoch += 1

  @property
  def mul_epoch(self):
    return len(self.e.elim_dist_mul.core.instantiated)

  def _line(self, a, b):
    return self.e.pair_to_line.get((a, b))

  def _known(self, *pts):
    if len(set(pts)) < 3:
      return False
    a, b = pts[:2]
    line = self._line(a, b)
    return line is not None and set(pts).issubset(line.points)

  def _coll(self, *pts):
    return 'coll ' + ' '.join(str(x) for x in pts)

  def _ratio(self, top, bottom):
    dist = self.e.pair_to_dist_mul
    value = dist[top[0][0],top[0][1]]
    for x,y in top[1:]:value=value*dist[x,y]
    for x,y in bottom:value=value/dist[x,y]
    return value

  def _nonzero(self, *pts):
    return len(set(pts)) == len(pts) and all(
        not self.e.num_identical(a,b) for a,b in itertools.combinations(pts,2))

  def _signed_side(self, a, x, b):
    # X=A+t(B-A); use the sign of t/(1-t) solely as a branch guard.
    delta=b.value-a.value
    denom=float(delta@delta)
    if denom < ng.ATOM:return None
    t=float((x.value-a.value)@delta)/denom
    if abs(t)<1e-10 or abs(1-t)<1e-10:return None
    return 1 if t*(1-t)>0 else -1

  def _orientation_ok(self, a,b,c,d,e,f,expected):
    sides=(self._signed_side(b,d,c),self._signed_side(c,e,a),
           self._signed_side(a,f,b))
    return None not in sides and math.prod(sides)==expected

  def _prove_ratio(self, rule, points, premises, top, bottom, detail=None):
    # DD makes the ratio available to ARmul; the DD result is logged *before*
    # the equation so every subsequent AR antecedent has a real source ID.
    key=(rule,tuple(str(x) for x in points))
    if key in self.forward_used:return False
    raw=self._ratio(top,bottom)
    self.forward_used.add(key)
    if self.e.elim_dist_mul.simplify(raw).is_one():return False
    expression=' * '.join(str(a)+str(b) for a,b in top) + ' = ' + \
        ' * '.join(str(a)+str(b) for a,b in bottom)
    ref=self.e._emit_with_source(rule,premises,[expression],detail)
    core=self.e.elim_dist_mul.core
    old=core.current_sources
    if ref:core.current_sources=(ref,)
    try:return bool(self.e.elim_dist_mul.force_one(raw))
    finally:core.current_sources=old

  @staticmethod
  def _equation_key(top, bottom):
    # Normalize inversion, reversed segment endpoints, and cancellation.
    # Six cyclic/reflected Menelaus templates should require one AR test.
    counts=Counter()
    for a,b in top:counts[tuple(sorted((str(a),str(b))))]+=1
    for a,b in bottom:counts[tuple(sorted((str(a),str(b))))]-=1
    reduced=tuple(sorted((p,n) for p,n in counts.items() if n))
    neg=tuple((p,-n) for p,n in reduced)
    return min(reduced,neg)

  def _pending(self, rule, pts, target, premises, top, bottom, guard):
    if not guard or self._known(*target):return
    key=(rule,tuple(sorted(str(x) for x in target)),
         self._equation_key(top,bottom))
    if key not in self.inverses:
      self.inverses[key]=(target,premises,top,bottom)

  def _apply_inverses(self, goal=None):
    changed=False
    # Demand mode narrows the check to templates capable of proving this goal.
    target_set=set(goal) if goal else None
    for key,(target,premises,top,bottom) in list(self.inverses.items()):
      if target_set is not None and not target_set.issubset(set(target)):
        continue
      if self._known(*target):
        del self.inverses[key]
        self.inverse_checked_at.pop(key,None)
        continue
      revision=(self.line_epoch,self.mul_epoch)
      if self.inverse_checked_at.get(key)==revision:continue
      self.inverse_checked_at[key]=revision
      raw=self._ratio(top,bottom)
      if not self.e.elim_dist_mul.simplify(raw).is_one():continue
      name=key[0]
      ar={'tag':'ARmul','conclusion':' * '.join(str(a)+str(b) for a,b in top)
          +' = '+' * '.join(str(a)+str(b) for a,b in bottom),
          'sources':self.e.elim_dist_mul.core.explain(raw.comb)}
      if not all(ng.collinear(target[0].value,target[1].value,p.value) for p in target[2:]):
        continue  # guard against invalid numeric branches / inconsistent drawing
      src=self.e._next_dd_source
      stage=self.e.proof_stage
      self.e.proof_stage='B theorem inverse'
      self.e._next_dd_source=[]
      event=self.e._emit_with_source(name,premises,['coll '+' '.join(map(str,target))],
              {'arNeeds':[ar], 'branch':'SSA-style automatic numeric branch; no extra premise'})
      try:
        self.e._next_dd_source=[event] if event else []
        updated=self.e.force_collinear(list(target))
        changed=bool(updated) or changed
      finally:
        self.e._next_dd_source=src
        self.e.proof_stage=stage
      if self._known(*target):
        del self.inverses[key]
        self.inverse_checked_at.pop(key,None)
    return changed

  def _build_triangles(self):
    # The intersection index is updated only when an incidence line changed.
    # The side lines must each contain a third, already proved, intercept.
    side_lines=sorted((x for x in self.e.lines if len(x.points)>=3),
        key=lambda line:tuple(sorted(map(str,line.points))))
    if not self.new_lines:return []
    dirty=set(self.new_lines)
    self.new_lines.clear()
    for key in list(self.triangles):
      if any(x not in self.e.lines for x in key):del self.triangles[key]
    added=[]
    for l1,l2,l3 in itertools.combinations(side_lines,3):
      if not ({l1,l2,l3} & dirty):continue
      common12=set(l1.points)&set(l2.points)
      common23=set(l2.points)&set(l3.points)
      common13=set(l1.points)&set(l3.points)
      if len(common12)!=1 or len(common23)!=1 or len(common13)!=1:continue
      a=next(iter(common12)); b=next(iter(common23)); c=next(iter(common13))
      if len({a,b,c})!=3 or ng.collinear(a.value,b.value,c.value):continue
      # l1: AB, l2: BC, l3: CA. All 6 permutations generate
      # the oriented ratios needed by ARmul; duplication is canonicalized.
      for aa,bb,cc,ab,bc,ca in ((a,b,c,l2,l3,l1),
                                  (b,c,a,l3,l1,l2),
                                  (c,a,b,l1,l2,l3),
                                  (b,a,c,l2,l1,l3),
                                  (c,b,a,l3,l2,l1),
                                  (a,c,b,l1,l3,l2)):
        key=(aa,bb,cc,ab,bc,ca)
        if key in self.triangles:continue
        self.triangles[key]=True
        added.append(key)
    return added, dirty

  def _templates(self,a,b,c,ab,bc,ca):
    # Local incidence joins (no global geometric conclusion enumeration).
    for d in sorted(bc.points,key=str):
      if d in (b,c):continue
      for e in sorted(ca.points,key=str):
        if e in (a,c):continue
        for f in sorted(ab.points,key=str):
          if f in (a,b):continue
          if not self._nonzero(a,b,c,d,e,f):continue
          # BD/DC · CE/EA · AF/FB = 1 (both Menelaus / Ceva)
          top=[(b,d),(c,e),(a,f)]
          bottom=[(d,c),(e,a),(f,b)]
          sides=[self._coll(b,d,c),self._coll(c,e,a),self._coll(a,f,b)]
          data=(a,b,c,d,e,f)
          if (self._orientation_ok(*data,-1) and
              ng.collinear(d.value,e.value,f.value)):
            if self._known(d,e,f):
              if self._prove_ratio('B1 Menelaus',data,sides+[self._coll(d,e,f)],top,bottom):yield True
            else:
              self._pending('B2 converse Menelaus',data,(d,e,f),sides,top,bottom,True)
          if self._orientation_ok(*data,1):
            # AD, BE, CF are cevians; P is a named intersection of AD and BE.
            ld=self._line(a,d);le=self._line(b,e)
            if ld and le:
              for p in sorted(set(ld.points)&set(le.points),key=str):
                if p in data or not self._nonzero(a,d,p,b,e):continue
                if not ng.collinear(c.value,f.value,p.value):continue
                concurrency=[self._coll(a,d,p),self._coll(b,e,p)]
                if self._known(c,f,p):
                  if self._prove_ratio('B3 Ceva',data+(p,),sides+concurrency+[self._coll(c,f,p)],top,bottom):yield True
                else:
                  self._pending('B4 converse Ceva',data+(p,),(c,f,p),sides+concurrency,top,bottom,True)
          # B5--B7 use D on AB, E on AC, F on BC (unlike
          # Menelaus/Ceva, whose D is on BC and F is on AB).
          D,E,F=f,e,d
          # DG/EG = (BF/FC) (AD/AB) / (AE/AC).
          # Hence DG * FC * AB * AE = EG * BF * AD * AC.
          top5_base=[(F,c),(a,b),(a,E)]
          bottom5_base=[(b,F),(a,D),(a,c)]
          ld=self._line(D,E);lf=self._line(a,F)
          possible=(set(ld.points)|set(lf.points)) - set(data)
          for g in sorted(possible,key=str):
            if not self._nonzero(D,g,E,a,F):continue
            got_de=self._known(D,E,g)
            got_af=self._known(a,F,g)
            if not got_de and not got_af:continue
            if not (ng.collinear(D.value,E.value,g.value) and
                    ng.collinear(a.value,F.value,g.value)):
              continue
            top5=[(D,g)]+top5_base
            bottom5=[(E,g)]+bottom5_base
            if got_de and got_af:
              if self.extended_sin is None and self._prove_ratio('B5 four ratios',(a,b,c,D,E,F,g),
                  sides+[self._coll(D,E,g),self._coll(a,F,g)],top5,bottom5):yield True
            elif got_af:
              self._pending('B6 converse DE',(a,b,c,D,E,F,g),(D,E,g),
                  sides+[self._coll(a,F,g)],top5,bottom5,True)
            elif got_de:
              self._pending('B7 converse AF',(a,b,c,D,E,F,g),(a,F,g),
                  sides+[self._coll(D,E,g)],top5,bottom5,True)

  # B8/B9: P is a *named, proved* intersection of AC and BD.  Angles
  # A and C are either supplementary (concyclic) or equal (the other
  # geometric branch).  A mere coordinate coincidence never supplies a
  # theorem premise; coordinates only select the SSA-style branch.
  def _angle_equations(self, a, b, c, d):
    direction=self.e.pair_to_dir
    at_a=direction[a,b]-direction[a,d]
    at_c=direction[c,b]-direction[c,d]
    return at_a-at_c, at_a+at_c  # supplementary/cyclic, equal

  @staticmethod
  def _angle_equal_numeric(a,b,c,d):
    ab=b.value-a.value;ad=d.value-a.value
    cb=b.value-c.value;cd=d.value-c.value
    norms=math.sqrt(float((ab@ab)*(ad@ad)*(cb@cb)*(cd@cd)))
    if norms < ng.ATOM:return False
    cos_a=float(ab@ad)/math.sqrt(float((ab@ab)*(ad@ad)))
    cos_c=float(cb@cd)/math.sqrt(float((cb@cb)*(cd@cd)))
    return abs(cos_a-cos_c)<1e-8

  @staticmethod
  def _cyclic_numeric(a,b,c,d):
    if ng.orientation(a.value,b.value,c.value)==0:return False
    circle=ng.NumCircle.through(a.value,b.value,c.value)
    return circle.distance(d.value)**2 < ng.ATOM

  def _cyclic_known(self,a,b,c,d):
    circle=self.e.triple_to_circle.get((a,b,c))
    return circle is not None and d in circle.points

  def _intersection_pair(self, first, second):
    if first is second or first not in self.e.lines or second not in self.e.lines:
      return
    if len(first.points)<3 or len(second.points)<3:return
    # Ordered: AC/BD and BD/AC give different independent inverse ratios.
    pair=(first,second)
    if pair in self.intersection_pairs_seen:return
    self.intersection_pairs_seen.add(pair)
    shared=set(first.points)&set(second.points)
    if not shared:return
    for p in sorted(shared,key=str):
      for a,c in itertools.combinations(sorted(set(first.points)-{p},key=str),2):
        for b,d in itertools.combinations(sorted(set(second.points)-{p},key=str),2):
          if not self._nonzero(a,b,c,d,p):continue
          if (ng.orientation(a.value,b.value,c.value)==0 or
              ng.orientation(a.value,b.value,d.value)==0):continue
          # There are two valid branches: cyclic, or opposite angles equal.
          cyclic=self._cyclic_numeric(a,b,c,d)
          equal=self._angle_equal_numeric(a,b,c,d)
          if not (cyclic or equal):continue
          key=(a,c,b,d,p)
          if key not in self.intersection_templates:
            self.intersection_templates[key]=(cyclic,equal)
            self.new_intersections.add(key)
            for pt in (a,b,c,d):self.intersections_at_point[pt].add(key)

  def _discover_intersections(self, dirty):
    # Only join a newly proved collinearity with another proved line
    # through a common named point.  No global five-point loop.
    active=self.e.lines
    for line in dirty:
      if line not in active:continue
      for p in line.points:
        for other in tuple(self.lines_at_point[p]):
          if other in active and other is not line:
            # Either diagonal can be the one denominated AC in B8/B9.
            self._intersection_pair(line,other)
            self._intersection_pair(other,line)

  @staticmethod
  def _b8_equation(a,b,c,d,p):
    # AP/PC = (AB·AD)/(CB·CD)
    return ([(a,p),(c,b),(c,d)],[(p,c),(a,b),(a,d)])

  @staticmethod
  def _angle_equal_label(a,b,c,d):
    # d(AB)-d(AD)+d(CB)-d(CD)=0 (mod pi)
    return 'angeq '+' '.join(map(str,(a,b,a,d,c,b,c,d)))+' 1,-1,1,-1,0'

  def _b8_b9(self, goal=None, changed_circles=()):
    changed=False
    goal_set=set(goal) if goal else None
    # Existing intersection templates are reconsidered only on a new
    # angle/circle fact (B8) or independent ARmul constraint (B9).
    angle_revision=(self.angle_epoch,self.circle_epoch)
    mul_revision=self.mul_epoch
    check_all=(self.angle_epoch!=self.last_angle_epoch or
               self.mul_epoch!=getattr(self, '_last_b9_mul_epoch', -1))
    if check_all:
      keys=set(self.intersection_templates)
    else:
      keys=set(self.new_intersections)
      for circle in changed_circles:
        if circle.points:
          cpts=set(circle.points)
          keys.update(key for key in self.intersections_at_point[circle.points[0]]
                      if set(key[:4]) <= cpts)
    self.new_intersections.clear()
    for key in tuple(keys):
      cyclic_numeric,equal_numeric=self.intersection_templates[key]
      a,c,b,d,p=key
      if not self._known(a,p,c) or not self._known(b,p,d):continue
      top,bottom=self._b8_equation(a,b,c,d,p)
      premises=[self._coll(a,p,c),self._coll(b,p,d)]
      # Forward: B8 only consumes an actual DD circle or a proved ARang
      # opposite-angle equality.  The numeric branch is never enough.
      if key not in self.b8_forward_checked and self.b8_checked_at.get(key)!=angle_revision:
        self.b8_checked_at[key]=angle_revision
        circle=self._cyclic_known(a,b,c,d)
        equal=False
        if not circle and equal_numeric:
          _,ang_equal=self._angle_equations(a,b,c,d)
          equal=self.e.elim_angle.simplify(ang_equal).is_zero()
        if circle or equal:
          self.b8_forward_checked.add(key)
          condition=('cyclic '+' '.join(map(str,(a,b,c,d)))) if circle else \
                    self._angle_equal_label(a,b,c,d)
          changed=self._prove_ratio('B8 intersecting diagonals',(a,b,c,d,p),
              premises+[condition],top,bottom,
              {'branch':'concyclic' if circle else 'equal opposite angles'}) or changed
      # Converse: B9 must additionally have a proven multiplicative
      # equation.  Coordinates only choose between cyclic and equal-angle
      # consequences, as SSA chooses an unambiguous orientation.
      target_cyclic=cyclic_numeric and not self._cyclic_known(a,b,c,d)
      target_equal=not cyclic_numeric and equal_numeric
      if not (target_cyclic or target_equal):continue
      if goal_set is not None and not goal_set.issubset({a,b,c,d}):continue
      revision=mul_revision
      if self.b9_inverse_checked_at.get(key)==revision:continue
      self.b9_inverse_checked_at[key]=revision
      if target_equal:
        _,eq=self._angle_equations(a,b,c,d)
        if self.e.elim_angle.simplify(eq).is_zero():continue
      raw=self._ratio(top,bottom)
      if not self.e.elim_dist_mul.simplify(raw).is_one():continue
      ar={'tag':'ARmul','conclusion':' * '.join(str(x)+str(y) for x,y in top)
          +' = '+' * '.join(str(x)+str(y) for x,y in bottom),
          'sources':self.e.elim_dist_mul.core.explain(raw.comb)}
      prior_source=self.e._next_dd_source
      prior_stage=self.e.proof_stage
      self.e.proof_stage='B9 opposite-angle inverse'
      conclusion=('cyclic '+' '.join(map(str,(a,b,c,d)))) if target_cyclic else \
                 self._angle_equal_label(a,b,c,d)
      event=self.e._emit_with_source('B9 converse intersecting diagonals',
          premises,[conclusion],
          {'arNeeds':[ar], 'branch':'numeric cyclic' if target_cyclic else 'numeric equal angles'})
      cores,previous=self.e._set_dd_equation_source(event)
      self.e._next_dd_source=[event] if event else []
      try:
        if target_cyclic:
          changed=bool(self.e.force_concyclic([a,b,c,d],())) or changed
        else:
          _,eq=self._angle_equations(a,b,c,d)
          changed=bool(self.e.elim_angle.force_zero(eq)) or changed
      finally:
        self.e._restore_dd_equation_source(cores,previous)
        self.e._next_dd_source=prior_source
        self.e.proof_stage=prior_stage
    # Preserve the revisions actually inspected.  B8 may add ARmul, and
    # B9 may add ARang *while* the candidates are being visited.  Both
    # require one more event-driven pass for templates visited earlier.
    self.last_angle_epoch=angle_revision[0]
    self._last_b9_mul_epoch=mul_revision
    return changed

  # B10: internal and external angle-bisector theorem AND converse.
  # Templates are indexed only when a genuine three-point incidence arrives.
  # The ratio is |BD|*|AC|=|DC|*|AB| for either branch; its converse
  # produces the angle equality in ARang, not a tautological ARmul equation.
  @staticmethod
  def _near(x, y):
    return abs(x-y) <= 1e-7 * max(abs(x),abs(y),1.0)

  def _build_b10(self, dirty):
    # In a line with m named points, do NOT enumerate all m^3 triples:
    # for each B,C,A solve the two possible positions of the angle-bisector
    # intersection D numerically, then look for an existing named point by
    # binary search.  O(outside_points * m^2 * log(m)) per new line.
    for line in dirty:
      if line not in self.e.lines or len(line.points)<3:continue
      members=sorted(set(line.points),key=str)
      origin=members[0].value
      vec=members[1].value-origin
      length2=float(vec@vec)
      if length2<ng.ATOM:continue
      positions=sorted((float((p.value-origin)@vec)/length2,str(p),p)
                       for p in members)
      scalars=[pos for pos,_,_ in positions]
      outsiders=[p for p in self.e.points if p not in line.points and
                 not ng.collinear(p.value,members[0].value,members[1].value)]
      for b,c in itertools.combinations(members,2):
        if self.e.num_identical(b,c):continue
        ab_cache={a:ng.distance(a.value,b.value) for a in outsiders}
        ac_cache={a:ng.distance(a.value,c.value) for a in outsiders}
        pos_b=float((b.value-origin)@vec)/length2
        pos_c=float((c.value-origin)@vec)/length2
        for a in outsiders:
          ab=ab_cache[a];ac=ac_cache[a]
          if min(ab,ac)<ng.ATOM:continue
          targets=[((ac*pos_b+ab*pos_c)/(ab+ac),'internal')]
          if abs(ab-ac)>1e-10*max(ab,ac):
            targets.append(((ab*pos_c-ac*pos_b)/(ab-ac),'external'))
          for target,branch in targets:
            ix=bisect.bisect_left(scalars,target)
            for index in (ix-1,ix):
              if not 0<=index<len(positions):continue
              d=positions[index][2]
              if d in (b,c) or not self._nonzero(a,b,c,d):continue
              if self._signed_side(b,d,c)!=(1 if branch=='internal' else -1):continue
              if not self._near(ng.distance(d.value,b.value)*ac,
                                ng.distance(d.value,c.value)*ab):continue
              self.b10_templates[(a,b,d,c)]=branch

  def _b10_angle(self,a,b,d,c):
    lines=self.e.pair_to_dir
    return lines[a,b]-lines[a,d]-(lines[a,d]-lines[a,c])

  @staticmethod
  def _b10_angle_label(a,b,d,c):
    return 'eqangle '+' '.join(map(str,(a,b,a,d,a,d,a,c)))

  def _b10(self):
    changed=False
    arang_epoch=self.angle_epoch
    armul_epoch=self.mul_epoch
    for key,branch in tuple(self.b10_templates.items()):
      a,b,d,c=key
      if not self._known(b,d,c):continue
      top=[(b,d),(a,c)]
      bottom=[(d,c),(a,b)]
      equation=self._b10_angle_label(a,b,d,c)
      incidence=self._coll(b,d,c)
      # Known bisector -> ARmul: no parallel/global equation enumeration.
      if self.b10_forward_at.get(key)!=arang_epoch:
        self.b10_forward_at[key]=arang_epoch
        if self.e.elim_angle.simplify(self._b10_angle(a,b,d,c)).is_zero():
          changed=self._prove_ratio('B10 angle bisector '+branch,key,
              [incidence,equation],top,bottom,{'branch':branch}) or changed
      # Genuine converse: proven multiplicative equality -> ARang.
      if self.b10_inverse_at.get(key)!=armul_epoch:
        self.b10_inverse_at[key]=armul_epoch
        ratio=self._ratio(top,bottom)
        if not self.e.elim_dist_mul.simplify(ratio).is_one():continue
        angle=self._b10_angle(a,b,d,c)
        if self.e.elim_angle.simplify(angle).is_zero():continue
        ar={'tag':'ARmul',
            'conclusion':' * '.join(str(x)+str(y) for x,y in top)+' = '+
                         ' * '.join(str(x)+str(y) for x,y in bottom),
            'sources':self.e.elim_dist_mul.core.explain(ratio.comb)}
        event=self.e._emit_with_source('B10 converse angle bisector '+branch,
            [incidence],[equation],{'arNeeds':[ar],'branch':branch})
        core=self.e.elim_angle.core
        previous=core.current_sources
        if event:core.current_sources=(event,)
        try:changed=bool(self.e.elim_angle.force_zero(angle)) or changed
        finally:core.current_sources=previous
    self.b10_last_angle_epoch=arang_epoch
    self.b10_last_mul_epoch=armul_epoch
    return changed

  # B11: tangent-secant (power of a point).  The presence of a circle
  # with a *proved center* is an index only; the actual log premises are
  # two proved equal radii, proved collinearity and OT perpendicular PT.
  def _build_b11(self, dirty_lines, dirty_circles):
    active=[x for x in self.e.circles if x.centers and len(x.points)>=3]
    if not active:return
    existing_lines=[x for x in self.e.lines if len(x.points)>=3]
    new_circles=set(dirty_circles)
    for circle in active:
      if circle not in new_circles and not dirty_lines:continue
      points=set(circle.points)
      centers=sorted(circle.centers,key=str)
      for line in existing_lines:
        if circle not in new_circles and line not in dirty_lines:continue
        on_circle=sorted(points.intersection(line.points),key=str)
        if len(on_circle)<2:continue
        outsiders=sorted(set(line.points)-points,key=str)
        for a,b in itertools.combinations(on_circle,2):
          for p in outsiders:
            if not self._nonzero(p,a,b):continue
            # Lengths here are unsigned. The converse PT²=PA·PB is false
            # for a point *between* A and B. Keep only the exterior secant
            # branch (A and B lie on the same ray from P). This is an
            # explicit numeric branch guard, never an AR/DD assumption.
            if self._signed_side(a,p,b)!=-1:continue
            for t in points-{a,b}:
              if t is p or not self._nonzero(p,t,a,b):continue
              if ng.orientation(a.value,b.value,t.value)==0:continue
              pa=ng.distance(p.value,a.value)
              pb=ng.distance(p.value,b.value)
              pt=ng.distance(p.value,t.value)
              if not self._near(pt*pt,pa*pb):continue
              # Numeric tangent test filters candidates, but never proves
              # perpendicularity (which is independently checked below).
              for o in centers:
                if not self._nonzero(o,t,p):continue
                self.b11_templates[(circle,o,p,a,b,t)]=True

  def _b11(self):
    changed=False
    angle_epoch=self.angle_epoch
    mul_epoch=self.mul_epoch
    active=self.e.circles
    for key in tuple(self.b11_templates):
      circle,o,p,a,b,t=key
      if (circle not in active or o not in circle.centers or
          not {a,b,t}.issubset(set(circle.points)) or
          not self._known(p,a,b)):
        del self.b11_templates[key]
        self.b11_checked_at.pop(key,None)
        self.b11_inverse_at.pop(key,None)
        continue
      # A formerly valid candidate may be in a now-merged point class.
      if self._signed_side(a,p,b)!=-1:continue
      angle=self.e.pair_to_dir[o,t]-self.e.pair_to_dir[t,p]-self.e.elim_angle.const(1,2)
      m=self.e.pair_to_dist_mul
      radii=(('cong '+' '.join(map(str,(o,a,o,b))),m[o,a]/m[o,b]),
             ('cong '+' '.join(map(str,(o,a,o,t))),m[o,a]/m[o,t]))
      if not all(self.e.elim_dist_mul.simplify(expr).is_one() for _,expr in radii):
        continue
      premises=[self._coll(p,a,b)]+[label for label,_ in radii]
      # Existing forward B11, unchanged in meaning: an established tangent
      # introduces its power-of-a-point equation into ARmul.
      if self.b11_checked_at.get(key)!=angle_epoch:
        self.b11_checked_at[key]=angle_epoch
        if self.e.elim_angle.simplify(angle).is_zero():
          changed=self._prove_ratio('B11 tangent-secant',(o,p,a,b,t),
              premises+['perp '+' '.join(map(str,(o,t,t,p)))],
              [(p,t),(p,t)],[(p,a),(p,b)]) or changed
      # New inverse B11: only a *proved* historical ARmul relation can
      # supply this antecedent. No coordinate equality is ever submitted
      # to either elimination system.
      if self.b11_inverse_at.get(key)==mul_epoch:continue
      self.b11_inverse_at[key]=mul_epoch
      if self.e.elim_angle.simplify(angle).is_zero():continue
      power=self._ratio([(p,t),(p,t)],[(p,a),(p,b)])
      if not self.e.elim_dist_mul.simplify(power).is_one():continue
      ar={'tag':'ARmul',
          'conclusion':str(p)+str(t)+' * '+str(p)+str(t)+' = '+
                       str(p)+str(a)+' * '+str(p)+str(b),
          'sources':self.e.elim_dist_mul.core.explain(power.comb)}
      # The exterior branch is essential for *unsigned* distances. Expose
      # this side condition in the proof text, rather than hiding it.
      detail={'arNeeds':[ar], 'branch':'exterior secant',
              'exteriorSecant':[str(p),str(a),str(b)]}
      conclusion='perp '+' '.join(map(str,(o,t,t,p)))
      event=self.e._emit_with_source('B11 converse tangent-secant',premises,
                                     [conclusion],detail)
      core=self.e.elim_angle.core
      old=core.current_sources
      if event:core.current_sources=(event,)
      try:changed=bool(self.e.elim_angle.force_zero(angle)) or changed
      finally:core.current_sources=old
    self.b11_last_angle_epoch=angle_epoch
    self.b11_last_mul_epoch=mul_epoch
    return changed

  def search(self, goal=None):
    """Event-driven: a newly proved line, or demand for a pending line."""
    if self._busy:return False
    had_new=bool(self.new_lines)
    had_circles=bool(self.new_circles)
    if not self.needs_search:return False
    self._busy=True
    changed=False
    try:
      dirty=set(self.new_lines)
      if had_new:
        added,dirty=self._build_triangles()
        # Incremental join: besides triangle sides that were just created,
        # revisit only existing triangles connected to >=2 points of a new
        # incidence line. Every B transversal/cevian contains at least two
        # such vertices/intercepts, so unrelated triangles are never scanned.
        fresh=set(added)
        dirty_points=[set(line.points) for line in dirty]
        for key in list(self.triangles):
          if key not in fresh:
            local=set(key[3].points)|set(key[4].points)|set(key[5].points)
            if not any(len(local & pts)>=2 for pts in dirty_points):continue
          for updated in self._templates(*key):changed=bool(updated) or changed
        self._discover_intersections(dirty)
        self._build_b10(dirty)
      # Process only circles/lines with new provenance; existing candidate
      # checks are triggered by ARang/ARmul rank changes, never per round.
      self._build_b11(dirty if had_new else set(),tuple(self.new_circles))
      changed=self._b10() or changed
      changed=self._b11() or changed
      checked_mul_epoch=self.mul_epoch
      changed=self._apply_inverses(goal) or changed
      # Consume only the circles known at entry: a newly derived B9
      # circle stays queued for any other B8 chords in the next event.
      changed_circles=tuple(self.new_circles)
      self.new_circles.difference_update(changed_circles)
      changed=self._b8_b9(goal,changed_circles) or changed
      # If a B8 equation appeared after B2/B4/B6/B7 were tested, their
      # older multiplicative revision remains pending for the next pass.
      self._last_mul_epoch=checked_mul_epoch
      if self.extended_sin is not None:
        changed=self.extended_sin.search() or changed
      return changed
    finally:self._busy=False
