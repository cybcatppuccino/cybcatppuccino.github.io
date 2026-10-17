"""Optional incremental B5ab/ac/bb/bc/cc sine-ratio DD -> ARmul bridge.

Every ratio is a product of *segment lengths*.  Sine expressions are only
transient angle-AR lookup keys, never inserted into an elimination system.
The two angles in a ratio are indexed as a pair of canonical ± angle forms;
thus equal and supplementary angles are handled by the same exact lookup.
"""
import itertools
import math
from collections import defaultdict

import numericals as ng


class SineRatios:
  def __init__(self, bridge):
    self.b = bridge
    self.e = bridge.e
    self.all_points = tuple(sorted(self.e.points, key=str))
    self.rows = {'a': [], 'b': []}
    self.row_seen = set()
    self.circle_chords = {}  # circle -> chord angle atoms, one per chord
    self.dirty_lines = set()
    self.dirty_circles = set()
    self.last_angle_epoch = -1
    self.dirty = True
    self.emitted = set()
    self.numeric_angles = {}  # stable numeric geometry prefilter
    self.a_built = False
    for line in self.e.lines:
      if len(line.points)>=3:self.dirty_lines.add(line)
    for circle in self.e.circles:
      if len(circle.points)>=4:self.dirty_circles.add(circle)

  def line_added(self, line):
    if len(line.points)>=3:self.dirty_lines.add(line)

  def circle_added(self, circle):
    if len(circle.points)>=4:self.dirty_circles.add(circle)

  @property
  def needs_search(self):
    return (self.dirty or bool(self.dirty_lines) or bool(self.dirty_circles) or
            self.last_angle_epoch!=self.b.angle_epoch)

  def _angle(self, a,b,c):
    return self.e.pair_to_dir[b,a] - self.e.pair_to_dir[b,c]

  def _sine(self,a,b,c):
    key=(a,b,c)
    if key not in self.numeric_angles:
      u=a.value-b.value;v=c.value-b.value
      denom=math.sqrt(float((u@u)*(v@v)))
      self.numeric_angles[key]=(abs(float(u[0]*v[1]-u[1]*v[0]))/denom
                                if denom > ng.ATOM else 0.)
    return self.numeric_angles[key]

  def _add_row(self,typ,ang1,ang2,top,bottom,premises,points):
    if min(self._sine(*ang1),self._sine(*ang2))<1e-9:return
    eq=self.b._equation_key(top,bottom)
    key=(typ,ang1,ang2,eq)
    if key in self.row_seen:return
    self.row_seen.add(key)
    self.rows[typ].append((ang1,ang2,top,bottom,premises,points))
    self.dirty=True

  def _build_a(self):
    # Triangle ABC: sin(ABC)/sin(ACB) = AC/AB.
    # The three vertices are chosen without any unproved incidence premise.
    for a,b,c in itertools.permutations(self.all_points,3):
      if abs(float(ng.orientation(a.value,b.value,c.value))) == 0:continue
      self._add_row('a',(a,b,c),(a,c,b),((a,c),),((a,b),),(),(a,b,c))

  def _build_b_lines(self):
    dirty=tuple(self.dirty_lines)
    self.dirty_lines.clear()
    for line in dirty:
      if line not in self.e.lines:continue
      pts=tuple(sorted(line.points,key=str))
      outside=(a for a in self.all_points if a not in line.points)
      for a in outside:
        if ng.collinear(a.value,pts[0].value,pts[1].value):continue
        for b,c,d in itertools.permutations(pts,3):
          # B,C,D are *formally* collinear.
          # sin(BAC)/sin(CAD)=(BC*AD)/(AB*CD).
          self._add_row('b',(b,a,c),(c,a,d),((b,c),(a,d)),
                        ((a,b),(c,d)),(self.b._coll(b,c,d),),(a,b,c,d))

  def _build_c_circles(self):
    dirty=tuple(self.dirty_circles)
    self.dirty_circles.clear()
    for circle in dirty:
      if circle not in self.e.circles or len(circle.points)<4:continue
      pts=tuple(sorted(circle.points,key=str))
      atoms=[]
      for a,c in itertools.combinations(pts,2):
        # The inscribed angle over chord AC has identical absolute sine
        # for *every* nondegenerate on-circle vertex B. Use one witness.
        b=next((b for b in pts if b not in (a,c)
                and self._sine(a,b,c)>1e-9),None)
        if b is None:continue
        other=next((d for d in pts if d not in (a,b,c)),None)
        if other is None:continue
        atoms.append(((a,b,c),(a,c),
                     ('cyclic '+' '.join(map(str,(a,b,c,other))),)))
      self.circle_chords[circle]=atoms
      self.dirty=True

  @staticmethod
  def _coeff_key(formal):
    return tuple(sorted((str(k),v.numerator,v.denominator)
                        for k,v in formal.comb.d.items()))

  def _sine_key(self,triplet,cache):
    if triplet not in cache:
      angle=self.e.elim_angle.simplify(self._angle(*triplet))
      # The two possible signs correspond to alpha / (pi-alpha);
      # simplifying modulo pi treats both as the same absolute sine.
      cache[triplet]=min(self._coeff_key(angle),
                         self._coeff_key(-angle))
    return cache[triplet]

  def _emit(self, rule, first, second):
    # Both rows encode sin(alpha)/sin(beta) as top/bottom.
    # Equate top1/bottom1 == top2/bottom2 without any trig AR variables.
    a1,a2,top1,bot1,prem1,pts1=first
    b1,b2,top2,bot2,prem2,pts2=second
    top=tuple(top1)+tuple(bot2)
    bottom=tuple(bot1)+tuple(top2)
    eq=self.b._equation_key(top,bottom)
    if not eq:return False
    if eq in self.emitted:return False
    self.emitted.add(eq)
    if self.e.elim_dist_mul.simplify(self.b._ratio(top,bottom)).is_one():return False
    # Only record ARang dependencies used by THIS DD inference, never
    # enumerate eqangle conclusions or install sine equations in AR.
    directions=self.e.elim_angle
    deps=[]
    for x,y in ((a1,b1),(a2,b2)):
      u=self._angle(*x);v=self._angle(*y)
      use=u-v
      relation='≡'
      if not directions.simplify(use).is_zero():
        use=u+v
        relation='≡ -'
      if not directions.simplify(use).is_zero():return False
      deps.append({'tag':'ARang','conclusion':'angle('+''.join(map(str,x))+') '+relation+' angle('+''.join(map(str,y))+') mod π',
                   'sources':directions.core.explain(use.comb)})
    labels=tuple(dict.fromkeys(tuple(prem1)+tuple(prem2)))
    return self.b._prove_ratio(rule,pts1+pts2,labels,top,bottom,
                                {'arNeeds':deps,'sine':'absolute sine, ARang ± equality'})

  def search(self):
    if not self.needs_search:return False
    self._build_b_lines()
    self._build_c_circles()
    if not self.a_built and (self.rows['b'] or self.circle_chords):
      self.a_built=True
      self._build_a()
    inspected_angle_epoch=self.b.angle_epoch
    self.last_angle_epoch=inspected_angle_epoch
    self.dirty=False
    angle_cache={}
    groups=defaultdict(lambda:defaultdict(list))
    # Coarse numeric lookup before any symbolic comparison: an a-pair
    # cannot contribute without a compatible b-pair or a pair of chords
    # from the same *proved* circle. Adjacent buckets tolerate float jitter.
    scale=1000000
    coarse=lambda triplet:round(self._sine(*triplet)*scale)
    b_targets={(coarse(r[0]),coarse(r[1])) for r in self.rows['b']}
    c_targets=[]
    for atoms in self.circle_chords.values():
      c_targets.append({coarse(atom[0]) for atom in atoms})
    def nearby(value, possibilities):
      return any(value+d in possibilities for d in (-1,0,1))
    for typ in ('a','b'):
      for row in self.rows[typ]:
        if typ=='a':
          x,y=coarse(row[0]),coarse(row[1])
          if not (any((x+dx,y+dy) in b_targets
                      for dx in (-1,0,1) for dy in (-1,0,1)) or
                  any(nearby(x,ks) and nearby(y,ks) for ks in c_targets)):
            continue
        key=(self._sine_key(row[0],angle_cache),self._sine_key(row[1],angle_cache))
        groups[key][typ].append(row)

    # Equations for B5ab and B5bb use a spanning star per exact angle
    # signature, not a quadratic all-against-all comparison.
    changed=False
    for grouped in groups.values():
      aa=grouped.get('a',())
      bb=grouped.get('b',())
      if bb:
        base=bb[0]
        for other in bb[1:]:changed=self._emit('B5bb sine ratio',base,other) or changed
        if aa:
          for other in bb:changed=self._emit('B5ab sine ratio',aa[0],other) or changed

    # A circle contributes one angle per chord, not one per choice of
    # inscribed-angle vertex.  Match by *exact normalized angle key* and
    # only materialize two-chord ratios demanded by a/b signatures.
    circle_keys={}
    for circle,atoms in list(self.circle_chords.items()):
      if circle not in self.e.circles:continue
      indexed={}
      for triple,chord,premise in atoms:
        key=self._sine_key(triple,angle_cache)
        indexed.setdefault(key,(triple,chord,premise))
      if indexed:circle_keys[circle]=indexed
    for (key1,key2),grouped in groups.items():
      for chord_atoms in circle_keys.values():
        if key1 not in chord_atoms or key2 not in chord_atoms:continue
        x=chord_atoms[key1];y=chord_atoms[key2]
        row=(x[0],y[0],(x[1],),(y[1],),x[2]+y[2],x[0]+y[0])
        if grouped.get('a'):
          changed=self._emit('B5ac sine ratio',grouped['a'][0],row) or changed
        if grouped.get('b'):
          changed=self._emit('B5bc sine ratio',grouped['b'][0],row) or changed

    # B5cc: if two circles share K independent inscribed-angle sine
    # signatures, K-1 *anchor* equations imply all O(K²) pair equations.
    circles=tuple(circle_keys)
    for i in range(len(circles)):
      left=circle_keys[circles[i]]
      for j in range(i+1,len(circles)):
        right=circle_keys[circles[j]]
        overlap=sorted(left.keys() & right.keys())
        if len(overlap)<2:continue
        anchor=overlap[0]
        for key in overlap[1:]:
          x1,y1=left[anchor],left[key]
          x2,y2=right[anchor],right[key]
          row1=(x1[0],y1[0],(x1[1],),(y1[1],),x1[2]+y1[2],x1[0]+y1[0])
          row2=(x2[0],y2[0],(x2[1],),(y2[1],),x2[2]+y2[2],x2[0]+y2[0])
          changed=self._emit('B5cc sine ratio',row1,row2) or changed
    return changed
