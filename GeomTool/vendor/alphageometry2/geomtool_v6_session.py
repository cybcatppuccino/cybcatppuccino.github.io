"""GeomTool v6.4: one AG2 DDAR closure; DD events + demand-only AR antecedents.

No conjecture enumeration, numerical inference, replay certificates.
A positive goal is decided exclusively by DDAR.check_pred.  Its geometric
plausibility is checked by the caller before it enters the wish list.
"""
import json
import time
import os
from fractions import Fraction
import numpy as np
from parse import AGPoint, AGPredicate
from ddar import DDAR

STAGES = [('similar triangles','search_similar'), ('cyclic quadrilaterals','search_concyclic'),
          ('circles','search_circles'), ('merge equal points','merge_points'),
          ('distance add/mul transfer','transfer_dist_add_mul'),
          ('arc/length transfer','transfer_dist_arc_mul')]

class Session:
    def __init__(self, payload):
        self.payload=payload
        self.started=time.perf_counter()
        compiled=payload['compiled']
        self.points={p['name']:AGPoint(p['name'],np.array(p['value'],dtype=float)) for p in compiled['points']}
        self.audit_path='/home/py/ag2/geomtool_ag2_v6.4_log.jsonl' if os.path.isdir('/home/py/ag2') else os.path.join(os.path.dirname(__file__),'geomtool_ag2_v6.4_log.jsonl')
        self.audit_file=open(self.audit_path,'w',encoding='utf-8')
        self.audit_count=0
        self.audit_index={}
        self.sources={}
        self.eq_nodes={}
        self.eq_serial=0
        self.input_by_label={}
        self.ar_cache={}
        self.engine=DDAR(list(self.points.values()))
        self.engine.b_theorems.set_extended_sin(bool(payload.get('extendedSinRatio',False)))
        self.engine.proof_logger=self._dd_event
        self.engine.proof_stage='assumptions'
        for key,core in [('ARang',self.engine.elim_angle.core),('ARmul',self.engine.elim_dist_mul.core),('ARadd',self.engine.elim_dist_add.core)]:
            core.on_add_constraint=lambda expr,refs,k=key:self._equation(k,expr,refs)
        for item in compiled['predicates']:
            label=self._label(item)
            ref=self._write({'type':'PREMISE','tag':'INPUT','conclusions':[label],
                             'premises':[],'source':item.get('source','construction')})
            self.input_by_label.setdefault(label,ref)
            if item['name'] in ('coll','cyclic','overlap'):
                self.sources.setdefault(item['name'],[]).append((ref,set(item['points'])))
            for core in self._cores():core.current_sources=(ref,)
            self.engine._next_dd_source=[ref]
            self.engine.force_pred(self._pred(item))
            self.engine._next_dd_source=[]
        for core in self._cores():core.current_sources=()
        self.engine.proof_stage='initialization'
        self.queries=payload.get('queries',[])
        if len(self.queries)>1:
            raise ValueError('v6 only accepts a single explicit goal or zero goals for closure')
        self.target_id=payload.get('targetId')
        self.goal_mode=payload.get('mode')=='goal'
        self.found={}
        self.goal_found=False
        self.depth=0
        self._mul_revision=0
        self._goal_bridge_revision=None
        self.stage='initialization'
        self.closed=False
        self.done=False
        self.generator=self._phases()
        self._check_goal()

    def _cores(self):
        return (self.engine.elim_angle.core,self.engine.elim_dist_mul.core,self.engine.elim_dist_add.core)

    def _label(self,item):
        return item['name']+' '+' '.join(item['points'])+ (' '+','.join(map(str,item.get('constants') or [])) if item.get('constants') else '')

    def _pred(self,item):
        constants=[]
        for x in item.get('constants',[]):
            f=Fraction(str(x));approx=f.limit_denominator(3600)
            constants.append(approx if abs(float(approx)-float(f))<1e-10 else f)
        return AGPredicate(item['name'],[self.points[x] for x in item['points']],constants)

    def _write(self,record):
        self.audit_count+=1
        record={'id':self.audit_count,**record}
        self.audit_index[record['id']]=record
        self.audit_file.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
        if self.audit_count%250==0:self.audit_file.flush()
        return record['id']

    def _equation(self,kind,expression,parents):
        # Private substitution DAG; NOT an AR log entry. The actual AR log is
        # created only if DD consumes a relation requiring this substitution.
        self.eq_serial+=1
        key=-self.eq_serial
        direct=tuple(next(core.current_sources for name,core in
                          [('ARang',self.engine.elim_angle.core),
                           ('ARmul',self.engine.elim_dist_mul.core),
                           ('ARadd',self.engine.elim_dist_add.core)] if name==kind))
        self.eq_nodes[key]={'type':kind,'expression':expression,'parents':tuple(parents),
                            'direct':direct, 'audit_at':self.audit_count}
        return key

    def _roots(self,ident,visited=None):
        visited=visited if visited is not None else set()
        if ident in visited:return set()
        visited.add(ident)
        if ident>0:return {ident} if ident in self.audit_index else set()
        node=self.eq_nodes.get(ident)
        return set().union(*(self._roots(k,visited) for k in node['parents'])) if node else set()

    def _as_pred(self,statement):
        # Statements come from the actual AG2 rule matcher, never from a
        # coordinate-derived conjecture generator.
        tokens=statement.split()
        if not tokens:return None
        typ=tokens[0]
        if typ not in {'para','perp','cong','eqratio','eqangle','angeq','rconst','aconst','coll','cyclic','overlap'}:return None
        arity={'para':4,'perp':4,'cong':4,'eqratio':8,'eqangle':8,
               'angeq':8,'rconst':4,'aconst':4,'overlap':2}.get(typ)
        if arity is None:arity=len(tokens)-1
        raw=tokens[1:]
        if len(raw)<arity:return None
        points=raw[:arity]
        for x in points:
            if x not in self.points:return None
        constants=[]
        if typ=='angeq':constants=[1,-1,-1,1,0] if len(raw)==arity else [Fraction(x) for x in raw[arity].split(',')]
        elif typ in {'rconst','aconst'}:
            if len(raw)<=arity:return None
            constants=[Fraction(raw[arity])]
        return {'name':typ,'points':points,'constants':constants}

    def _structural_ref(self,statement):
        data=self._as_pred(statement)
        if data is None:return None
        label=self._label(data)
        if label in self.input_by_label:return self.input_by_label[label]
        typ=data['name']
        if typ not in {'coll','cyclic','overlap'}:return None
        required=set(data['points'])
        for ref,points in reversed(list(self.sources.get(typ,[]))):
            if required<=points:return ref
        return None

    def _demand_ar(self,statement,data):
        typ=data['name']
        tag='ARang' if typ in {'para','perp','eqangle','angeq','aconst'} else 'ARmul' if typ in {'cong','eqratio','rconst'} else None
        if tag is None:return []
        try:
            pred=self._pred(data)
            if not self.engine.check_pred(pred):return []
            # Explain against the ORIGINAL line/distance variables, not the
            # already-simplified get_dist_ratio cache (which can be zero and
            # would otherwise lose its pivot dependency IDs).
            if tag=='ARmul':
                pts=pred.points
                mul=self.engine.pair_to_dist_mul
                if typ=='cong':
                    raw=mul[pts[0],pts[1]]/mul[pts[2],pts[3]]
                elif typ=='eqratio':
                    raw=mul[pts[0],pts[1]]/mul[pts[2],pts[3]]/(
                        mul[pts[4],pts[5]]/mul[pts[6],pts[7]])
                elif typ=='rconst':
                    raw=mul[pts[0],pts[1]]/mul[pts[2],pts[3]]/pred.constants[0]
                else:raw=self.engine.pred_to_dist_mul(pred)
                parents=self.engine.elim_dist_mul.core.explain(raw.comb)
            elif tag=='ARang':
                raw=self.engine.pred_to_angle(pred)
                parents=self.engine.elim_angle.core.explain(raw.comb)
            else:
                parents=self.engine.explain_pred(pred)
        except (AssertionError,ValueError,KeyError,IndexError,ZeroDivisionError):return []
        supports=sorted(set().union(*(self._roots(x) for x in parents)))
        if not supports and not raw.comb.d:
            return []  # reflexive equality is algebraic identity, not a DD antecedent
        known=self.input_by_label.get(self._label(data))
        if known:return [known]
        cache=(tag,statement,tuple(supports))
        if cache in self.ar_cache:return [self.ar_cache[cache]]
        ref=self._write({'type':'AR','tag':tag,'conclusions':[statement],
                         'premises':supports,'rule':'exact symbolic elimination (used by DD)'})
        self.ar_cache[cache]=ref
        return [ref]

    def _dd_event(self,event):
        typ=event.get('type','DD')
        if typ not in ('DD','DD_APPLY'):return None
        rule=event.get('rule','')
        stage=event.get('stage','')
        tag=self._tag(rule,stage,event)
        refs=[]
        statements=list(event.get('premiseStatements',[]))
        # For ordinary _emit_proof the DD module supplies the real matched
        # input conditions as textual predicates, not global input conjunctions.
        for x in event.get('premises',[]):
            if isinstance(x,int):
                if x>0 and x in self.audit_index:refs.append(x)
                elif x<0:refs.extend(self._roots(x))
            elif isinstance(x,str):statements.append(x)
        for ar in (event.get('detail') or {}).get('arNeeds',[]):
            ar_tag=ar['tag']
            sources=sorted(set().union(*(self._roots(i) for i in ar.get('sources',[]))))
            conclusion=ar['conclusion']
            cache=(ar_tag,conclusion,tuple(sources))
            if cache not in self.ar_cache:
                self.ar_cache[cache]=self._write({'type':'AR','tag':ar_tag,
                    'conclusions':[conclusion],'premises':sources,
                    'rule':'exact AR antecedent consumed by native DD'})
            refs.append(self.ar_cache[cache])
        for statement in statements:
            ref=self._structural_ref(statement)
            if ref:refs.append(ref);continue
            data=self._as_pred(statement)
            if data:refs.extend(self._demand_ar(statement,data))
        refs=sorted(set(refs))
        conclusions=event.get('conclusions',[])
        # Similarity DD_APPLY comes before its AR equations; all subsequent
        # derived DD facts cite this ID rather than repeating every hypothesis.
        ident=self._write({'type':'DD','tag':tag,'rule':rule,
                           'stage':stage,'conclusions':conclusions,
                           'premises':refs, 'detail':{k:v for k,v in (event.get('detail') or {}).items() if k!='arNeeds'}})
        for c in conclusions:
            tokens=c.split()
            if tokens and tokens[0] in {'coll','cyclic','overlap'}:
                self.sources.setdefault(tokens[0],[]).append((ident,set(tokens[1:])))
        return ident

    @staticmethod
    def _tag(rule,stage,event):
        r=rule.lower()
        if 'similarity' in r or 'similar triangles' in r:
            if r.startswith('aa'):return 'T1'
            if r.startswith('sss'):return 'T2'
            if r.startswith('sas'):return 'T3'
            if r.startswith('ssa'):return 'T4'
        if r.startswith('angle consequence'):return 'T5'
        if r.startswith('ratio consequence'):return 'T6'
        if r.startswith('zero-angle'):return 'C1'
        if r.startswith('inscribed-angle'):return 'C2'
        if r.startswith('circle angle'):return 'C3'
        if r.startswith('circle-center'):return 'C4'
        if r.startswith('circle merge'):return 'C5'
        if r.startswith('radius consequence'):return 'C6'
        if r.startswith('collinear line merge'):return 'C7'
        if r.startswith('collinear length'):return 'C8'
        if r.startswith('collinear direction'):return 'C9'
        if r.startswith('circle inheritance'):return 'C10'
        if r.startswith('equal-radius circle'):return 'R1'
        if r.startswith('center assigned'):return 'R2'
        if r.startswith('small circle'):return 'R3'
        if r.startswith('unique center'):return 'P1'
        if r.startswith('unique intersection'):return 'P2'
        if r.startswith('overlap assumption'):return 'P3'
        if r.startswith('line inheritance'):return 'P4'
        if r.startswith('circle point inheritance'):return 'P5'
        if r.startswith('equal-point distance'):return 'P6'
        if r.startswith('point substitute'):return 'P7'
        if r.startswith('mul-to-add'):return 'D1'
        if r.startswith('add-to-mul'):return 'D2'
        if r.startswith('b1 '):return 'B1'
        if r.startswith('b2 '):return 'B2'
        if r.startswith('b3 '):return 'B3'
        if r.startswith('b4 '):return 'B4'
        if r.startswith('b5ab '):return 'B5ab'
        if r.startswith('b5ac '):return 'B5ac'
        if r.startswith('b5bb '):return 'B5bb'
        if r.startswith('b5bc '):return 'B5bc'
        if r.startswith('b5cc '):return 'B5cc'
        if r.startswith('b5 '):return 'B5'
        if r.startswith('b6 '):return 'B6'
        if r.startswith('b7 '):return 'B7'
        if r.startswith('b8 '):return 'B8'
        if r.startswith('b9 '):return 'B9'
        if r.startswith('arc-to-chord'):return 'A1'
        if r.startswith('chord-to-arc'):return 'A2'
        # Unrecognized DD facts are never silently omitted or mislabelled as AR.
        return 'C10' if 'circle' in stage else 'C7' if 'line' in stage else 'P7'

    def _check_goal(self):
        if not self.queries:return
        q=self.queries[0]
        if self.found.get(q['id'],{}).get('status')=='proved':return
        try:
            predicates=q.get('predicates',[])
            if not predicates:raise ValueError('无法转换为原生 AG2 命题')
            proved=all(self.engine.check_pred(self._pred(row)) for row in predicates)
            if not proved and any(row['name'] in ('coll','cyclic') for row in predicates):
                bridge=self.engine.b_theorems
                target=next((row for row in predicates if row['name'] in ('coll','cyclic') and
                             not self.engine.check_pred(self._pred(row))),None)
                if target is not None:
                    revision=(tuple(target['points']),bridge.line_epoch,bridge.mul_epoch,bridge.circle_epoch,bridge.angle_epoch)
                    if self._goal_bridge_revision!=revision:
                        self._goal_bridge_revision=revision
                        bridge.search(goal=tuple(self.points[n] for n in target['points']))
                    proved=all(self.engine.check_pred(self._pred(row)) for row in predicates)
            self.found[q['id']]={'id':q['id'],'status':'proved' if proved else 'unknown',
                                 'reason':'AG2 形式闭包可推导' if proved else '当前 AG2 形式系统尚未证明'}
            if proved:self.goal_found=True
        except (ValueError,AssertionError,KeyError,IndexError,ZeroDivisionError) as exc:
            self.found[q['id']]={'id':q['id'],'status':'unsupported','reason':str(exc)}

    def _phases(self):
        changed=True
        while changed:
            self.depth+=1
            self.engine.update_cache()
            changed=False
            for stage,method in STAGES:
                self.stage=stage
                self.engine.proof_stage=stage
                updated=getattr(self.engine,method)(verbose=False) if method=='search_similar' else getattr(self.engine,method)()
                changed=bool(updated) or changed
                if updated and method=='search_similar':self.engine.update_cache()
                bridge=self.engine.b_theorems
                if bridge.needs_search:
                    prior_stage=self.engine.proof_stage
                    self.engine.proof_stage='B1–B9 / optional B5 sin bridge'
                    try: changed=bridge.search() or changed
                    finally:self.engine.proof_stage=prior_stage
                    self._mul_revision=bridge.mul_epoch
                self._check_goal()
                yield
        self.closed=True

    def step(self):
        if not self.closed and not (self.goal_found and self.goal_mode):
            try:next(self.generator)
            except StopIteration:self.closed=True
        self.done=self.closed or (self.goal_found and self.goal_mode)
        return self.progress()

    def progress(self):
        return {'done':self.done,'engine':'ag2','phase':self.stage,'depth':self.depth,
                'queries':list(self.found.values()),'goalFound':self.goal_found,
                'elapsedMs':round((time.perf_counter()-self.started)*1000),
                'pointCount':len(self.points),'premiseCount':len(self.input_by_label),
                'auditCount':self.audit_count,'auditTruncated':False}

    def log_text(self):
        self.audit_file.flush()
        with open(self.audit_path,encoding='utf-8') as handle:return handle.read()

_SESSION=None

def start_json(src):
    global _SESSION
    _SESSION=Session(json.loads(src))
    return json.dumps(_SESSION.progress(),ensure_ascii=False)

def step_json():return json.dumps(_SESSION.step(),ensure_ascii=False)

def audit_json():
    if _SESSION is None:return json.dumps({'available':False,'reason':'还没有 AG2 会话'},ensure_ascii=False)
    return json.dumps({'available':True,'filename':'geomtool_ag2_v6.4_log.jsonl',
                       'content':_SESSION.log_text(),'events':_SESSION.audit_count,
                       'truncated':False},ensure_ascii=False)


def proof_json():
    # A separate read-only extractor; no changes to Session, DDAR or its log.
    from geomtool_v64_proof import render
    if _SESSION is None:
        return json.dumps({'available':False,'reason':'还没有 AG2 会话'},ensure_ascii=False)
    try:
        return json.dumps(render(_SESSION),ensure_ascii=False)
    except Exception as exc:
        return json.dumps({'available':False,'reason':'证明依赖还原失败：'+str(exc)},ensure_ascii=False)
