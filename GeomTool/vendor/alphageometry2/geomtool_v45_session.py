"""v4.5 cooperative execution wrapper around the unmodified AG2 operations.
No assumption is created from numerical geometry. Each yielded stage invokes
one of the original DDAR methods, using the original closure stage order.
"""
import json
import itertools
import time
import os
from collections import deque
from fractions import Fraction
import numpy as np
from parse import AGPoint, AGPredicate
from ddar import DDAR

STAGES = [('similar triangles','search_similar'), ('cyclic quadrilaterals','search_concyclic'),
          ('circles','search_circles'), ('merge equal points','merge_points'),
          ('AR distance conversion','transfer_dist_add_mul'), ('AR arcs','transfer_dist_arc_mul')]

class Session:
    def __init__(self, payload):
        self.payload=payload
        self.started=time.perf_counter()
        compiled=payload['compiled']
        self.points={p['name']:AGPoint(p['name'],np.array(p['value'],dtype=float)) for p in compiled['points']}
        self.audit_path='/home/py/ag2/geomtool_ag2_proof_log.jsonl' if os.path.isdir('/home/py/ag2') else os.path.join(os.path.dirname(__file__),'geomtool_ag2_proof_log.jsonl')
        self.audit_file=open(self.audit_path,'w',encoding='utf-8')
        self.audit_events=deque(maxlen=10000)
        self.audit_index={}
        self.active_rule_sources=[]
        self.input_ids=[]
        self.audit_count=0
        self.audit_limit=None
        self.audit_truncated=False
        self._audit({'type':'INPUT','rule':'GeomTool provided coordinates and construction-defined axioms',
             'points':compiled['points'],'premises':compiled['predicates'],
             'virtualDescriptions':compiled.get('virtualDescriptions',[]),
             'limitations':compiled.get('limitations',[]),'goal':payload.get('targetId'),
             'goalPredicates':next((q.get('predicates',[]) for q in payload.get('queries',[])
                                    if q.get('id')==payload.get('targetId')),[]),
             'queryCount':len(payload.get('queries',[]))})
        self.engine=DDAR(list(self.points.values()))
        self.engine.proof_logger=self._audit
        for label,domain in [('angle',self.engine.elim_angle),('length-mul',self.engine.elim_dist_mul),('length-add',self.engine.elim_dist_add)]:
            domain.core.on_add_constraint=lambda eq, refs, core=domain.core, name=label: self._audit({
                'type':'AR' if self.active_rule_sources else 'EQUATION',
                'stage':self.engine.proof_stage,
                'rule':'exact Gaussian equation insertion ('+name+')',
                'conclusions':[name+': '+eq+' = 0'],
                'premises':self._support(refs),
                'complete':bool(refs) and not (bool(self.active_rule_sources) and tuple(core.current_sources)==tuple(self.input_ids)),
                'dependencyLevel':('macro: all input premises + pivot dependencies' if bool(self.active_rule_sources) and tuple(core.current_sources)==tuple(self.input_ids)
                    else 'exact logged input/DD equation + elimination pivots') if refs else 'missing source'})
        self.premises={}
        self.premise_rows=list(compiled['predicates'])
        self.proof_cache={}
        self.first_proof_stage={}
        self.input_by_key={}
        for item in compiled['predicates']:
            label=item['name']+' '+' '.join(item['points'])
            if item.get('constants'):label+=' '+','.join(map(str,item['constants']))
            input_id=self._audit({'type':'PREMISE','rule':'GeomTool construction',
                'conclusions':[label],'premises':[], 'root':True,
                'source':item.get('source','construction')})
            self.input_ids.append(input_id)
            self.input_by_key[self._key(item)]=input_id
            self.engine.proof_stage='assumptions'
            for domain in (self.engine.elim_angle,self.engine.elim_dist_mul,self.engine.elim_dist_add):
                domain.core.current_sources=(input_id,)
            pred=self._pred(item)
            self.engine.force_pred(pred)
            self.premises[self._key(item)]=item['source']
        for domain in (self.engine.elim_angle,self.engine.elim_dist_mul,self.engine.elim_dist_add):
            domain.core.current_sources=()
        self.engine.proof_stage='initialization'
        self.queries=payload.get('queries',[])
        self.found={}
        self.cursor=0
        self.attempts=0
        self.depth=0
        self.stepname='initialization'
        self.done=False
        self.closed=False
        self.final_remaining=0
        self.goal_id=payload.get('targetId')
        self.goal_found=False
        self.generator=self._phases()
        self.native={}
        self._collect_native()
        # The target is always attempted first; all results are checked by the
        # native check_pred, not by a numerical consistency test.
        self._scan(limit=10, target_first=True)

    def _audit(self, item):
        """Append every inference event to JSONL, retaining only a bounded
        recent window in memory for the on-screen proof panel. No log cap.
        """
        self.audit_count+=1
        entry={'id':self.audit_count,**item}
        if entry.get('type')=='DD' and not all(isinstance(x,int) for x in (entry.get('premises') or [])):
            textual=list(entry.get('premises') or [])
            # DD engine's textual citations are explanatory, not stable IDs.
            # A guaranteed sound, though possibly non-minimal, parent set is
            # the entire original assumption conjunction.
            entry['premiseStatements']=textual
            direct=[]
            for stmt in textual:
                # Only claim a precise reference if the cited geometric
                # property already has a logged source or the original native
                # engine confirms it and can explain the elimination pivots.
                for previous in self.audit_index.values():
                    if stmt in previous.get('conclusions',[]):direct.append(previous['id'])
                if not direct:
                    words=stmt.split()
                    if len(words)>=2 and words[0] in ('para','perp','cong','coll','cyclic','eqangle','eqratio'):
                        try:
                            from parse import AGPredicate
                            pred=AGPredicate(words[0],[self.points[t] for t in words[1:]],[])
                            if self.engine.check_pred(pred):direct.extend(self.engine.explain_pred(pred))
                        except (KeyError,ValueError,AssertionError,IndexError,ZeroDivisionError):pass
            entry['premises']=self._support(direct) if direct else self.input_ids[:]
            entry['dependencyLevel']=('exact earlier conclusions / elimination equations' if direct else 'macro: complete input conjunction; not minimal rule antecedents')
            entry['complete']=bool(direct)
        if entry.get('type')=='DD_APPLY' and not entry.get('premises'):
            entry['premises']=self.input_ids[:]
            entry['complete']=False
            entry['dependencyLevel']='macro: exhaustive input predicates, exact DD antecedents not available'
        self.audit_events.append(entry)
        self.audit_index[entry['id']]=entry
        self.audit_file.write(json.dumps(entry,ensure_ascii=False,default=str)+'\n')
        if self.audit_count%250==0:self.audit_file.flush()
        return entry['id']

    def _support(self, refs):
        return sorted({int(i) for i in refs if isinstance(i,int) and i in self.audit_index})

    def _audit_rule(self, entry):
        return self._audit(entry)

    def log_text(self):
        self.audit_file.flush()
        with open(self.audit_path,encoding='utf-8') as stream:
            return stream.read()

    def _key(self, data):
        return json.dumps([data['name'], data['points'], data.get('constants',[])],sort_keys=True)

    def _pred(self, item):
        constants=item.get('constants',[])
        if item['name'] in ('aconst','rconst'):
            rational=[]
            for value in constants:
                f=Fraction(str(value)); simple=f.limit_denominator(3600)
                rational.append(simple if abs(float(simple)-float(value))<1e-10 else f)
            constants=rational
        return AGPredicate(item['name'],[self.points[x] for x in item['points']],constants)

    def _phases(self):
        changed=True
        while changed:
            self.depth+=1
            self.engine.update_cache()
            changed=False
            for label,name in STAGES:
                self.engine.proof_stage=label
                method=getattr(self.engine,name)
                # Conservative provenance for a geometry-stage equation: the
                # original construction assumptions, not an invented single
                # direct theorem premise. Exact elimination parents are added
                # separately inside ElimCore.add_constraint.
                for domain in (self.engine.elim_angle,self.engine.elim_dist_mul,self.engine.elim_dist_add):
                    domain.core.current_sources=tuple(self.input_ids)
                self.active_rule_sources=self.input_ids
                try:
                    updated=method(verbose=False) if name=='search_similar' else method()
                finally:
                    self.active_rule_sources=[]
                    for domain in (self.engine.elim_angle,self.engine.elim_dist_mul,self.engine.elim_dist_add):
                        domain.core.current_sources=()
                changed=bool(updated) or changed
                if updated and name=='search_similar':self.engine.update_cache()
                self.stepname=label
                if updated:self._collect_native()
                yield

    def _collect_native(self):
        # Structured facts exposed by the *actual* AG2 geometric graph. This
        # supplements numerically screened candidate queries, without guessing
        # from screen coordinates or exhaustively inventing angle relations.
        for line in self.engine.lines:
            pts=sorted({str(p) for p in line.points})
            if len(pts)<3:continue
            for triple in itertools.combinations(pts,3):
                if len(set(triple))<3:continue
                code='coll '+ ' '.join(triple)
                self.native[code]={'id':'native:'+code,'queryId':None,
                                   'kind':'coll','args':list(triple),'text':code,
                                   'rule':'AG2 line closure','depth':self.depth,
                                   'premises':[]}
        for circle in self.engine.circles:
            pts=sorted({str(p) for p in circle.points})
            if len(pts)<4:continue
            for quad in itertools.combinations(pts,4):
                code='cyclic '+ ' '.join(quad)
                self.native[code]={'id':'native:'+code,'queryId':None,
                                   'kind':'cyclic','args':list(quad),'text':code,
                                   'rule':'AG2 circle closure','depth':self.depth,
                                   'premises':[]}

    def _one(self,q):
        if q['id'] in self.found and self.found[q['id']]['status']=='proved':return
        predicates=q.get('predicates',[])
        if not predicates:
            self.found[q['id']]={'id':q['id'],'status':'unsupported','reason':'无法转换为有效内核命题'}
            return
        try:
            ok=True
            for p in predicates:
                self.attempts+=1
                if not self.engine.check_pred(self._pred(p)):
                    ok=False;break
            if ok:
                source='构造定义' if all(self._key(p) in self.premises for p in predicates) else ('AR' if self.stepname.startswith('AR') else 'DDAR')
                self.first_proof_stage.setdefault(q['id'],self.stepname)
                refs=[]
                for item in predicates:
                    if self._key(item) in self.input_by_key:
                        refs.append(self.input_by_key[self._key(item)])
                    else:
                        try:refs.extend(self.engine.explain_pred(self._pred(item)))
                        except (ValueError, KeyError, AssertionError, ZeroDivisionError):pass
                refs=self._support(refs)
                if not refs:refs=self.input_ids[:]
                self._audit({'type':'INPUT' if source=='构造定义' else ('AR' if source=='AR' else 'DDAR'),
                     'stage':self.stepname,'rule':'goal/query proven by native check_pred',
                     'conclusions':[item['name']+' '+' '.join(item['points']) for item in predicates],
                     'queryId':q['id'],'premises':refs,
                     'dependencyLevel':'exact elimination pivots' if any(self.audit_index.get(i,{}).get('type') in ('AR','EQUATION') for i in refs) else 'macro input-conjunction',
                     'complete':bool(refs)})
                self.found[q['id']]={'id':q['id'],'status':'proved','rule':source,
                    'depth':self.depth,'premises':[],
                    'reason':('构造前提' if source=='构造定义' else 'AG2 原生闭包检验成立；点击查看可重放的前提来源证书')}
                if q['id']==self.goal_id:self.goal_found=True
            else:
                self.found[q['id']]={'id':q['id'],'status':'unknown','reason':'尚未证明'}
        except (KeyError,IndexError,AssertionError,ValueError,ZeroDivisionError,StopIteration) as exc:
            self.found[q['id']]={'id':q['id'],'status':'unsupported','reason':type(exc).__name__+': '+str(exc)}

    def _scan(self,limit=250,target_first=False):
        if target_first and self.goal_id:
            for q in self.queries:
                if q['id']==self.goal_id:self._one(q);break
        if not self.queries:return
        for _ in range(min(limit,len(self.queries))):
            q=self.queries[self.cursor]
            self.cursor=(self.cursor+1)%len(self.queries)
            self._one(q)
            if self.goal_found:break

    def step(self):
        if not self.closed and not self.goal_found:
            try:next(self.generator)
            except StopIteration:
                self.closed=True;self.stepname='closure complete'
                self.final_remaining=len(self.queries)
            self._scan(limit=250,target_first=True)
        if self.goal_found and self.payload.get('mode')=='goal':
            self.done=True;self.stepname='target proved'
        elif self.closed:
            if self.final_remaining:
                n=min(450,self.final_remaining)
                self._scan(limit=n)
                self.final_remaining-=n
            self.done=self.final_remaining<=0
        return self.progress()

    def progress(self):
        proven=[v for v in self.found.values() if v['status']=='proved']
        return {'done':self.done,'engine':'ag2','phase':self.stepname,'depth':self.depth,
                'attempts':self.attempts,'proved':len(proven),'checked':len(self.found),
                'totalCandidates':len(self.queries),'elapsedMs':round((time.perf_counter()-self.started)*1000),
                'queries':list(self.found.values()),'nativeFacts':list(self.native.values()),'pointCount':len(self.points),
                'premiseCount':len(self.premises),'goalFound':self.goal_found,
                'auditCount':self.audit_count,'auditTruncated':self.audit_truncated}

    def _replay(self, premises, goals, until_done=True):
        """Verify a proposed dependency witness with the ORIGINAL AG2 engine.

        Numeric coordinates are ONLY used to initialize official AGPoint values;
        neither replay nor the minimizer creates new hypotheses from them.
        The result is a checkable *DDAR macro certificate*, not a fictitious
        theorem-by-theorem derivation from AG2 (whose elimination tables do not
        carry symbolic dependency lists).
        """
        engine=DDAR(list(self.points.values()))
        try:
            for row in premises:
                engine.force_pred(self._pred(row))
            if all(engine.check_pred(self._pred(g)) for g in goals):
                return 'initialization'
            for _ in range(24):
                engine.update_cache()
                changed=False
                for label,name in STAGES:
                    op=getattr(engine,name)
                    updated=op(verbose=False) if name=='search_similar' else op()
                    changed=bool(updated) or changed
                    if updated and name=='search_similar':engine.update_cache()
                    if all(engine.check_pred(self._pred(g)) for g in goals):
                        return name
                if not changed:break
        except (KeyError,IndexError,AssertionError,ValueError,ZeroDivisionError,StopIteration,TypeError):
            return None
        return None

    def _minimize(self, goals):
        """Remove dispensable assumptions, preserving replay verification.

        Chunk elimination prevents a proof click from triggering one native
        closure per premise on large boards.  We deliberately do not claim the
        remaining witness is cardinality-minimal.
        """
        rows=self.premise_rows[:]
        if self._replay(rows,goals) is None:
            return None,None
        # For large diagrams, at most 32 replay trials. A valid (perhaps
        # nonminimal) certificate is preferable to UI lockups.
        budget=32
        group=max(1,(len(rows)+7)//8)
        while group>=1 and budget>0:
            i=0
            while i<len(rows) and budget>0:
                trial=rows[:i]+rows[i+group:]
                budget-=1
                if self._replay(trial,goals) is not None:
                    rows=trial
                else:i+=group
            if group==1:break
            group=max(1,group//2)
        return rows,self._replay(rows,goals)

    def proof(self, ident):
        if ident in self.proof_cache:return self.proof_cache[ident]
        query=next((q for q in self.queries if q['id']==ident),None)
        if query and self.found.get(ident,{}).get('status')!='proved':
            return {'available':False,'reason':'该结论尚未被 AG2 证明'}
        if query:
            goals=query.get('predicates',[])
        elif ident.startswith('native:') and ident[7:] in self.native:
            name,*args=ident[7:].split()
            goals=[{'name':name,'points':args,'constants':[]}]
        else:return {'available':False,'reason':'找不到对应已证命题'}
        if not goals:return {'available':False,'reason':'缺少有效 AG2 形式谓词'}
        rows,stage=self._minimize(goals)
        if stage is None:
            return {'available':False,'reason':'原生回放未能核验该结论的证明依赖；未生成可能虚假的证书'}
        steps=[]
        for item in rows:
            steps.append({'id':len(steps)+1,'predicate':item['name']+' '+' '.join(item['points'])+
                    (' '+','.join(map(str,item.get('constants',[]))) if item.get('constants') else ''),
                    'rule':'构造 / '+item.get('source','形式前提'),'premises':[],
                    'sourceType':'construction'})
        # A macro certificate explicitly names the exact ORIGINAL solver
        # operation and source predicates. It is not falsely attributed to a
        # named AG1 DD axiom. AR-only conclusions need no internal AR details.
        label={'initialization':'AR','search_similar':'DD · AG2 相似三角形搜索',
               'search_concyclic':'DD · AG2 共圆规则',
               'search_circles':'DD · AG2 圆规则',
               'merge_points':'DD · AG2 同一点合并',
               'transfer_dist_add_mul':'AR',
               'transfer_dist_arc_mul':'AR'}.get(stage,'DDAR · '+stage)
        premise_ids=[x['id'] for x in steps]
        for goal in goals:
            expression=goal['name']+' '+' '.join(goal['points'])
            if goal.get('constants'):expression+=' '+','.join(map(str,goal['constants']))
            direct=next((x for x in steps if x['predicate']==expression),None)
            if direct:continue
            steps.append({'id':len(steps)+1,'predicate':expression,
                          'rule':label,'premises':premise_ids[:],
                          'sourceType':'ar' if label=='AR' else 'native-replay',
                          'verifiedBy':'AG2 original closure replay'})
        if len(goals)>1:
            steps.append({'id':len(steps)+1,
                          'predicate':' ∧ '.join(x['name']+' '+' '.join(x['points']) for x in goals),
                          'rule':'目标合取','premises':[x['id'] for x in steps if x['predicate'] in
                              [g['name']+' '+' '.join(g['points']) for g in goals]]})
        # Events contain executed DD rules and AR equations, but their elimination
        # matrix has no strict parent pointers. Display them as an additional
        # audit trail, not as a fabricated sound theorem-level dependency DAG.
        events=[e for e in self.audit_events if e.get('type')=='DD'][-100:]
        relevant=[]
        for goal in goals:
            try:relevant.extend(self.engine.explain_pred(self._pred(goal)))
            except (ValueError,KeyError,AssertionError,ZeroDivisionError):pass
        used=set()
        def visit(ref):
            if ref in used:return
            event=self.audit_index.get(ref)
            if not event:return
            used.add(ref)
            for parent in event.get('premises',[]):
                if isinstance(parent,int):visit(parent)
        for ref in relevant:visit(ref)
        if used:
            dag=[self.audit_index[i] for i in sorted(used)]
            # The log's equation IDs and sources are maintained separately;
            # do not use fictitious rule numbers for algebra equations.
            steps=[{'id':e['id'], 'predicate':'；'.join(e.get('conclusions',[])),
                    'rule':e.get('rule',e['type']), 'premises':e.get('premises',[]),
                    'sourceType':'construction' if e['type']=='PREMISE' else ('dd' if e['type'] in ('DD','DD_APPLY') else 'ar'),
                    'dependencyLevel':e.get('dependencyLevel',''),
                    'complete':e.get('complete',True)}
                   for e in dag if e.get('conclusions')]
            steps.append({'id':self.audit_count+1,'predicate':' ∧ '.join(
                g['name']+' '+' '.join(g['points']) for g in goals),
                'rule':'AG2 符号关系检验','premises':sorted(set(relevant)),
                'sourceType':'ar'})
        proof={'available':True,'steps':steps,'events':events,'certificate':'AG2 原生重放核验；代数方程带主元依赖，DD 宏步骤可能只提供全体输入前提而非最小直接父命题',
               'verified':True,'stage':stage,'premiseCount':len(rows)}
        self.proof_cache[ident]=proof
        return proof

_SESSION=None

def start_json(src):
    global _SESSION
    _SESSION=Session(json.loads(src))
    return json.dumps(_SESSION.progress(),ensure_ascii=False)

def step_json():
    return json.dumps(_SESSION.step(),ensure_ascii=False)


def proof_json(id):
    return json.dumps(_SESSION.proof(id),ensure_ascii=False)


def audit_json():
    if _SESSION is None:
        return json.dumps({'available':False,'reason':'尚无 AG2 运行会话'})
    return json.dumps({'available':True,'filename':'geomtool_ag2_proof_log.jsonl',
                       'content':_SESSION.log_text(),'events':_SESSION.audit_count,
                       'truncated':_SESSION.audit_truncated},ensure_ascii=False)
