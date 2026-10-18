"""GeomTool v7.7: AG2 DDAR closure with versioned provenance and DD/AR audit.

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
        self.audit_path='/home/py/ag2/geomtool_ag2_v7.7_log.jsonl' if os.path.isdir('/home/py/ag2') else os.path.join(os.path.dirname(__file__),'geomtool_ag2_v7.7_log.jsonl')
        self.audit_file=open(self.audit_path,'w',encoding='utf-8')
        self.audit_count=0
        self.audit_index={}
        self.sources={}
        self.eq_nodes={}
        self.eq_serial=0
        self.state_by_object = {}    # immutable FormalLine/Circle -> event ID
        self.state_history = {'line': [], 'circle': [], 'point': []}
        self.point_merges = []
        # Original named assertions are never overwritten by point/line/circle merges.
        self.named_facts = {'coll': [], 'cyclic': [], 'overlap': [],
                            'cong': [], 'para': [], 'perp': [], 'eqratio': [],
                            'eqangle': [], 'angeq': [], 'rconst': [], 'aconst': []}
        self.goal_witness = {}
        self.input_by_label={}
        self.ar_cache={}
        self.engine=DDAR(list(self.points.values()))
        self.engine.b_theorems.set_extended_sin(bool(payload.get('extendedSinRatio',False)))
        self.engine.proof_logger=self._dd_event
        self.engine.proof_state_logger=self._state_event
        self.engine.proof_stage='assumptions'
        for key,core in [('ARang',self.engine.elim_angle.core),('ARmul',self.engine.elim_dist_mul.core),('ARadd',self.engine.elim_dist_add.core)]:
            core.on_add_constraint=lambda expr,refs,pivot,row,k=key:self._equation(k,expr,refs,pivot,row)
        for item in compiled['predicates']:
            label=self._label(item)
            ref=self._write({'type':'PREMISE','tag':'INPUT','conclusions':[label],
                             'premises':[],'source':item.get('source','construction')})
            self.input_by_label.setdefault(label,ref)
            if item['name'] in ('coll','cyclic','overlap'):
                self.sources.setdefault(item['name'],[]).append((ref,set(item['points'])))
            # force_pred canonicalizes already-merged point names. The original
            # submitted statement must carry the exact prior equality path,
            # otherwise e.g. A=B, B=C, C=D would falsely cite only C=D.
            aliases=set()
            for name in item['points']:
                representative=str(self.engine.point_subst[self.points[name]])
                chain=self._point_path(name,representative)
                if chain is None:
                    raise ValueError('缺失原始条件的点替换历史：'+name+' -> '+representative)
                aliases.update(chain)
            input_sources=tuple(sorted({ref}|aliases))
            for core in self._cores():core.current_sources=input_sources
            self.engine._next_dd_source=list(input_sources)
            self.engine.force_pred(self._pred(item))
            self.engine._next_dd_source=[]
        for core in self._cores():core.current_sources=()
        self.engine.proof_stage='initialization'
        self.queries=payload.get('queries',[])
        if len(self.queries)>1:
            raise ValueError('当前版本只接受一个明确的目标，或以零目标执行闭包')
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
        if item['name']=='cyclic_with_centers':
            # The first constant is an arity/count, not a rational angle or
            # distance.  DDAR uses it as a Python slice bound.
            if len(constants)!=1 or constants[0].denominator!=1 or constants[0]<0:
                raise ValueError('cyclic_with_centers 的圆心数量必须为非负整数')
            constants=[int(constants[0])]
        return AGPredicate(item['name'],[self.points[x] for x in item['points']],constants)

    def _write(self,record):
        self.audit_count+=1
        record={'id':self.audit_count,**record}
        self.audit_index[record['id']]=record
        if record.get('type') in ('PREMISE', 'DD', 'AR'):
            for statement in record.get('conclusions', ()):
                kind=statement.split(' ', 1)[0]
                if kind in self.named_facts:
                    self.named_facts[kind].append((record['id'], statement))
        self.audit_file.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
        if self.audit_count%250==0:self.audit_file.flush()
        return record['id']

    def _state_event(self, kind, operation, obj, parents, source, context):
        """Append-only object identity/version history, orthogonal to DDAR state.

        Bare two-point lines are intrinsic carriers (no incidence axiom).
        Only a genuine 3+-point incidence needs a proof source.
        """
        parent_ids = [self.state_by_object[p] for p in parents if p in self.state_by_object]
        parent_members = set().union(*(set(map(str,p.points)) for p in parents)) if parents else set()
        refs = set(parent_ids)
        for item in ([source] if source else []) + list(context):
            if isinstance(item, int) and item > 0 and item in self.audit_index:
                refs.add(item)
            elif isinstance(item, str):
                ref = self._structural_ref(item)
                if ref: refs.add(ref)
        if kind == 'point':
            a,b = obj
            record={'type':'POINT_MERGE', 'operation':operation,
                    'points':[str(a), str(b)], 'numericGuard':'coincident_coordinates',
                    'pointMergeRevision':len(self.point_merges)+1,
                    'reasonEvent':source, 'historicalNames':[str(a),str(b)],
                    'conclusions':['overlap '+str(a)+' '+str(b)],
                    'premises':sorted(refs)}
        else:
            members=list(map(str,obj.points))
            record={'type':'LINE_STATE' if kind=='line' else 'CIRCLE_STATE',
                    'operation':operation, 'points':members,
                    'introducedMembers':sorted(set(members)-parent_members),
                    'sharedWitness':sorted(set.intersection(*(set(map(str,p.points)) for p in parents))) if len(parents)>1 else [],
                    'pointMergeRevision':len(self.point_merges),
                    'objectVersion':len(self.state_history[kind])+1,
                    'revision':self.audit_count+1,
                    'numericGuard':'collinear_coordinates' if kind=='line' else 'concyclic_coordinates',
                    'parents':parent_ids, 'premises':sorted(refs),
                    'conclusions':[], 'source':source}
            if kind=='circle':
                record['definingPoints']=list(map(str,obj.defining_points))
                record['centers']=list(map(str,obj.centers))
                old_centers=set().union(*(set(map(str,p.centers)) for p in parents)) if parents else set()
                record['introducedCenters']=sorted(set(record['centers'])-old_centers)
        ident=self._write(record)
        self.state_history[kind].append(ident)
        if kind=='point':self.point_merges.append((str(obj[0]),str(obj[1]),ident))
        else:self.state_by_object[obj]=ident
        return ident

    def _named_incidence(self, typ, requested, before=None):
        """Minimal on-demand incidence substitutions from ORIGINAL fact names.

        A merged line/circle may contain hundreds of points; only the source
        incidence and the exact equality paths needed by the request are read.
        A fact cannot justify itself or use a future point merger.
        """
        result=[]
        for ident,statement in self.named_facts.get(typ, ()):
            if before is not None and ident>=before: continue
            source=statement.split()[1:]
            if len(source)<(3 if typ=='coll' else 4): continue
            references={ident}
            for name in requested:
                paths=[self._point_path(name,other,before) for other in source]
                paths=[p for p in paths if p is not None]
                if not paths: break
                references.update(min(paths,key=lambda p:(len(p),sorted(p))))
            else: result.append(references)
        return result

    def _structural_candidates(self, typ, requested, before=None):
        """Historical incidence witness: named facts first, versioned states second.

        A state includes all causal parent versions (not just a matching point
        set); a named witness can be much smaller after a point substitution.
        Every reference must predate the consuming DD step.
        """
        requested=list(requested)
        candidates=self._named_incidence(typ,requested,before)
        for ident in self.state_history.get('line' if typ=='coll' else 'circle', ()):
            if before is not None and ident>=before: continue
            ev=self.audit_index[ident]
            members=set(ev['points'])
            if typ=='cyclic' and len(members)<4: continue
            paths=[]
            for name in requested:
                if name in members: continue
                possibilities=[self._point_path(name,other,before) for other in members]
                possibilities=[p for p in possibilities if p is not None]
                if not possibilities: break
                paths.append(min(possibilities,key=lambda v:(len(v),sorted(v))))
            else:
                refs={ident}
                for p in paths: refs.update(p)
                candidates.append(refs)
        return candidates

    def _support_cost(self, refs):
        """Prefer less total ancestry, not a single large merged-state ID."""
        visited=set()
        def walk(ref):
            if ref in visited or ref not in self.audit_index: return
            visited.add(ref)
            for p in self.audit_index[ref].get('premises',()):walk(p)
        for ident in refs:walk(ident)
        return (len(visited),len(refs),sorted(refs))

    def _named_relation_candidates(self, data, before=None):
        """Use an old geometric assertion with only necessary point equalities.

        Handles segment order and interchange for congruence/parallelism and
        a conservative set of exact ratio symmetries. Other relations use the
        literal order and fall through to exact AR elimination if needed.
        """
        typ=data['name']; requested=data['points']
        if typ not in self.named_facts:return []
        out=[]
        for ident, statement in self.named_facts[typ]:
            if before is not None and ident>=before:continue
            source=statement.split()[1:]
            if len(source)<len(requested):continue
            src_points=source[:len(requested)]
            # Preserve constants: unlike segment order they cannot be renamed.
            if len(source)>len(requested) and source[len(requested):] != [','.join(map(str,data.get('constants',())))]:
                continue
            if typ in ('cong','para','perp') and len(src_points)==4:
                a,b,c,d=src_points
                variants=[(*u,*v) for u in ((a,b),(b,a)) for v in ((c,d),(d,c))]
                variants += [(x[2],x[3],x[0],x[1]) for x in variants]
            elif typ=='eqratio' and len(src_points)==8:
                a,b,c,d,e,f,g,h=src_points
                l=((a,b),(c,d));r=((e,f),(g,h))
                variants=[]
                for left,right in ((l,r),(r,l)):
                    for inv in (False,True):
                        aa,bb=(left[1],left[0]) if inv else left
                        cc,dd=(right[1],right[0]) if inv else right
                        variants.append((*aa,*bb,*cc,*dd))
            else:variants=[tuple(src_points)]
            for variation in variants:
                if len(variation)!=len(requested):continue
                refs={ident}
                for wanted,old in zip(requested,variation):
                    path=self._point_path(wanted,old,before)
                    if path is None:break
                    refs.update(path)
                else:out.append(refs)
        return out

    def _trivial_collinearity(self, requested, before=None):
        # An AG2 line containing <=2 distinct points needs no incidence axiom;
        # repeated names still need their historical point-equality witnesses.
        groups={}
        for name in requested:
            rep=str(self.engine.point_subst[self.points[name]])
            groups.setdefault(rep,set()).add(name)
        if len(groups)>2:return None
        refs=set()
        for names in groups.values():
            anchor=min(names)
            for name in names-{anchor}:
                path=self._point_path(anchor,name,before)
                if path is None:return None
                refs.update(path)
        return refs

    def _point_path(self, start, finish, before=None):
        if start==finish:return set()
        from collections import deque
        graph={}
        for a,b,ident in self.point_merges:
            if before is not None and ident>=before:continue
            graph.setdefault(a,[]).append((b,ident))
            graph.setdefault(b,[]).append((a,ident))
        q=deque([(start,())]); seen={start}
        while q:
            name,path=q.popleft()
            for name2,ref in graph.get(name,()):
                if name2 in seen:continue
                new=path+(ref,)
                if name2==finish:return set(new)
                seen.add(name2);q.append((name2,new))
        return None

    @staticmethod
    def _ar_terms(expression):
        return [{'variable':str(variable), 'numerator':coef.numerator,
                 'denominator':coef.denominator}
                for variable,coef in expression.d.items() if coef]

    def _equation(self,kind,expression,parents,pivot,row):
        # Private substitution DAG; NOT an AR log entry. The actual AR log is
        # created only if DD consumes a relation requiring this substitution.
        self.eq_serial+=1
        key=-self.eq_serial
        direct=tuple(next(core.current_sources for name,core in
                          [('ARang',self.engine.elim_angle.core),
                           ('ARmul',self.engine.elim_dist_mul.core),
                           ('ARadd',self.engine.elim_dist_add.core)] if name==kind))
        self.eq_nodes[key]={'type':kind,'expression':str(expression),'terms':self._ar_terms(expression),
                            'parents':tuple(parents),
                            'direct':direct, 'audit_at':self.audit_count}
        # Native elimination still stores negative equation IDs. The public
        # history stores their original, immutable values and pivot ancestry.
        # Avoid treating equation IDs as positive DD premise IDs.
        ancestors=sorted(set().union(*(self._roots(i) for i in parents)))
        refs=sorted(set(direct) | set(ancestors))
        self._write({'type':'AR_EQUATION','system':kind, 'equationId':key,
                     'expression':str(expression), 'terms':self._ar_terms(expression),
                     'pivot':str(pivot), 'reducedRow':self._ar_terms(row),
                     'equationParents':list(parents),
                     'directSources':list(direct),'premises':refs,
                     'conclusions':[], 'equationRevision':self.eq_serial})
        return key

    def _roots(self,ident,visited=None):
        visited=visited if visited is not None else set()
        if ident in visited:return set()
        visited.add(ident)
        if ident>0:return {ident} if ident in self.audit_index else set()
        node=self.eq_nodes.get(ident)
        return (set(node.get('direct',())) |
                set().union(*(self._roots(k,visited) for k in node['parents']))) if node else set()

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
        if typ=='overlap':
            path=self._point_path(*data['points'])
            return next(iter(path)) if path is not None and len(path)==1 else None
        candidates=self._structural_candidates(typ,data['points'])
        if candidates:
            # Multiple supports for a merged line/circle are represented by a
            # single history node, whose parents retain all historical causes.
            return next(iter(min(candidates,key=self._support_cost))) if all(len(c)==1 for c in candidates) else None
        required=set(data['points'])
        for ref,points in reversed(list(self.sources.get(typ,[]))):
            if required<=points:return ref
        return None

    def _demand_ar(self,statement,data):
        typ=data['name']
        tag='ARang' if typ in {'para','perp','eqangle','angeq','aconst'} else 'ARmul' if typ in {'cong','eqratio','rconst'} else None
        if tag is None:return []
        direct=self._named_relation_candidates(data,before=self.audit_count+1)
        if direct:return sorted(min(direct,key=self._support_cost))
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
                         'premises':supports,'rule':'exact symbolic elimination (used by DD)',
                         'arEquationRevision':self.eq_serial,
                         'readEquationIds':list(parents),
                         'readExpression':str(raw.comb)})
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
                    'arEquationRevision':self.eq_serial,
                    'readEquationIds':list(ar.get('sources',[])),
                    'rule':'exact AR antecedent consumed by native DD'})
            refs.append(self.ar_cache[cache])
        read_receipts=[]
        read_before=self.audit_count+1
        for statement in statements:
            start=len(refs)
            data=self._as_pred(statement)
            resolution='unresolved'
            if data is not None and data['name'] in ('coll','cyclic'):
                candidates=self._structural_candidates(data['name'],data['points'],before=read_before)
                if candidates:
                    refs.extend(min(candidates,key=self._support_cost))
                    resolution='incidence_history'
            if resolution=='unresolved' and data is not None and data['name']=='overlap':
                equality=self._point_path(*data['points'],before=read_before)
                if equality is not None:
                    refs.extend(equality)
                    resolution='point_merge_history'
            if resolution=='unresolved' and data is not None and data['name']=='coll':
                trivial=self._trivial_collinearity(data['points'],before=read_before)
                if trivial is not None:
                    refs.extend(trivial)
                    resolution='two_point_incidence'
            if resolution=='unresolved':
                ref=self._structural_ref(statement)
                if ref:
                    refs.append(ref)
                    resolution='earlier_fact'
                elif data:
                    refs.extend(self._demand_ar(statement,data))
                    if len(refs)>start:
                        resolution='ar_history'
                    elif data['name'] in ('para','perp','cong','eqratio','eqangle','angeq','rconst','aconst'):
                        try:
                            from geomtool_v64_proof import _algebra, _min_ar
                            kind, _, vector, _=_algebra(self,data)
                            if not vector or (kind=='ARang' and set(vector)=={'pi'}
                                              and vector['pi'].denominator==1):
                                resolution='exact_identity'
                            else:
                                evidence=_min_ar(self,data,before_event=read_before,
                                                  max_equation_revision=self.eq_serial)
                                if evidence:
                                    ref=self._write({'type':'AR', 'tag':kind,
                                        'conclusions':[statement],
                                        'premises':sorted(evidence),
                                        'readEquationRevision':self.eq_serial,
                                        'rule':'exact historical AR replay consumed by DD'})
                                    refs.append(ref)
                                    resolution='historical_ar_replay'
                        except (AssertionError,ValueError,KeyError,IndexError,ZeroDivisionError):
                            pass
            read_receipts.append({'statement':statement,'resolution':resolution,
                                  'sources':sorted(set(refs[start:]))})
        for state_kind, obj in (event.get('detail') or {}).get('stateParents',[]):
            if obj in self.state_by_object:refs.append(self.state_by_object[obj])
        unresolved=[x['statement'] for x in read_receipts
                    if x['resolution']=='unresolved']
        refs=sorted(set(refs))
        conclusions=event.get('conclusions',[])
        # Similarity DD_APPLY comes before its AR equations; all subsequent
        # derived DD facts cite this ID rather than repeating every hypothesis.
        ident=self._write({'type':'DD','tag':tag,'rule':rule,
                           'stage':stage,'conclusions':conclusions,
                           'premises':refs,
                           'readReceipts':read_receipts,
                           'unresolvedReads':unresolved,
                           'provenanceStatus':'partial' if unresolved else 'recorded',
                           'stateReads':[self.state_by_object[obj] for _,obj in
                                         (event.get('detail') or {}).get('stateParents',[])
                                         if obj in self.state_by_object],
                           'arReads':[{'tag':ar['tag'],'conclusion':ar['conclusion'],
                                       'equationIds':list(ar.get('sources',[])),
                                       'atEquationRevision':self.eq_serial}
                                      for ar in (event.get('detail') or {}).get('arNeeds',[])],
                           'detail':{k:v for k,v in (event.get('detail') or {}).items()
                                     if k not in ('arNeeds','stateParents')}})
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
        if r.startswith('b10 '):return 'B10'
        if r.startswith('b11 '):return 'B11'
        if r.startswith('b12 '):return 'B12'
        if r.startswith('b13 '):return 'B13'
        if r.startswith('b14 '):return 'B14'
        if r.startswith('arc-to-chord'):return 'A1'
        if r.startswith('chord-to-arc'):return 'A2'
        # Unknown native operations retain their actual rule name as mnemonic.
        return 'DD_UNCLASSIFIED'

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
            if proved:
                self.goal_found=True
                if q['id'] not in self.goal_witness:
                    observations=[]
                    for row in predicates:
                        kind=row['name']; supports=[]; ar_roots=[]
                        if kind in ('coll','cyclic'):
                            candidates=self._structural_candidates(kind,row['points'])
                            if candidates:supports=sorted(min(candidates,key=self._support_cost))
                        elif kind=='overlap':
                            supports=sorted(self._point_path(*row['points']) or ())
                        elif kind in ('para','perp','cong','eqratio','eqangle','angeq','rconst','aconst'):
                            # An AR goal read is recorded with its row revision
                            # below. Do not create a redundant new deduction.
                            try:
                                ar_roots=list(self.engine.explain_pred(self._pred(row)))
                            except (ValueError,AssertionError,KeyError,IndexError,ZeroDivisionError):
                                ar_roots=[]
                        else: ar_roots=[]
                        ar_terms=None
                        if kind in ('para','perp','cong','eqratio','eqangle','angeq','rconst','aconst'):
                            # Store the precise named AR query, not only the
                            # *current* reduced pivot IDs. It is an immutable
                            # witness against which old equations can replay.
                            try:
                                from geomtool_v64_proof import _algebra
                                ar_system, _, original, _ = _algebra(self,row)
                                ar_terms=[{'variable':v,'numerator':c.numerator,
                                           'denominator':c.denominator} for v,c in original.items()]
                            except (AssertionError,ValueError,KeyError,IndexError,ZeroDivisionError):
                                ar_system=None
                        else:
                            ar_system=None
                        observations.append({'predicate':self._label(row),'sources':supports,
                                             'auditRevision':self.audit_count,
                                             'pointMergeRevision':len(self.point_merges),
                                             'arEquationRevision':self.eq_serial,
                                             'arEquationRoots':ar_roots,
                                             'arReadSystem':ar_system,
                                             'arReadTerms':ar_terms})
                    self.goal_witness[q['id']]=observations
                    self._write({'type':'GOAL_WITNESS','conclusions':[x['predicate'] for x in observations],
                                 'reads':observations,'premises':sorted(set().union(*(set(x['sources']) for x in observations)))})
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
                    self.engine.proof_stage='B1–B14 / optional B5 sin bridge'
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
    return json.dumps({'available':True,'filename':'geomtool_ag2_v7.7_log.jsonl',
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
