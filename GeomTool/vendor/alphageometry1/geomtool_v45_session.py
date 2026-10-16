"""AG1's original DD/AR algorithms executed cooperatively one BFS level at a time.
The theorem matcher, algebra tables and graph propagation are unmodified.
"""
import json
import time
import os
from fractions import Fraction
import ddar
import dd
import graph
import problem
import trace_back

_DEFS=None
_RULES=None
_SESSION=None

class Session:
    def __init__(self,payload):
        global _DEFS,_RULES
        self.payload=payload
        self.started=time.perf_counter()
        self.queries=payload.get('queries',[])
        self.found={}
        self.depth=0
        self.attempts=0
        self.phase='initialization'
        self.done=False
        self.closed=False
        self.final_remaining=0
        self.goal_id=payload.get('targetId')
        self.goal_found=False
        self.problem=payload['ag1']
        if _DEFS is None:
            base=os.path.dirname(__file__)
            _DEFS=problem.Definition.from_txt_file(os.path.join(base,'defs.txt'),to_dict=True)
            _RULES=problem.Theorem.from_txt_file(os.path.join(base,'rules.txt'),to_dict=True)
        # AG1's native numerical builder may retry forever when distinct
        # symbolic points have identical explicit sketch coordinates. Reject
        # ambiguous equalities *before* entering that loop: never merge points
        # from screen proximity alone or pass them as known equalities.
        coords=list(self.problem['pointCoordinates'].items())
        for i,(name,x) in enumerate(coords):
            for other,y in coords[i+1:]:
                if (x[0]-y[0])**2+(x[1]-y[1])**2<1e-16:
                    raise ValueError('AG1 存在未由构造证明为同一点的坐标重合 '+name+' / '+other+
                                     '；为避免原生建图无限重试，拒绝该退化输入')
        if not self.problem['problem']:
            raise ValueError('No AG1 point constructors')
        p=problem.Problem.from_txt(self.problem['problem'],translate=False)
        self.g,_=graph.Graph.build_problem(p,_DEFS,verbose=False)
        self.controller=p
        self.native={}
        self.premises=[]
        for item in self.problem['predicates']:
            args=[self.g.get(x,lambda x=x:x) for x in item['points']]
            name=item['name']
            if name=='rconst':
                n=Fraction(str(item['constants'][0])).limit_denominator(10000)
                if abs(float(n)-float(item['constants'][0]))>1e-12:
                    raise ValueError('AG1 unsupported irrational length coefficient: '+str(item))
                args+=[str(n.numerator)+'/'+str(n.denominator)]
            elif name=='aconst':
                n=Fraction(str(float(item['constants'][0])/180)).limit_denominator(360)
                if abs(float(n)*180-float(item['constants'][0]))>1e-10:
                    raise ValueError('AG1 unsupported angle coefficient: '+str(item))
                args+=[str(n.numerator)+'pi/'+str(n.denominator)]
            try:
                deps=problem.EmptyDependency(level=0,rule_name=problem.CONSTRUCTION_RULE)
                added=self.g.add_piece(name,args,deps)
                for d in added:
                    self.g.add_algebra(d,level=0)
                    self._register(d)
                self.premises.append({'name':name,'points':item['points'],'source':item['source']})
            except (ValueError,AssertionError,KeyError,TypeError,IndexError,ZeroDivisionError) as exc:
                raise ValueError('AG1 failed to inject construction premise '+str(item)+': '+str(exc)) from exc
        self.times=[]
        self.generator=self._steps()
        self._scan(10,target_first=True)

    def _register(self,dep):
        if not hasattr(dep,'name') or not hasattr(dep,'args'):return
        kind=str(dep.name)
        if kind not in {'coll','collx','para','perp','cong','midp','cyclic','eqangle','eqratio','rconst','aconst','circle','s_angle','eqangle6'}:return
        args=[str(getattr(x,'name',x)) for x in dep.args]
        expr=kind+' '+' '.join(args)
        self.native[expr]={'id':'native:'+expr,'queryId':None,'kind':kind,'args':args,
                           'text':expr,'rule':str(dep.rule_name or 'AR'),
                           'depth':self.depth,'premises':[]}

    def _steps(self):
        derv,eq4=self.g.derive_algebra(level=0,verbose=False)
        derives=[derv];eq4s=[eq4]
        maxlevel=int(self.payload.get('maxLevel') or 1000)
        timeout=int(self.payload.get('levelTimeout') or 600)
        while self.depth<maxlevel:
            while self.depth<maxlevel:
                start=time.time()
                self.depth+=1
                added,d,e,_=dd.bfs_one_level(self.g,_RULES,self.depth,self.controller,verbose=False,nm_check=True,timeout=timeout)
                self.times.append(time.time()-start)
                for dep in added:self._register(dep)
                derives.append(d);eq4s.append(e)
                self.phase='DD theorem BFS'
                yield
                if self.goal_found or not added or self.times[-1]>timeout:break
            if self.goal_found:break
            inserted=[]
            while derives and not inserted:inserted+=dd.apply_derivations(self.g,derives.pop(0))
            for dep in inserted:self._register(dep)
            self.phase='AR algebra transfer'
            yield
            if self.goal_found:break
            if inserted:continue
            while eq4s and not inserted:inserted+=dd.apply_derivations(self.g,eq4s.pop(0))
            for dep in inserted:self._register(dep)
            self.phase='AR angle/ratio transfer'
            yield
            if self.goal_found:break
            if not inserted:break

    def _one(self,q):
        if q['id'] in self.found and self.found[q['id']]['status']=='proved':return
        goal=q.get('ag1Goal')
        if not goal:
            self.found[q['id']]={'id':q['id'],'status':'unsupported','reason':'AG1 不支持该种形式谓词'}
            return
        try:
            name,*args=goal.split()
            args=[self.g.get(x,lambda x=x:x) for x in args]
            self.attempts+=1
            success=bool(self.g.check(name,args))
            if success:
                rule='AR' if self.phase.startswith('AR') else 'AG1 DDAR'
                self.found[q['id']]={'id':q['id'],'status':'proved','rule':rule,'depth':self.depth,
                        'premises':[],'reason':'由原版 AG1 符号图确认，可查看原生证明 DAG'}
                if q['id']==self.goal_id:self.goal_found=True
            else:self.found[q['id']]={'id':q['id'],'status':'unknown','reason':'尚未证明'}
        except (KeyError,ValueError,IndexError,AssertionError,TypeError,ZeroDivisionError) as exc:
            self.found[q['id']]={'id':q['id'],'status':'unsupported','reason':type(exc).__name__+': '+str(exc)}

    def _scan(self,limit,target_first=False):
        if target_first and self.goal_id:
            for q in self.queries:
                if q['id']==self.goal_id:self._one(q);break
        if not self.queries:return
        cursor=getattr(self,'cursor',0)
        for _ in range(min(limit,len(self.queries))):
            q=self.queries[cursor];cursor=(cursor+1)%len(self.queries)
            self._one(q)
            if self.goal_found:break
        self.cursor=cursor

    def step(self):
        if not self.closed and not self.goal_found:
            try:next(self.generator)
            except StopIteration:
                self.closed=True;self.phase='closure complete'
                self.final_remaining=len(self.queries)
            self._scan(250,target_first=True)
        if self.goal_found and self.payload.get('mode')=='goal':
            self.done=True;self.phase='target proved'
        elif self.closed:
            if self.final_remaining:
                n=min(450,self.final_remaining)
                self._scan(n)
                self.final_remaining-=n
            self.done=self.final_remaining<=0
        return self.progress()

    def progress(self):
        return {'engine':'ag1','phase':self.phase,'depth':self.depth,'attempts':self.attempts,
                'proved':sum(x['status']=='proved' for x in self.found.values()),'checked':len(self.found),
                'totalCandidates':len(self.queries),'elapsedMs':round((time.perf_counter()-self.started)*1000),
                'queries':list(self.found.values()),'nativeFacts':list(self.native.values()),'pointCount':len(self.problem['pointCoordinates']),
                'premiseCount':len(self.premises),'goalFound':self.goal_found,'done':self.done}

    def proof(self,id):
        q=next((q for q in self.queries if q['id']==id),None)
        native=self.native.get(id.removeprefix('native:')) if id.startswith('native:') else None
        if q is None and native is not None:q={'ag1Goal':native['text']}
        if q is None or (native is None and self.found.get(id,{}).get('status')!='proved'):
            return {'available':False,'reason':'尚未确认'}
        try:
            text=q.get('ag1Goal')
            name,*names=text.split()
            query=problem.Dependency(name,self.g.names2points_or_int(names),None,None)
            setup,aux,log,_=trace_back.get_logs(query,self.g,merge_trivials=False)
            mapping={}
            steps=[]
            def desc(d):return str(d.name)+' '+ ' '.join(getattr(x,'name',str(x)) for x in d.args)
            def key(d):return str(d.hashed())
            def register(d,rule,parents=()):
                k=key(d)
                if k not in mapping:
                    num=len(steps)+1;mapping[k]=num
                    steps.append({'id':num,'predicate':desc(d),'rule':rule,'premises':list(parents)})
                return mapping[k]
            def signature(name,pts):
                if name in ('coll','cyclic'):return (name,tuple(sorted(pts)))
                if name in ('cong','para','perp') and len(pts)==4:
                    return (name,tuple(sorted((tuple(sorted(pts[:2])),tuple(sorted(pts[2:]))))))
                return (name,tuple(pts))
            sources={signature(x['name'],x['points']):x['source'] for x in self.premises}
            def source_of(d):
                pts=[getattr(x,'name',str(x)) for x in d.args]
                origin=sources.get(signature(d.name,pts))
                return '构造 / '+origin if origin else '作图定义 / 已知前提'
            for d in setup:register(d,source_of(d))
            for d in aux:register(d,source_of(d))
            for prems,cons in log:
                ids=[register(d,'构造 / 已知前提') for d in prems]
                for d in cons:register(d,d.rule_name or 'AR',ids)
            if not steps:
                # The upstream tracer may omit a fact that is directly an
                # assumption because there is no deduction DAG to traverse.
                if not any(row['name']==name and row['points']==names for row in self.premises):
                    return {'available':False,'reason':'该命题经 AG1 确认，但原生 trace_back 未返回可核实的依赖链'}
                steps=[{'id':1,'predicate':text,'rule':'构造定义 / 已知形式前提','premises':[]}]
            return {'available':bool(steps),'steps':steps}
        except Exception as exc:
            return {'available':False,'reason':'原版依赖回溯失败：'+type(exc).__name__+': '+str(exc)}

def start_json(text):
    global _SESSION
    _SESSION=Session(json.loads(text));return json.dumps(_SESSION.progress(),ensure_ascii=False)

def step_json():
    return json.dumps(_SESSION.step(),ensure_ascii=False)

def proof_json(id):
    return json.dumps(_SESSION.proof(id),ensure_ascii=False)
