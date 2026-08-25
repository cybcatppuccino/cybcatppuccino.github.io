from pathlib import Path
import csv, json, re, glob, collections, math, shutil, zipfile, hashlib, html
ROOT=Path('/mnt/data')
WORK=ROOT/'v4_work'
V3=ROOT/'FrenchCards_V3_Current_Checked'
OUT=ROOT/'FrenchCards_V4_Merged'
if OUT.exists(): shutil.rmtree(OUT)
(OUT/'FrenchCards_V4_Lazy'/'data').mkdir(parents=True)
(OUT/'tools'/'source_extracts').mkdir(parents=True)

def norm(x):
    if x is True: x='vrai'
    elif x is False: x='faux'
    elif x is None: x=''
    else: x=str(x)
    return x.strip().lower().replace('’',"'").replace('œ','oe')

def num(x):
    try:
        if x is None or str(x).strip()=='': return None
        return float(x)
    except: return None

def intnum(x):
    v=num(x)
    return int(v) if v is not None else None

D=json.load(open(V3/'French_Core_3000_V3_AppData.json',encoding='utf-8'))
W=D['words']
core_norm={norm(w['fr']):w for w in W}

# ---------- MorphoLex ----------
morph={}
for f in glob.glob(str(WORK/'morpholex/morpholex-fr-master/csv/[0-9]*.csv')):
    with open(f,encoding='utf-8-sig',newline='') as fh:
        for row in csv.DictReader(fh): morph[norm(row.get('item'))]=row

def clean_morpheme(s):
    s=re.sub(r'\[[^\]]+\]','',s or '').strip(' <>')
    s=s.replace('>>','>').replace('<<','<').strip()
    return s

def morpholex_struct(row):
    if not row: return None
    nm=intnum(row.get('n_morphemes')) or 1
    if nm<=1: return None
    raw=row.get('canon_segm') or ''
    prefixes=[clean_morpheme(x) for x in re.findall(r'<([^<>()>]+)<',raw)]
    roots=[clean_morpheme(x) for x in re.findall(r'\(([^()]*)\)',raw)]
    suffixes=[]
    for x in re.findall(r'>([^<>]+)>',raw):
        x=clean_morpheme(x)
        if x and not x.startswith('['): suffixes.append(x)
    comps=[]
    for p in prefixes:
        if p: comps.append({'type':'prefix','form':p+'-'})
    for r in roots:
        if r: comps.append({'type':'root','form':r})
    for s in suffixes:
        if s: comps.append({'type':'suffix','form':'-'+s})
    if len(comps)<2: return None
    seg=' + '.join(c['form'] for c in comps)
    return {'kind':'derivation','segmentation':seg,'components':comps,'morpheme_count':nm}

# ---------- Lexique 4 ----------
lex_by_word=collections.defaultdict(list); lex_by_lemma=collections.defaultdict(list)
lex_path=WORK/'lexique/Lexique400short/Lexique4/Lexique4.tsv'
with open(lex_path,encoding='utf-8-sig',newline='') as fh:
    for row in csv.DictReader(fh,delimiter='\t'):
        lex_by_word[norm(row.get('1_Mot'))].append(row)
        lex_by_lemma[norm(row.get('4_Lemme'))].append(row)
POS_CODES={
 'noun':{'NOM'},'verb':{'VER','AUX'},'adj':{'ADJ'},'adv':{'ADV'},'prep':{'PRE'},
 'conj':{'CON'},'det':{'DET','ADJ:num'},'pron':{'PRO','PRO:ind','PRO:dem','PRO:per','PRO:rel'},
 'num':{'NUM','ADJ:num'},'interj':{'ONO','INT'}
}
def pos_match(pos,cgram):
    if not cgram:return False
    allowed=POS_CODES.get(pos,set())
    return cgram in allowed or any(cgram.startswith(x+':') for x in allowed)

def preferred_lex_rows(w):
    k=norm(w['fr']); pos=w.get('pos')
    lemma_rows=[r for r in lex_by_lemma.get(k,[]) if r.get('14_IsLem')=='1']
    exact_rows=[r for r in lex_by_word.get(k,[]) if norm(r.get('4_Lemme'))==k]
    rows=lemma_rows or exact_rows or lex_by_word.get(k,[]) or lex_by_lemma.get(k,[])
    pm=[r for r in rows if pos_match(pos,r.get('5_Cgram',''))]
    return pm or rows

def lex_profile(w):
    k=norm(w['fr']); pos=w.get('pos')
    rows=preferred_lex_rows(w)
    if not rows:return None
    exact=[r for r in lex_by_word.get(k,[]) if pos_match(pos,r.get('5_Cgram',''))]
    if not exact: exact=lex_by_word.get(k,[])
    # word frequency: sum exact orthography across relevant grammatical categories, avoiding duplicate category copies.
    wf_by_cat={}
    for r in exact:
        f=num(r.get('10_FreqMot'))
        if f is None: continue
        c=r.get('5_Cgram') or '?'
        wf_by_cat[c]=max(wf_by_cat.get(c,0),f)
    word_freq=sum(wf_by_cat.values()) if wf_by_cat else None
    # lemma frequency: one value per grammatical category, then sum relevant categories.
    lf_by_cat={}
    for r in rows:
        f=num(r.get('12_FreqLemme'))
        if f is None: continue
        c=r.get('5_Cgram') or '?'
        lf_by_cat[c]=max(lf_by_cat.get(c,0),f)
    lemma_freq=sum(lf_by_cat.values()) if lf_by_cat else None
    f=lemma_freq if lemma_freq is not None else word_freq
    if f is None:return None
    if f>=500: band=('very_high','超高频')
    elif f>=100: band=('high','高频')
    elif f>=20: band=('common','常用')
    elif f>=5: band=('medium','中频')
    elif f>=1: band=('lower','较低频')
    else: band=('low','低频')
    # exact lemma row for pronunciation / syllables / practice features.
    best=None
    for r in rows:
        if r.get('14_IsLem')=='1' and norm(r.get('1_Mot'))==k:
            best=r;break
    if best is None: best=rows[0]
    syll=intnum(best.get('26_SyllNb'))
    old=num(best.get('17_OLD20'))
    vois=intnum(best.get('21_VoisOrtho'))
    out={'frequency_per_million':round(f,3),'band':band[0],'band_zh':band[1]}
    if word_freq is not None: out['headword_frequency_per_million']=round(word_freq,3)
    if syll is not None: out['syllables']=syll
    ipa=(best.get('3_Phono_IPA') or '').strip()
    ipa_reference='/'+ipa.strip('/')+'/' if ipa else None
    # Practice-only values are intentionally compact.
    out['practice']={'length':len(w['fr'].replace(' ',''))}
    if old is not None: out['practice']['old20']=round(old,3)
    if vois is not None: out['practice']['orthographic_neighbors']=vois
    return out


def lex_inflections(w):
    if w.get('pos') not in ('noun','adj'): return None
    k=norm(w['fr']); cgram='NOM' if w['pos']=='noun' else 'ADJ'
    rows=[r for r in lex_by_lemma.get(k,[]) if r.get('5_Cgram')==cgram]
    if not rows:return None
    buckets=collections.defaultdict(list)
    for r in rows:
        form=(r.get('1_Mot') or '').strip()
        if not form:continue
        g=(r.get('7_Genre') or '').strip(); n=(r.get('8_Nombre') or '').strip()
        f=num(r.get('10_FreqMot')) or 0
        keys=[]
        genders=['m','f'] if g in ('e','') and w['pos']=='adj' else ([g] if g in ('m','f') else [''])
        numbers=['s','p'] if n=='i' else ([n] if n in ('s','p') else [''])
        for gg in genders:
            for nn in numbers:buckets[(gg,nn)].append((f,form))
    def pick(key):
        vals=buckets.get(key) or []
        if not vals:return None
        vals=sorted(vals,key=lambda x:(-x[0],len(x[1]),x[1]))
        return vals[0][1]
    out={}
    labels={( 'm','s'):'masculine_singular',('f','s'):'feminine_singular',('m','p'):'masculine_plural',('f','p'):'feminine_plural'}
    for key,label in labels.items():
        v=pick(key)
        if v:out[label]=v
    # Nouns with only number but no gender metadata: still preserve common singular/plural.
    if w['pos']=='noun':
        if not any(k.endswith('singular') for k in out):
            v=pick(('', 's'))
            if v:out['singular']=v
        if not any(k.endswith('plural') for k in out):
            v=pick(('', 'p'))
            if v:out['plural']=v
    # When Lexique only supplies a marked plural (e.g. oeil -> yeux), use the curated headword/gender for the missing singular.
    if w['pos']=='noun' and w.get('gender') in ('m','f'):
        gkey='masculine_singular' if w['gender']=='m' else 'feminine_singular'
        pkey='masculine_plural' if w['gender']=='m' else 'feminine_plural'
        if pkey in out and gkey not in out: out[gkey]=w['fr']
    # Do not add a section when it contains only the unchanged headword.
    vals=set(out.values())
    if not out or (len(vals)==1 and norm(next(iter(vals)))==k):return None
    return out

# ---------- Cobb 2026 ----------
list_vals=json.load(open(WORK/'cobb_list.json',encoding='utf-8'))
comp_vals=json.load(open(WORK/'cobb_complete.json',encoding='utf-8'))
list_headers=list_vals[0]; comp_headers=comp_vals[0]
clist={}
for r in list_vals[1:]:
    d=dict(zip(list_headers,r)); clist[norm(d.get('Headword '))]=d
family_members=collections.defaultdict(list); comp_by_word=collections.defaultdict(list)
for r in comp_vals[1:]:
    d=dict(zip(comp_headers,r))
    word=norm(d.get('Word')); head=norm(d.get('Headword'))
    if not word or not head: continue
    # repair spreadsheet boolean coercion of vrai/faux
    d['Word']='vrai' if d.get('Word') is True else ('faux' if d.get('Word') is False else d.get('Word'))
    d['Headword']='vrai' if d.get('Headword') is True else ('faux' if d.get('Headword') is False else d.get('Headword'))
    comp_by_word[word].append(d); family_members[head].append(d)

def family_profile(w):
    rows=comp_by_word.get(norm(w['fr']),[])
    if not rows:return None
    # Prefer family row with highest family frequency.
    d=max(rows,key=lambda x:(num(x.get('Family_Freq')) or 0))
    head=norm(d.get('Headword')); members=family_members.get(head,[])
    members=sorted(members,key=lambda x:(num(x.get('Orthographic_Freq')) or 0),reverse=True)
    seen=set(); rel=[]
    for m in members:
        form='vrai' if m.get('Word') is True else ('faux' if m.get('Word') is False else str(m.get('Word') or '').strip())
        n=norm(form)
        if not form or n in seen or n==norm(w['fr']): continue
        seen.add(n)
        rel.append({'form':form,'frequency':int(num(m.get('Orthographic_Freq')) or 0)})
        if len(rel)>=10:break
    level=str(d.get('List') or '').replace('List_','')
    out={'headword':str(d.get('Headword') or ''),'level':level,'family_frequency':int(num(d.get('Family_Freq')) or 0),'related_forms':rel}
    return out

# ---------- Merge ----------
counts=collections.Counter()
for w in W:
    # Morphology: keep manually curated V3 explanation when present, but enrich with canonical MorphoLex structure.
    ml=morpholex_struct(morph.get(norm(w['fr'])))
    curated=w.get('morphology_v3')
    if curated:
        mv4=json.loads(json.dumps(curated,ensure_ascii=False))
        if ml:
            mv4['canonical_segmentation']=ml['segmentation']
            mv4['morpheme_count']=ml['morpheme_count']
            counts['morph_curated_and_morpholex']+=1
        w['morphology_v4']=mv4;counts['morph_total']+=1
    elif ml:
        w['morphology_v4']=ml;counts['morph_total']+=1;counts['morph_new_from_morpholex']+=1
    lp=lex_profile(w)
    if lp:
        w['frequency_v4']=lp;counts['lexique_profile']+=1
    infl=lex_inflections(w)
    if w.get('pos')=='adj' and w.get('adjective_forms'):
        infl=None
    if infl:
        w['inflections_v4']=infl;counts['lexique_inflections']+=1
    fp=family_profile(w)
    if fp:
        w['word_family_v4']=fp;counts['cobb_family']+=1
        if fp.get('related_forms'): counts['cobb_with_related']+=1
    # Remove inherited technical normalization metadata from learner payload.
    w.pop('source_form',None)
    # V4 tags: useful learning/selection tags without changing legacy V3 tags.
    tags=list(w.get('learning_tags_v3') or [])
    if lp:
        tags.append('freq:'+lp['band'])
        if lp.get('syllables') is not None: tags.append('syllables:'+str(lp['syllables']))
    if fp:
        tags.append('family:L'+str(fp.get('level') or '?'))
    if w.get('morphology_v4'): tags.append('morphology')
    w['learning_tags_v4']=list(dict.fromkeys(tags))

# meta
D['meta']['app_data_enrichment']='V4 local merged: MorphoLex-FR morphology, Lexique 4 learning frequency/practice profile, Cobb 2026 word-family data; learner-facing HTML remains fully local.'
D['meta']['app_version']='V4'
D['meta']['v4_import_counts']=dict(counts)

# ---------- Build HTML ----------
LAZY=OUT/'FrenchCards_V4_Lazy'; STAND=OUT/'FrenchCards_V4_Standalone.html'
base=(V3/'FrenchCards_V3_Lazy'/'index.html').read_text(encoding='utf-8')
s=base.replace('French Core 3000 · Cards V3','French Core 3000 · Cards V4').replace('V3 local enhanced','V4 merged local')
# Small styles for new compact sections.
insert_css='''\n.freqline{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}.freqpill,.familyform{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:4px 8px;font-size:13px;background:var(--soft)}.familywrap{display:flex;gap:6px;flex-wrap:wrap}.familyform b{font-weight:650}.morphcanon{font-size:13px;color:var(--sub);margin-top:7px}.usagegrid{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;font-size:14px}.usagegrid .k{color:var(--sub)}\n'''
s=s.replace('</style>',insert_css+'</style>')
# Replace morphology function and add family/frequency renderers.
old_start=s.index('function renderMorphV3')
old_end=s.index('function renderCognates')
new_funcs=r'''function renderMorphV4(w){let m=w.morphology_v4||w.morphology_v3;if(!m)return'';let parts=(m.components||[]).map(c=>`<span class="morphpart"><b>${esc(c.form||'')}</b>${c.zh?' · '+esc(c.zh):''}</span>`).join('');let fam=(m.family||[]).map(x=>`${esc(x.form||'')}${x.zh?' · '+esc(x.zh):''}`).join('；');let canon=m.canonical_segmentation&&m.canonical_segmentation!==m.segmentation?`<div class="morphcanon">规范构词：${esc(m.canonical_segmentation)}</div>`:'';return `<section class="section"><h3>${m.kind==='compound'?'构词 / 组成':m.kind==='pronominal'?'代词式结构':'词根 / 词缀'}</h3><div class="morphbox"><div class="morphseg" lang="fr">${esc(m.segmentation||w.fr)}</div>${parts?`<div class="morphparts">${parts}</div>`:''}${canon}${fam?`<div class="morphnote">同词族：${fam}</div>`:''}${m.note_zh?`<div class="morphnote">${esc(m.note_zh)}</div>`:''}</div></section>`}
function renderFamilyV4(w){let f=w.word_family_v4;if(!f||!f.related_forms?.length)return'';let forms=f.related_forms.map(x=>`<span class="familyform" lang="fr"><b>${esc(x.form)}</b></span>`).join('');let level=f.level?`<span class="freqpill">词族层级 ${esc(f.level)}</span>`:'';return `<section class="section"><h3>词族 / 常见词形</h3><div class="familywrap">${forms}</div>${level?`<div class="freqline">${level}</div>`:''}</section>`}
function renderFrequencyV4(w){let f=w.frequency_v4;if(!f)return'';let rows=[];if(f.frequency_per_million!=null)rows.push(['现代词频',`${Number(f.frequency_per_million).toLocaleString()} / 百万词`]);if(f.headword_frequency_per_million!=null&&Math.abs(f.headword_frequency_per_million-f.frequency_per_million)>.001)rows.push(['当前词形',`${Number(f.headword_frequency_per_million).toLocaleString()} / 百万词`]);if(f.syllables!=null)rows.push(['音节',String(f.syllables)]);if(!rows.length)return'';return `<section class="section"><h3>使用信息</h3><div class="usagegrid">${rows.map(([k,v])=>`<div class="k">${esc(k)}</div><div>${esc(v)}</div>`).join('')}</div></section>`}
'''
s=s[:old_start]+new_funcs+s[old_end:]
# renderWord: frequency chip and new sections.
s=s.replace("${w.gender?`<span class=\"chip\">${esc(genderLabel(w.gender))}</span>`:''}</div><div class=\"wordrow\">", "${w.gender?`<span class=\"chip\">${esc(genderLabel(w.gender))}</span>`:''}${w.frequency_v4?.band_zh?`<span class=\"chip\">${esc(w.frequency_v4.band_zh)}</span>`:''}</div><div class=\"wordrow\">")
s=s.replace("html+=renderMorphV3(w);html+=renderCognates", "html+=renderMorphV4(w);html+=renderFamilyV4(w);html+=renderFrequencyV4(w);html+=renderCognates")
s=s.replace("html+=renderKVSection('性别形式',w.gender_forms);html+=renderMorphV4(w);", "html+=renderKVSection('性别形式',w.gender_forms);html+=renderKVSection('标准词形',w.inflections_v4);html+=renderMorphV4(w);")
s=s.replace("stem:'词干'};", "stem:'词干',singular:'单数',plural:'复数'};")
# Search index will include frequency/length features and improved distractor scoring.
# Build index/packs first then standalone injection.
index=[]
for w in W:
    fp=w.get('frequency_v4') or {}; pr=fp.get('practice') or {}
    index.append({'r':w['rank'],'f':w['fr'],'z':w.get('meaning',{}).get('zh',''),'e':w.get('meaning',{}).get('en',''),
                  'pos':w.get('pos',''),'lv':w.get('level',''),'ipa':w.get('ipa',''),'g':w.get('gender',''),
                  'p':(w['rank']-1)//100+1,'cj':bool((w.get('conjugation') or {}).get('present')),
                  'ex':bool((w.get('examples') or []) or (w.get('learning_extension') or {}).get('extra_examples')),
                  'fq':fp.get('frequency_per_million',0),'ln':pr.get('length',len(w['fr'])),'old':pr.get('old20')})
# Improve distractors using actual corpus frequency and word length while retaining POS/CEFR/Levenshtein.
s=re.sub(r"function distractors\(target,n=3,field='f'\)\{.*?return out\}",r'''function distractors(target,n=3,field='f'){let t=BYR.get(target.rank)||{f:target.fr,pos:target.pos,lv:target.level,r:target.rank,z:target.meaning?.zh,fq:target.frequency_v4?.frequency_per_million||0,ln:(target.fr||'').length};let pool=INDEX.filter(x=>x.r!==t.r&&x.pos===t.pos);let flog=x=>Math.log1p(Math.max(0,+x||0));pool.sort((a,b)=>{let score=x=>(x.lv===t.lv?0:2.5)+Math.min(lev(x.f,t.f),6)*1.15+Math.abs((x.ln||x.f.length)-(t.ln||t.f.length))*.35+Math.abs(flog(x.fq)-flog(t.fq))*.45+Math.abs(x.r-t.r)/600;return score(a)-score(b)});let seen=new Set([String(t[field]||'')]),out=[];for(const x of shuffle(pool.slice(0,Math.max(60,n*15)))){let v=String(x[field]||'');if(!v||seen.has(v))continue;seen.add(v);out.push(x);if(out.length>=n)break}return out}''',s,flags=re.S)
# Version text leftovers.
s=s.replace('Cards V3','Cards V4').replace('V3 懒加载版','V4 懒加载版')
# No external/network additions allowed.
for bad in ['https://','http://','fetch(','Wiktion','CNRTL']:
    if bad in s: raise RuntimeError('forbidden in HTML: '+bad)

# Write lazy data.
meta_js={'name':'French Core 3000','version':'V4 merged local','count':len(W)}
(LAZY/'data'/'search_index.js').write_text('window.FC_META='+json.dumps(meta_js,ensure_ascii=False,separators=(',',':'))+';\nwindow.FC_INDEX='+json.dumps(index,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
for p in range(1,31):
    chunk=W[(p-1)*100:p*100]
    (LAZY/'data'/f'pack_{p:03d}.js').write_text('window.FC_PACKAGES=window.FC_PACKAGES||{};\n'+f'window.FC_PACKAGES["{p:03d}"]='+json.dumps(chunk,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
(LAZY/'index.html').write_text(s,encoding='utf-8')
(LAZY/'README.txt').write_text('FrenchCards V4 懒加载版。3000 词分为 30 个本地数据包；已融合 MorphoLex-FR、Lexique 4 与 Cobb 2026 的学习相关字段。页面不访问外部网络。\n',encoding='utf-8')
# Standalone with all words inline.
idx=(LAZY/'data'/'search_index.js').read_text(encoding='utf-8')
stand=s.replace('<script src="data/search_index.js"></script>','<script>'+idx+'</script>\n<script>window.FC_ALL_WORDS='+json.dumps(W,ensure_ascii=False,separators=(',',':'))+';</script>')
STAND.write_text(stand,encoding='utf-8')
# Data
(OUT/'French_Core_3000_V4_AppData.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
# source extracts used for reproducibility
shutil.copy2(WORK/'cobb_list.json',OUT/'tools'/'source_extracts'/'cobb_list.json')
shutil.copy2(WORK/'cobb_complete.json',OUT/'tools'/'source_extracts'/'cobb_complete.json')
shutil.copy2(Path(__file__),OUT/'tools'/'build_v4.py')
# Import report
report={
 'base':'FrenchCards V3 Current Checked','words':len(W),
 'uploaded_resources':{
   'MorphoLex-FR':{'exact_core_matches':sum(norm(w['fr']) in morph for w in W),'multi_morpheme_core_matches':sum(bool(morpholex_struct(morph.get(norm(w['fr'])))) for w in W)},
   'Lexique4':{'exact_word_matches':sum(norm(w['fr']) in lex_by_word for w in W),'exact_lemma_matches':sum(norm(w['fr']) in lex_by_lemma for w in W),'profiles_merged':counts['lexique_profile']},
   'Cobb2026':{'core_word_matches':sum(norm(w['fr']) in comp_by_word for w in W),'families_merged':counts['cobb_family']}
 },
 'merged_counts':dict(counts),
 'learner_facing_policy':[
   'Existing V3 curated definitions/examples/conjugations are preserved.',
   'MorphoLex canonical morphology expands morphology coverage; V3 curated morphology remains preferred where available.',
   'Lexique 4 contributes compact modern frequency/syllable data and practice-ranking features; it does not overwrite curated definitions or conjugations.',
   'Lexique 4 also adds standard noun/adjective inflection tables where reliable.',
   'Cobb 2026 contributes word-family/common-form groupings.',
   'No external links or runtime network fetches are added.'
 ]
}
(OUT/'V4_导入报告.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README_测试说明.txt').write_text(f'''FrenchCards V4 · 本地融合版\n\n最快测试：直接打开 FrenchCards_V4_Standalone.html。\n网页部署：将 FrenchCards_V4_Lazy 整个目录放到静态服务器。\n\n本版在 V3 基础上正式融合你上传的三份数据：\n- MorphoLex-FR：构词覆盖扩展至 {counts['morph_total']} 个词（其中 {counts['morph_new_from_morpholex']} 个是新加入的数据库构词分析）。\n- Lexique 4：{counts['lexique_profile']} 个词加入现代词频/音节与练习辅助特征，并为 {counts['lexique_inflections']} 个名词/形容词加入标准词形；四选一干扰项同时参考词频和长度。\n- Cobb 2026：{counts['cobb_family']} 个词加入词族映射，其中 {counts['cobb_with_related']} 个有可展示的相关词形。\n\n原有 3000 词、释义、例句、词组、练习、法语 TTS、本地懒加载与无系统键盘逻辑均保留。\n''',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
