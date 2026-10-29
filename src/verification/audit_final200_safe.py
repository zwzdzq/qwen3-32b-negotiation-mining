import pathlib,tarfile,json,hashlib,re,runpy,ast,copy,shutil
P=pathlib.Path(__file__).resolve().parent;NAME='candidate_37_47_safe_test200';archive=P/(NAME+'.tar.gz')
with tarfile.open(archive) as t:d={m.name.removeprefix(NAME+'/'):t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
hs=json.loads(d['sha256.json'])
for n,h in hs.items():assert hashlib.sha256(d[n]).hexdigest()==h,n
j=lambda n:json.loads(d['run/'+n]);rows=lambda n:[json.loads(x) for x in d['run/'+n].splitlines() if x.strip()]
assert j('status.json')['stage']=='complete'
assert hashlib.sha256(d['run/source24.jsonl']).hexdigest()=='1dd312e0a345c594d30383474531e53ef44e701ba34631bd61b8f8741d3c5088'
assert hashlib.sha256(d['run/input.jsonl']).hexdigest()=='cd9294679e772ed217e00eb1ab3a7354e9c7f6ab7e8c54d9892949dafb5ce1ac'
for local,remote in [('coverage37_safe.py','phase37/worker.py'),('future47_safe.py','phase47/worker.py')]:
 expected=(P/local).read_text(encoding='utf-8').replace('B=R.parent.parent',"B=pathlib.Path('/root/autodl-tmp/semifinal-20261015-codex')")
 assert d['run/'+remote].decode().replace('\r\n','\n')==expected
for n,h in j('manifest.json')['files'].items():assert hashlib.sha256(d['run/'+n]).hexdigest()==h
original=rows('source24.jsonl');inputs=rows('input.jsonl');base=rows('phase37/baseline.jsonl');extracted=rows('phase37/candidate.jsonl');out=rows('result.jsonl')
assert len(original)==len(inputs)==len(base)==len(extracted)==len(out)==200
ids=[x['sample_id'] for x in original];assert len(set(ids))==200
for group in [inputs,base,extracted,out]:assert [x['sample_id'] for x in group]==ids
assert d['run/input.jsonl']==d['run/phase37/input.jsonl']==d['run/phase47/input.jsonl']
pat=re.compile(r'\s*【参与方[^】]*】\s*');removed=[]
for a,b in zip(original,base):
 keep=[]
 for i,it in enumerate(a['issue_list']):
  if it['argument_chain'] and all(pat.fullmatch(c) for c in it['argument_chain']):removed.append([a['sample_id'],i])
  else:keep.append(i)
 expected=copy.deepcopy(a);expected['issue_list']=[a['issue_list'][i] for i in keep];expected['future_argument']=[a['future_argument'][i] for i in keep];assert expected==b
assert removed==j('phase30_provenance.json')['removed']
m37=runpy.run_path(str(P/'coverage37_safe.py'));m47=runpy.run_path(str(P/'future47_safe.py'))
def promptcode(file):
 tree=ast.parse((P/file).read_text(encoding='utf-8'));expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='prompt' for x in n.targets));return compile(ast.Expression(expr),'<prompt>','eval')
c37=promptcode('coverage37_safe.py');c47=promptcode('future47_safe.py')
fb37=j('phase37/fallbacks.json');fl37={(r['sample_id'],r['target_index']):r for r in fb37}
tr37=rows('phase37/components.jsonl');refs37=j('phase37/provenance.json');lookup={(r['sample_id'],r['target_index']):r for r in refs37};keys=set()
for old,new,inp in zip(base,extracted,inputs):
 sid=old['sample_id'];expected=copy.deepcopy(old);full='\n\n'.join(x['full_text'] for x in inp['docs'])
 for ti,target in enumerate(m37['targets'](full,old)):
  key=(sid,ti);keys.add(key)
  if key in fl37:
   ref=fl37[key];assert ref['target']==target and len(ref['trace_indices'])==3
   for ix in ref['trace_indices']:
    tr=tr37[ix];assert (tr['sample_id'],tr['target_index'])==key
    prompt=eval(c37,{'json':json,'target':target,'new':expected});assert hashlib.sha256(prompt.encode()).hexdigest()==tr['prompt_sha256']
    try:
     assert tr['ended_with_eos'] and not tr['hit_token_limit'];m37['parse'](tr['decoded'],target,expected)
    except (AssertionError,ValueError,TypeError,KeyError):pass
    else:raise AssertionError('invalid fallback: acceptable attempt')
   continue
  ref=lookup[key];assert target==ref['target'];tr=tr37[ref['trace_index']]
  assert (tr['sample_id'],tr['target_index'])==key and tr['ended_with_eos'] and not tr['hit_token_limit'] and tr['raw']==tr['decoded']+'<|im_end|>'
  prompt=eval(c37,{'json':json,'target':target,'new':expected});assert hashlib.sha256(prompt.encode()).hexdigest()==tr['prompt_sha256']
  issues,futures=m37['parse'](tr['decoded'],target,expected);assert ref['start']==len(expected['issue_list']) and ref['added']==len(issues)
  expected['issue_list']+=issues;expected['future_argument']+=futures
 assert expected==new
assert not(set(lookup)&set(fl37)) and keys==set(lookup)|set(fl37) and len(lookup)==len(refs37)
assert rows('phase47/baseline.jsonl')==extracted and rows('phase47/precoverage.jsonl')==base and j('phase47/source37_provenance.json')==refs37
assert rows('phase47/candidate.jsonl')==out
fb47=j('phase47/fallbacks.json');fl47={(r['sample_id'],r['index']):r for r in fb47}
targets=m47['added_targets'](extracted,base);refs47=j('phase47/provenance.json');lookup47={(r['sample_id'],r['index']):r for r in refs47};tr47=rows('phase47/components.jsonl');assert targets==set(lookup47)|set(fl47) and not(set(lookup47)&set(fl47)) and len(refs47)==len(lookup47)
assert targets=={tuple(x) for x in j('phase47/manifest.json')['targets']}
for a,b in zip(extracted,out):
 assert {k:v for k,v in a.items() if k!='future_argument'}=={k:v for k,v in b.items() if k!='future_argument'}
 assert len(a['future_argument'])==len(b['future_argument'])==len(b['issue_list'])
 for i,(before,after) in enumerate(zip(a['future_argument'],b['future_argument'])):
  key=(a['sample_id'],i)
  if key not in targets:assert before==after;continue
  if key in fl47:
   ref=fl47[key];assert before==after and len(ref['trace_indices'])==3
   country=next(r['target']['country'] for r in refs37 if r['sample_id']==key[0] and r['start']<=i<r['start']+r['added']);issue=a['issue_list'][i]
   for ix in ref['trace_indices']:
    tr=tr47[ix];assert (tr['sample_id'],tr['index'])==key
    prompt=eval(c47,{'json':json,'country':country,'issue':issue});assert hashlib.sha256(prompt.encode()).hexdigest()==tr['prompt_sha256']
    try:
     assert tr['ended_with_eos'] and not tr['hit_token_limit'];m47['parse_choice'](tr['decoded'],issue,country)
    except (AssertionError,ValueError,TypeError,KeyError):pass
    else:raise AssertionError('invalid selection fallback')
   continue
  ref=lookup47[key];tr=tr47[ref['trace_index']];assert (tr['sample_id'],tr['index'])==key and tr['ended_with_eos'] and not tr['hit_token_limit'] and tr['raw']==tr['decoded']+'<|im_end|>'
  country=next(r['target']['country'] for r in refs37 if r['sample_id']==key[0] and r['start']<=i<r['start']+r['added']);issue=a['issue_list'][i]
  prompt=eval(c47,{'json':json,'country':country,'issue':issue});assert hashlib.sha256(prompt.encode()).hexdigest()==tr['prompt_sha256']
  assert m47['parse_choice'](tr['decoded'],issue,country)==after
assert not d['run/result.jsonl'].startswith(b'\xef\xbb\xbf');d['run/result.jsonl'].decode('utf-8')
for a,b in zip(original,out):
 assert set(a)==set(b) and b['issue_list'] and len(b['issue_list'])==len(b['future_argument'])
 assert len({json.dumps(x,sort_keys=True) for x in b['issue_list']})==len(b['issue_list'])
 for it,f in zip(b['issue_list'],b['future_argument']):
  assert set(it)=={'issue_name','stance','argument_chain'} and it['stance'] in ['support','oppose','neutral']
  assert isinstance(it['issue_name'],str) and it['issue_name'].strip() and it['argument_chain'] and all(isinstance(v,str) and v.strip() for v in it['argument_chain'])
  assert isinstance(f,str) and f.strip() and not pat.fullmatch(f)
r=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files_verified=len(hs),phase37_fallbacks=len(fb37),phase47_fallbacks=len(fb47),rows=200,issues=sum(len(x['issue_list']) for x in out),removed=len(removed),speaker_calls=len(refs37),added=len(targets),phase37_attempts=len(tr37),phase47_attempts=len(tr47),phase37_seconds=sum(x['seconds'] for x in tr37),phase47_seconds=sum(x['seconds'] for x in tr47),all_raw_eos_prompt_sha_and_source_mapping_verified=True,all_other_fields_frozen=True,result_sha256=hashlib.sha256(d['run/result.jsonl']).hexdigest(),no_test_gold_or_score=True,status=j('status.json'))
assert r['result_sha256']==j('validation.json')['result_sha256']
(P/(NAME+'_local_audit.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
O=P.parent/'outputs/复赛提交/candidate_37_47_safe';O.mkdir(exist_ok=False)
for n in ['result.jsonl','manifest.json','validation.json']:(O/n).write_bytes(d['run/'+n])
(O/'local_validation.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'provenance.json').write_text(json.dumps({'phase30':j('phase30_provenance.json'),'phase37':refs37,'phase47':refs47,'phase37_fallbacks':fb37,'phase47_fallbacks':fb47},ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(archive,O/archive.name)
print(json.dumps(r,ensure_ascii=False))
