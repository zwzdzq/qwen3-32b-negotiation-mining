"""Validation-only repair of label-only futures; preserve all other predictions."""
import hashlib,json,os,pathlib,re,shutil,subprocess,sys,tarfile,time,traceback
R=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/root/autodl-tmp/semifinal-20261015-codex')
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def marker_only(s):return bool(re.fullmatch(r'\s*【参与方[^】]*】\s*',s))
def clean(s):return re.sub(r'\s+',' ',s).strip()
def valid_future(s):
 return bool(s.strip()) and not marker_only(s) and s.strip() not in {'无','待预测'} and not s.lstrip().startswith(('{','[','```')) and all(t not in s for t in ['<think>','</think>','<|'])
def save(name,data):
 p=R/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def added_targets(base,prior):
 assert len(base)==len(prior)
 targets=set()
 for a,b in zip(prior,base):
  assert a['sample_id']==b['sample_id']
  assert a['issue_list']==b['issue_list'][:len(a['issue_list'])]
  assert a['future_argument']==b['future_argument'][:len(a['future_argument'])]
  targets.update((b['sample_id'],i) for i in range(len(a['issue_list']),len(b['issue_list'])))
 return targets

def speaker_sources(base,inputs,provenance):
 sources={}
 for ref in provenance:
  sid=ref['sample_id'];t=ref['target'];raw='\n\n'.join(x['full_text'] for x in inputs[sid]['docs'])
  section=raw[t['start']:t['end']]
  assert section.startswith(t['heading'])
  assert clean(section[len(t['heading']):])==''.join(t['spans'])
  for i in range(ref['start'],ref['start']+ref['added']):sources[(sid,i)]=clean(section)
 return sources

def parse_choice(decoded,issue,country):
 text=decoded.strip()
 if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
 obj=json.loads(text)
 assert isinstance(obj,dict) and set(obj)=={'evidence'}
 i=obj['evidence'];assert type(i)==int and 1<=i<len(issue['argument_chain'])
 clause=issue['argument_chain'][i];assert clause.strip() and not marker_only(clause)
 return country+'可能继续强调：'+clause

def worker():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.manual_seed(42)
 assert hashlib.sha256((B/'runs/candidate_02_train/adapter/adapter_model.safetensors').read_bytes()).hexdigest()=='45250b13aa1cc02e5b25789ccad242bf7bebc222c61f393093e5d49c81756d8a'
 tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen/Qwen3-32B',local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(str(B/'runs/candidate_02_train/adapter'),device_map={'':0},torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 model.eval();assert bool(getattr(model,'peft_config',None))
 save('effective_generation_config.json',model.generation_config.to_dict())
 eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 fallbacks=[]
 base=rows(R/'baseline.jsonl');out=json.loads(json.dumps(base));inputs={r['sample_id']:r for r in rows(R/'input.jsonl')};refs=[];traces=[]
 manifest=json.loads((R/'manifest.json').read_text());expected_targets={tuple(x) for x in manifest['targets']};expected_rows=manifest['expected_rows']
 actual_targets=added_targets(base,rows(R/'precoverage.jsonl'))
 assert actual_targets==expected_targets and len(base)==len({x['sample_id'] for x in base})==expected_rows
 assert {x['sample_id'] for x in base}==set(inputs)
 sources=speaker_sources(base,inputs,json.loads((R/'source37_provenance.json').read_text()))
 assert set(sources)==expected_targets
 for row in out:
  sid=row['sample_id'];full='\n\n'.join(clean(x['full_text']) for x in inputs[sid]['docs']);assert len(full)<=12000
  for index,old in enumerate(row['future_argument']):
   if (sid,index) not in expected_targets:continue
   issue=row['issue_list'][index]
   full=sources[(sid,index)]
   country=next(x['target']['country'] for x in json.loads((R/'source37_provenance.json').read_text()) if x['sample_id']==sid and x['start']<=index<x['start']+x['added'])
   prompt=('为当前发言方预测下一轮最可能重申的一项已有核心观点。仅从该议题给定的原文论据中选择一句最能代表其核心立场的实质句子，不选择发言标题，不改写，不补充新行动，不依据其他方观点。只输出JSON对象，格式为{"evidence":1}，evidence是论据数组中的整数索引（0是标题，不可选）。\n发言方：'+country+'\n议题及原文论据：'+json.dumps({**issue,'argument_chain':[{'index':i,'text':v} for i,v in enumerate(issue['argument_chain'])]},ensure_ascii=False))
   rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   inp=tok(rendered,return_tensors='pt',add_special_tokens=False).to(model.device);assert inp.input_ids.shape[1]<=16384;chosen=None
   for attempt in range(3):
    kwargs=dict(max_new_tokens=384,do_sample=attempt>0,pad_token_id=tok.eos_token_id)
    if attempt:kwargs.update(temperature=.3,top_p=.9)
    st=time.time()
    with torch.no_grad():generated=model.generate(**inp,**kwargs)[0,inp.input_ids.shape[1]:]
    ids=generated.tolist();decoded=tok.decode(ids,skip_special_tokens=True)
    d=dict(sample_id=sid,index=index,attempt=attempt,input_tokens=inp.input_ids.shape[1],new_tokens=len(ids),ended_with_eos=bool(ids) and ids[-1] in eos,hit_token_limit=len(ids)>=384 and ids[-1] not in eos,seconds=time.time()-st,generation_kwargs=kwargs,decoded=decoded,raw=tok.decode(ids,skip_special_tokens=False),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    traces.append(d)
    with (R/'components.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    try:
     assert d['ended_with_eos'] and not d['hit_token_limit']
     prediction=parse_choice(decoded,issue,country)
    except (AssertionError,ValueError,TypeError,KeyError):continue
    chosen=(prediction,len(traces)-1);break
   if chosen is None:
    fallbacks.append(dict(sample_id=sid,index=index,trace_indices=list(range(len(traces)-3,len(traces))),reason='all3strictselectionattemptsfailed;preserve37future'));save('fallbacks.json',fallbacks);continue
   row['future_argument'][index]=chosen[0];refs.append(dict(sample_id=sid,index=index,trace_index=chosen[1]))
   save('progress.json',dict(completed_components=len(refs),expected_components=len(expected_targets),sample_id=sid,index=index))
   save('partial_predictions.json',out);save('provenance.json',refs)
 manifest=json.loads((R/'manifest.json').read_text());assert hashlib.sha256((R/'baseline.jsonl').read_bytes()).hexdigest()==manifest['baseline_sha256']
 assert len(base)==len(out)==expected_rows and len(refs)+len(fallbacks)==len(expected_targets)
 fallbackkeys={(x['sample_id'],x['index']) for x in fallbacks}
 assert {(x['sample_id'],x['index']) for x in refs}|fallbackkeys==expected_targets
 lookup={(x['sample_id'],x['index']):x for x in refs}
 for old,new in zip(base,out):
  assert old['sample_id']==new['sample_id'] and old['issue_list']==new['issue_list']
  assert len(old['future_argument'])==len(new['future_argument'])==len(new['issue_list'])
  for index,(before,after) in enumerate(zip(old['future_argument'],new['future_argument'])):
   if (new['sample_id'],index) not in expected_targets or (new['sample_id'],index) in fallbackkeys:assert before==after;continue
   ref=lookup[(new['sample_id'],index)];d=traces[ref['trace_index']]
   assert d['sample_id']==new['sample_id'] and d['index']==index and d['ended_with_eos'] and not d['hit_token_limit'] and parse_choice(d['decoded'],new['issue_list'][index],next(x['target']['country'] for x in json.loads((R/'source37_provenance.json').read_text()) if x['sample_id']==new['sample_id'] and x['start']<=index<x['start']+x['added']))==after
 with (R/'candidate.jsonl').open('x',encoding='utf-8',newline='\n') as f:
  for row in out:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 save('fallbacks.json',fallbacks)
 save('validation.json',dict(fallbacks=len(fallbacks),rows=expected_rows,components_repaired=len(refs),issues_unchanged=True,other_futures_unchanged=True,component_eos=len(refs),component_limit_hits=0,component_attempts=len(traces),component_generation_seconds=sum(x['seconds'] for x in traces),all_components_audited=True,result_sha256=hashlib.sha256((R/'candidate.jsonl').read_bytes()).hexdigest()))
def main():
 if '--worker' in sys.argv:worker();return
 start=time.time();env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
 try:
  save('status.json',dict(stage='inference_running',started=start))
  with (R/'inference.log').open('x') as f:
   p=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],stdout=f,stderr=subprocess.STDOUT,env=env);(R/'active_child.pid').write_text(str(p.pid))
   try:rc=p.wait(timeout=300)
   except subprocess.TimeoutExpired:p.kill();p.wait();raise RuntimeError('inference timeout')
   assert rc==0
  save('status.json',dict(stage='scoring',started=start))
  for name in ['baseline','candidate']:
   with (R/(name+'.log')).open('x') as f:subprocess.run([sys.executable,str(R/'score.py'),name],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=300,check=True)
  save('status.json',dict(stage='complete',started=start,ended=time.time()))
 except Exception:save('status.json',dict(stage='failed',started=start,error=traceback.format_exc()))
 finally:
  state=json.loads((R/'status.json').read_text());D=B/'backups'/(R.name+('' if state['stage']=='complete' else '_failed'));shutil.copytree(R,D/'run')
  hs={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file()};(D/'sha256.json').write_text(json.dumps(hs,indent=2))
  with tarfile.open(str(D)+'.tar.gz','x:gz') as t:t.add(D,arcname=D.name)
  save('backup.json',dict(archive=str(D)+'.tar.gz',sha256=hashlib.sha256(pathlib.Path(str(D)+'.tar.gz').read_bytes()).hexdigest()))
if __name__=='__main__':main()
