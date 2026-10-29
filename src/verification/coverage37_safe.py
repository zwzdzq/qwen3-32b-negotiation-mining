import pathlib,json,re,time,sys,os,hashlib,subprocess,traceback,tarfile,shutil
R=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/root/autodl-tmp/semifinal-20261015-codex')
HEADER=r'(?m)^([\w·.\- ]{1,55}(?:先生|女士|女爵士|爵士|大使|博士))[（(]([^\n（）()：:]{2,20})[）)]\s*[：:]'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def targets(full,old):
 represented=set();allchains=[v for x in old['issue_list'] for v in x['argument_chain']]
 for v in allchains:
  m=re.fullmatch(r'【参与方[^：】]+：([^｜】]+)(?:｜[^】]*)?】',v)
  if m:represented.add(m.group(1).strip())
 if not represented:return []
 heads=list(re.finditer(HEADER,full));bounds=sorted([m.start() for m in heads]+[m.start() for m in re.finditer(r'【参与方[^】]*】',full)])
 out=[]
 for h in heads:
  if h.group(2).strip() in represented or any(h.group(0) in v for v in allchains):continue
  end=min([v for v in bounds if v>=h.end()],default=len(full));section=re.sub(r'\s+',' ',full[h.end():end]).strip()
  if not section:continue
  spans=[x for x in re.split(r'(?<=[。！？])',section) if x];assert ''.join(spans)==section
  out.append(dict(heading=h.group(0),country=h.group(2).strip(),start=h.start(),end=end,spans=spans))
 return out
def parse(text,target,old):
 text=text.strip()
 if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
 obj=json.loads(text);assert isinstance(obj,dict) and set(obj)=={'items'} and isinstance(obj['items'],list)
 issues=[];future=[]
 for item in obj['items']:
  assert isinstance(item,dict) and set(item)=={'issue_name','stance','evidence','future_argument'}
  assert isinstance(item['issue_name'],str) and item['issue_name'].strip()
  assert item['stance'] in ['support','oppose','neutral']
  indices=item['evidence'];assert isinstance(indices,list) and 1<=len(indices)<=3
  assert all(type(i)==int and 0<=i<len(target['spans']) for i in indices) and len(indices)==len(set(indices))
  f=item['future_argument'];assert isinstance(f,str) and f.strip() and not re.fullmatch(r'\s*【[^】]*】\s*',f) and f.strip() not in [target['country'],target['heading']]
  issues.append(dict(issue_name=item['issue_name'],stance=item['stance'],argument_chain=[target['heading']]+[target['spans'][i] for i in sorted(indices)]));future.append(f)
 allissues=old['issue_list']+issues;assert len({json.dumps(x,sort_keys=True) for x in allissues})==len(allissues)
 return issues,future
def worker():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.manual_seed(42);adapter=B/'runs/candidate_02_train/adapter'
 assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()=='45250b13aa1cc02e5b25789ccad242bf7bebc222c61f393093e5d49c81756d8a'
 tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen/Qwen3-32B',local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(str(adapter),device_map={'':0},torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True);model.eval();assert model.peft_config
 save('effective_generation_config.json',model.generation_config.to_dict());eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 replay={}
 if (R/'resume_components.jsonl').exists():
  for t in rows(R/'resume_components.jsonl'):replay.setdefault((t['sample_id'],t['target_index']),[]).append(t)
 fallbacks=[]
 base=rows(R/'baseline.jsonl');inputs={x['sample_id']:x for x in rows(R/'input.jsonl')};traces=[];refs=[];out=[]
 for old in base:
  sid=old['sample_id'];full='\n\n'.join(d['full_text'] for d in inputs[sid]['docs']);new=json.loads(json.dumps(old))
  for ti,target in enumerate(targets(full,old)):
   section=''.join(target['spans']);assert len(section)<=12000
   prompt=('只从下面这一位发言方的完整发言中抽取有实质表态的议题，并预测该方下一步可能提出的观点。该发言标题未被现有参与方标记覆盖。先核对现有条目，若该方实际上已经被覆盖则不要重复；本次不能改动现有条目。只抽取当前发言方，不替其引用或批评的其他国家生成条目。区分具体命题的肯定、否定与中立，不因提到某议题就凑条目。不同的真实子议题可独立列出，不用同义名称重复同一命题。每条用1至3个编号引用本段直接论据；不要改写证据。future_argument必须是该方具有实际主张的完整中文句子。没有新增内容则items为空数组。只输出JSON对象，格式为{"items":[{"issue_name":"具体议题","stance":"support或oppose或neutral","evidence":[0],"future_argument":"完整观点句"}]}。\n发言标题：'+target['heading']+'\n该方完整发言（句子编号从0开始）：'+json.dumps(list(enumerate(target['spans'])),ensure_ascii=False)+'\n现有全部抽取（仅供避免重复，保持不变）：'+json.dumps([{k:v for k,v in x.items() if k!='stance'} for x in new['issue_list']],ensure_ascii=False))
   rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False);inp=tok(rendered,return_tensors='pt',add_special_tokens=False).to(model.device);assert inp.input_ids.shape[1]<=16384
   chosen=None
   previous=replay.get((sid,ti))
   if previous:
    assert len(previous)<=3 and [t['attempt'] for t in previous]==list(range(len(previous)))
    for tr in previous:
     assert tr['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
     try:assert tr['ended_with_eos'] and not tr['hit_token_limit'];chosen=parse(tr['decoded'],target,new)
     except (AssertionError,ValueError,TypeError,KeyError):chosen=None
     traces.append(tr)
     with (R/'components.jsonl').open('a') as f:f.write(json.dumps(tr,ensure_ascii=False)+'\n')
     if chosen is not None:break
    assert chosen is not None or len(previous)==3
   else:
    for attempt in range(3):
     kw=dict(max_new_tokens=2048,do_sample=attempt>0,pad_token_id=tok.eos_token_id)
     if attempt:kw.update(temperature=.3,top_p=.9)
     st=time.time()
     with torch.no_grad():gen=model.generate(**inp,**kw)[0,inp.input_ids.shape[1]:]
     ids=gen.tolist();decoded=tok.decode(ids,skip_special_tokens=True)
     tr=dict(sample_id=sid,target_index=ti,attempt=attempt,seconds=time.time()-st,input_tokens=inp.input_ids.shape[1],new_tokens=len(ids),ended_with_eos=bool(ids) and ids[-1] in eos,hit_token_limit=len(ids)>=2048 and ids[-1] not in eos,decoded=decoded,raw=tok.decode(ids,skip_special_tokens=False),generation_kwargs=kw,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
     try:assert tr['ended_with_eos'] and not tr['hit_token_limit'];chosen=parse(decoded,target,new)
     except Exception as e:tr['rejection']=type(e).__name__;chosen=None
     traces.append(tr)
     with (R/'components.jsonl').open('a') as f:f.write(json.dumps(tr,ensure_ascii=False)+'\n')
     if chosen is not None:break
   if chosen is None:
    fallbacks.append(dict(sample_id=sid,target_index=ti,target=target,reason='all3existingstrictprotocolattemptsfailed; preserveoriginal',trace_indices=list(range(len(traces)-3,len(traces)))))
    save('fallbacks.json',fallbacks);save('progress.json',dict(completed=len(refs)+len(fallbacks),accepted=len(refs),fallbacks=len(fallbacks),attempts=len(traces),seconds=sum(t['seconds'] for t in traces)));continue
   start=len(new['issue_list']);new['issue_list']+=chosen[0];new['future_argument']+=chosen[1];refs.append(dict(sample_id=sid,target_index=ti,trace_index=len(traces)-1,start=start,added=len(chosen[0]),target=target))
   save('provenance.json',refs);save('progress.json',dict(completed=len(refs),attempts=len(traces),seconds=sum(t['seconds'] for t in traces)))
  assert new['issue_list'][:len(old['issue_list'])]==old['issue_list'] and new['future_argument'][:len(old['future_argument'])]==old['future_argument'];out.append(new)
 (R/'candidate.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 save('fallbacks.json',fallbacks)
 save('validation.json',dict(fallbacks=len(fallbacks),rows=len(out),calls=len(refs),attempts=len(traces),seconds=sum(t['seconds'] for t in traces),added=sum(r['added'] for r in refs),all_original_prefixes_unchanged=True))
def main():
 if '--worker' in sys.argv:worker();return
 start=time.time();env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
 try:
  save('status.json',dict(stage='inference_running',started=start))
  with (R/'inference.log').open('x') as f:
   p=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],stdout=f,stderr=subprocess.STDOUT,env=env);(R/'active_child.pid').write_text(str(p.pid))
   try:rc=p.wait(timeout=1200)
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
