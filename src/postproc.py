import json, sys, re, difflib
mode, inf, srcf, outf = sys.argv[1:5]
src = {o['sample_id']: (o.get('full_text') or '') for o in map(json.loads, open(srcf, encoding='utf-8'))}
def sents(t):
    return [s.strip() for s in re.split(r'(?<=[。！？；\n])', t) if len(s.strip()) >= 4]
KW = ['支持','赞扬','赞赏','欢迎','反对','批评','谴责','抵制','不满','拒绝']
out = []
for line in open(inf, encoding='utf-8'):
    o = json.loads(line)
    il = o.get('issue_list') or []; fu = list(o.get('future_argument') or [])
    ft = src.get(o['sample_id'], '')
    if mode == 'rescue':
        os_ = sents(ft)
        for x in il:
            nc = []
            for c in (x.get('argument_chain') or []):
                best, br = c, 0.0
                for s in os_:
                    r = difflib.SequenceMatcher(None, c, s).ratio()
                    if r > br: best, br = s, r
                nc.append(best if br >= 0.85 else c)
            x['argument_chain'] = nc
    elif mode == 'trim':
        for x in il:
            x['argument_chain'] = (x.get('argument_chain') or [])[:3]
    elif mode == 'demote':
        for x in il:
            if x.get('stance') in ('support', 'oppose'):
                blob = (x.get('issue_name') or '') + ''.join(x.get('argument_chain') or [])
                if not any(k in blob for k in KW):
                    x['stance'] = 'neutral'
    elif mode == 'dedup':
        keep = []
        for idx, x in enumerate(il):
            hit = None
            for y in keep:
                if difflib.SequenceMatcher(None, x.get('issue_name',''), y.get('issue_name','')).ratio() > 0.9:
                    hit = y; break
            if hit is not None:
                for c in (x.get('argument_chain') or []):
                    if c not in hit['argument_chain']: hit['argument_chain'].append(c)
                hit['_fu'].append(fu[idx] if idx < len(fu) else '')
            else:
                x['_fu'] = [fu[idx] if idx < len(fu) else '']
                keep.append(x)
        il = keep
        fu = []
        for y in il:
            f = y.pop('_fu')
            fu.append(max(f, key=len) if f else '')
    while len(fu) < len(il): fu.append('')
    fu = fu[:len(il)]
    for x in il:
        x['issue_name'] = str(x.get('issue_name','')).replace('\n','')
        x['argument_chain'] = [str(c).replace('\n','') for c in (x.get('argument_chain') or [])]
    o['issue_list'] = il; o['future_argument'] = [str(t).replace('\n','') for t in fu]
    out.append(o)
with open(outf, 'w', encoding='utf-8') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + '\n')
print(mode, 'rows', len(out))
