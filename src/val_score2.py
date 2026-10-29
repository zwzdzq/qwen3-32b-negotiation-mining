import os, sys, json
import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import BERTScorer
import bert_score.utils as _bu

BGE = os.environ.get('BGE_PATH', 'BAAI/bge-small-zh-v1.5')
BERT = os.environ.get('BERT_PATH', 'bert-base-chinese')
_bu.model2layers.setdefault(BERT, _bu.model2layers['bert-base-chinese'])
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GOLD = os.environ.get('GOLD_FILE', 'val.jsonl')

tok = AutoTokenizer.from_pretrained(BGE)
mdl = AutoModel.from_pretrained(BGE).to(DEV).eval()

def emb(texts, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = texts[i:i+bs]
            enc = tok(b, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEV)
            h = mdl(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            v = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, mdl.config.hidden_size)

scorer = BERTScorer(model_type=BERT, lang='zh', rescale_with_baseline=False, device=DEV)
btok = AutoTokenizer.from_pretrained(BERT)
def T(s, n=500):
    ids = btok.encode(str(s), add_special_tokens=False)[:n]
    return btok.decode(ids, skip_special_tokens=True)

def rougeL_f1(p, g):
    P, G = list(p), list(g)
    if not P or not G: return 0.0
    m = [[0]*(len(G)+1) for _ in range(len(P)+1)]
    for i in range(1, len(P)+1):
        for j in range(1, len(G)+1):
            m[i][j] = m[i-1][j-1]+1 if P[i-1] == G[j-1] else max(m[i-1][j], m[i][j-1])
    l = m[-1][-1]
    if l == 0: return 0.0
    pr, rc = l/len(P), l/len(G)
    return 2*pr*rc/(pr+rc)

def load(fp):
    d = {}
    for line in open(fp, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        o = json.loads(line); d[o['sample_id']] = o
    return d

gold, pred = load(GOLD), load(sys.argv[1])
sids = [s for s in gold if s in pred]
print('samples', len(sids))

ext_num = 0.0; alphas = []; f_prd = []
for sid in sids:
    gi = gold[sid].get('issue_list') or []; pi = pred[sid].get('issue_list') or []
    gf = gold[sid].get('future_argument') or []; pf = pred[sid].get('future_argument') or []
    gt = [f"{x.get('issue_name','')} {' '.join(x.get('argument_chain') or [])}" for x in gi]
    pt = [f"{x.get('issue_name','')} {' '.join(x.get('argument_chain') or [])}" for x in pi]
    n = max(len(gf), len(pf)) or 1
    rs = [rougeL_f1(str(pf[i]) if i < len(pf) else '', str(gf[i]) if i < len(gf) else '') for i in range(n)]
    _, _, bf = scorer.score([T(str(pf[i]) if i < len(pf) else '') for i in range(n)],
                            [T(str(gf[i]) if i < len(gf) else '') for i in range(n)])
    bl = bf.tolist()
    f_prd.append(0.5*(sum(rs)/n + sum(bl)/n))
    if not gt or not pt:
        continue
    gn, pn = [x.get('issue_name','') for x in gi], [x.get('issue_name','') for x in pi]
    gc = [' '.join(x.get('argument_chain') or []) for x in gi]
    pc = [' '.join(x.get('argument_chain') or []) for x in pi]
    GN, PN, GC, PC = emb(gn), emb(pn), emb(gc), emb(pc)
    S = 0.4*(GN@PN.T) + 0.6*(GC@PC.T)
    stg, stp = [x.get('stance') for x in gi], [x.get('stance') for x in pi]
    cand = []
    for a in range(len(gi)):
        for b in range(len(pi)):
            if stg[a] == stp[b] and float(S[a][b]) > 0.7:
                cand.append((float(S[a][b]), a, b))
    cand.sort(reverse=True); ug = set(); up = set(); pairs = []
    for s_, a, b in cand:
        if a in ug or b in up: continue
        ug.add(a); up.add(b); pairs.append((a, b))
    tp = len(pairs)
    ext_num += tp
    pr = tp/len(pi) if pi else 0.0; rc = tp/len(gi) if gi else 0.0
    if pairs:
        _, _, F_ = scorer.score([T(pt[b]) for _, b in pairs], [T(gt[a]) for a, _ in pairs])
        alphas += F_.tolist()

alpha = sum(alphas)/len(alphas) if alphas else 0.0
Ng = sum(len(gold[s].get('issue_list') or []) for s in sids)
Np = sum(len(pred[s].get('issue_list') or []) for s in sids)
F1 = 2*ext_num/(Ng+Np) if (Ng+Np) else 0.0
ext = F1*alpha
prd = sum(f_prd)/len(f_prd) if f_prd else 0.0
print(f"triplet_F1 {F1:.4f}  alpha {alpha:.4f}  ext {ext:.4f} | pred {prd:.4f} | TOTAL {0.8*ext+0.2*prd:.4f}")
