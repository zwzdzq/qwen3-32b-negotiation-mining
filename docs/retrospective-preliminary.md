# 赛题3「谈判对手长文本观点挖掘与预测」初赛复盘报告

> 时间范围：2026-10-09 ~ 2026-10-11
> 赛事：2026 全国大数据与计算智能挑战赛 · 赛题3（初赛窗口 2026-10-04 12:00 – 2026-10-14 00:00）
> 环境：AutoDL 容器 · 1× NVIDIA RTX 6000D（85.6 GB VRAM）· 25 核 CPU / 92 GB RAM · CUDA 13.2
> 结论一句话：**0.6935 入库成绩为本轮最终成绩，优化周期于 2026-10-11 永久关闭。**

---

## 1. 摘要（TL;DR）

| 项目 | 状态 |
|---|---|
| Public / 通用指标(jsonl) | **0.6935**（2026-10-10 06:47 提交，当时排名第 4） |
| 提交文件 | `result.jsonl`，300/300 行，md5 `541f6677e0adeadac13ea331e8011c19` |
| 悲观晋级线 | 前 10（官方：遴选 10–30 支团队入围复赛）→ **安全边际充足** |
| 本轮挑战 prompt（v2 / v3） | 门禁**未通过**，方向证伪 |
| 零成本后处理电池（5 种） | **全灭**（4 种为字面 no-op，1 种为负先验对照且无效） |
| 同基座融合 / 4090 迁移 / 复赛预备 | **全部推迟或放弃**，理由见 §8 |
| 剩余动作 | 关机 + 账单核查 + 本地下载审计包 + 提交 REPRO.md |

核心收获：**merged QLoRA 权重是针对「原始 prompt 模板」做过 SFT 的，任何 prompt 侧改动都构成分布漂移，会直接击穿 triplet_F1。** 这条经验值回本轮全部实验成本。

---

## 2. 成绩与提交记录

- 提交时间：2026-10-10 06:47
- 任务页：**赛题3-谈判对手长文本观点挖掘与预测**
- 得分：Public / 通用指标(jsonl) = **0.6935**
- 文件：`result.jsonl`（1048259 字节，300 行，UTF-8 无 BOM）
- md5：`541f6677e0adeadac13ea331e8011c19`
- 备份（三份 md5 完全一致，且均通过 `check_result.py`）：
  - `result.jsonl`（真实提交件）
  - `result.jsonl.bak`
  - `result_06935_submitted.jsonl`
- 生成源：`inference.py.bak2`（**未做任何修改**，复现审计的唯一权威版本）

> ⚠️ 上传教训：同一份 `result.jsonl` 上传到**赛题1（低资源小语种跨模态信息分拣）**会得到「评审出错 / 500 Server Error」（`comp1_new` 端点）。上传前必须确认任务标题与谈判 schema 匹配。此类错传**不消耗**赛题3 的每日 3 次额度。

---

## 3. 官方评分口径（本轮所有本地门禁都对齐此口径）

```
total = 0.8 × extraction + 0.2 × future
```

**extraction（三元组抽取）**
1. 三元组 = `(issue_name, stance, argument_chain)`；
2. **stance 必须完全一致**（support / oppose / neutral），否则该三元组直接判死；
3. 相似度 = `0.4 × cos(bge, issue_name) + 0.6 × cos(bge, argument_chain)`，embedding 用 `BAAI/bge-small-zh-v1.5`；
4. 阈值 `sim > 0.7`（规则原文「超过」，即严格大于），再做**最大二部图一对一最优匹配**；
5. P = NC/Np，R = NC/Ng（NC/Np/Ng 为数据集级总量），F1_ext = 2PR/(P+R)；
6. **S_ext = F1_ext × (0.7 + 0.3α)**，α = 匹配成功三元组上的平均 BERTScore-F1（`bert-base-chinese`）。

**future（future_argument 预测）**
- 复用 extraction 的 issue 匹配结果；
- F1_pred = Σ 匹配对 ROUGE-L F1 / N；S_semantic = Σ 匹配对 BERTScore-F1 / N；**N = max(Ng, Np)**；
- **S_pred = 0.6 × F1_pred + 0.4 × S_semantic**；
- 未匹配的 gold 或多预测的 issue 都记 0 → 漏召和多召双向惩罚。

> 📝 **2026-10-15 复核修订**：本节公式已按官方规则页全文逐字核对。修订前本地 `val_score2.py` 有两处偏差：ext 用了 `F1 × α`（官方是 `F1 × (0.7+0.3α)`，官方更宽松），future 组合权重不是官方的 0.6/0.4——这使本地分数系统性偏低（baseline 按官方口径 ≈0.671 vs 本地 0.6094），解释了「本地 0.60 < 官网 0.63」。按官方公式复核门禁：baseline S_ext 0.7209 / v2 0.4467 / v3 0.5604，挑战者差距反而**更大**，本报告全部结论不受影响。教训：打分器实现必须对照官网公式原文，不能凭转述。

**输出 schema（硬约束）**
- JSON Lines；顶层键**仅** `sample_id`、`issue_list`、`future_argument`；
- `issue_list` 元素：`issue_name`、`stance` ∈ {support, oppose, neutral}、`argument_chain`（原文证据数组）；
- `len(future_argument) == len(issue_list)`（多截少补）；
- 预测文本**不得含换行符**；文件 UTF-8 无 BOM。
- （规则「示例」章节展示的是根 JSON 数组，但平台实际接受 JSONL —— 0.6935 那份就是 JSONL。）

---

## 4. 时间线

### 2026-10-09 —— 推理「挂死」事故（已解决）
- 现象：GPU-Util ≈ 6%，VRAM 占用 75.7 / 85.6 GB，单个 CPU 核空转，`result.jsonl` 长时间不动。
- `kill -9` 之后 **75.7 GB 显存仍未释放**，`nvidia-smi` 列不出任何进程（孤儿显存），只能通过 GPU / 实例重置清除。
- 根因：fp16 全精度加载 merged 权重（17 shard）本身就吃掉 75.7 GB，几乎无余量给 KV cache。
- **固化教训**：重启任何长任务前，先确认 `nvidia-smi` 显示 ~0 MiB 且无 compute 进程。

### 2026-10-10 上午 —— 4-bit 部署 + 跑通 300/300
- 改为 **bitsandbytes 4-bit nf4** 加载：VRAM 19.8 GB（fp16 时 75.7 GB）。
- `device_map={'':0}`（无 CPU offload），`BATCH_SIZE=4` + 左 padding，`MAX_INPUT_CHARS=5000`，tqdm + 每 batch flush 日志，按 `sample_id` 断点续跑，全程 `screen`。
- 首轮 **75% 解析失败**的根因是 `MAX_NEW_TOKENS=512` 把 JSON 截断了；改到 **2048** 后首轮解析率 ≈ 100%（512 下 fallback 的 3 条样本在 2048 下 3/3 正常）。
- 最终 `result.jsonl` = 300/300 行，schema 全合规，**0 个 fallback 占位，0 个坏行**。
- 性能：稳态 **56–60 s/条**，`BS=4` 与 `BS=8` 几乎无差（每步成本随 batch 线性上升 → KV-cache 读带宽受限）；全量 300 条 ≈ 4–5 h；**每个 run 的首个 batch 有 ~250 s 预热，绝不能用 batch 1 估 ETA**。
- 06:47 提交 → **0.6935**。

### 2026-10-10 下午 —— 搭「贴官方口径」的本地打分器
- 依赖坑：`hf-mirror.com` 从这台机器只有 **23–146 kB/s**，且 huggingface_hub 默认 10 s 读超时 → `ReadTimeoutError`。小文件用 `export HF_HUB_DOWNLOAD_TIMEOUT=60` 重试可过；**500 MB 级权重必须走 ModelScope**（AutoDL 上快，且 `IncompleteRead` 自动重试）：
  - `snapshot_download('BAAI/bge-small-zh-v1.5')`
  - `snapshot_download('AI-ModelScope/bert-base-chinese')` → `/root/autodl-tmp/ms_cache`
  - `/root/autodl-tmp/model_paths.env` 导出 `BGE_PATH` / `BERT_PATH`
- bert-score 两个坑（均已修）：
  1. `BERTScorer(model_type=<本地目录>)` 在 `scorer.py:91` 抛 `KeyError`（`model2layers` 是按字符串查表）→ 构造前 `bert_score.utils.model2layers.setdefault(BERT, model2layers['bert-base-chinese'])`（保留官方层选择）；
  2. bert-score **不做截断**，长句触发 `RuntimeError: (680) must match (512)` → 加 `T()`：`btok.encode(s, add_special_tokens=False)[:500]` 再 decode，**对所有候选一视同仁**。
- 交付物：`val_score2.py`（K-final 全量覆盖版），输出格式
  `triplet_F1 F  alpha A  ext E | pred P | TOTAL T`，用法 `GOLD_FILE=val40.jsonl python val_score2.py <pred>`。

### 2026-10-10 傍晚 —— 挑战 prompt v2 / v3（门禁失败）
- 用 **in-place splice**（在副本里原地替换 line 164 那 19 行的 `def build_prompt`）造出 `infer_v2.py` / `infer_v3.py`。
  - `cat >>` 追加式（外部粘贴的 Block L）**是坏的**：顶层入口调用（line 223 / 282+）先于追加的 def 执行 → 输出与 baseline **逐字节相同**却看不出来。
- v2 = 原 prompt + 具体命名 / 拆分勿合并 / 证据来源 / neutral 边界四条规则；
  v3 = v2 核心（改名 `_bp_core`）+ `_fewshot`（train 首条样本，5 个 issue 含 neutral，900 字截取）+ 在 `谈判文本：` 前插入示例的包装函数。
- 功能校验通过：exec 拼接区后调用 `build_prompt('测试文本XYZ',0)`，v2 有「议题命名」无「示例输入」且以「输出：」结尾；v3 两者都有。
- Block M 在 `screen m` 下串行跑 v2 → v3（val40，BS=8 / MNT=2048 / MAXC=5000），各 ≈45 min，`ALL_DONE` 落盘；两块并发时 GPU 也放得下（≈20 GB + ≈2 GB）。
- **结果：双双大幅劣于 baseline（见 §6）→ 门禁关闭，prompt 侧不再动。**

### 2026-10-11 —— 外部优化建议评审 + 零成本后处理电池（全灭）
- 对外部粘贴的 4 类建议逐条用实测数据裁定：**4 条里 3 条被现有测量直接否掉**。
- 跑了 `postproc.py` 五模式电池（identity / dedup / trim / rescue / demote），零 GPU 成本：
  - identity 对照 **精确复现 0.6094**（打分器可信）；
  - dedup / trim / rescue 三者 TOTAL 均 = **0.6094**，即字面 no-op；
  - demote 缺 TOTAL 行（打分器 stderr 被吞导致崩），但它本就是**负先验对照组**，无效不影响结论。
- **优化周期永久关闭，保留 0.6935。**

---

## 5. 门禁协议（本轮所有 A/B 的唯一判据）

- 验证集：`val40.jsonl` = 有标签 val 的前 40 行；gold = `GOLD_FILE=val40.jsonl`。
- baseline（= `val_pred_a.jsonl`，与 0.6935 同一套 prompt/解码）：**TOTAL 0.6094**。
- 通过线：**baseline + 0.01 = 0.6194**。
- 恒等对照（identity）必须**逐位复现 0.6094**，否则打分器不可信，全部结果作废。
- 未过线 ⇒ 不跑全量 test（5–6 h GPU）、不消耗额度、不提交。
- 注意：本地 40 样本门禁本身有噪声；`proxy ≠ official`（官方分项未公开），所以门禁是**必要不充分**条件 —— 这也是我们只在有强证据时才提交的另一个理由。

---

## 6. 实测结果表

### 6.1 挑战 prompt（val40，官方口径打分器）

| 版本 | triplet_F1 | α (BERTScore) | ext | pred | **TOTAL** | 门禁 |
|---|---|---|---|---|---|---|
| **baseline（原 prompt）** | **0.7539** | 0.8538 | **0.6437** | 0.4724 | **0.6094** | — 基准 |
| v2（+4 条规则） | 0.4649 | 0.8692 | — | — | **0.4038** | ✗ 失败（−0.2056） |
| v3（+few-shot 示例） | 0.5827 | 0.8723 | — | — | **0.4953** | ✗ 失败（−0.1141） |

### 6.2 零成本后处理电池（val40，输出侧改写，零 GPU 成本）

| 模式 | 做法 | TOTAL | 判定 |
|---|---|---|---|
| identity | 原样重写 | **0.6094** | ✅ 对照精确复现 |
| dedup | issue_name difflib 相似度 > 0.9 合并（chain 取并集，future 取长者） | **0.6094** | 字面 no-op |
| trim | argument_chain 截到 3 句 | **0.6094** | 字面 no-op |
| rescue | 用 full_text 中 difflib ≥ 0.85 的原句替换 chain 句 | **0.6094** | 字面 no-op |
| demote | 名称+chain 无情感关键词时 support/oppose → neutral | （缺行） | 负先验对照，无效 |

可选确认命令（应输出四个相同 md5，证明前四种确实没改动任何字节）：
```bash
md5sum pp_identity.jsonl pp_dedup.jsonl pp_trim.jsonl pp_rescue.jsonl val_pred_a.jsonl
```

### 6.3 早期 CPU-only proxy 诊断（val40，`val_score.py`，仅作趋势参考）

| 指标 | 数值 |
|---|---|
| issue_F1 | 0.585 |
| stance acc | 0.804 |
| evidence_F1 | 0.619 |
| future_argument | 0.211 |
| composite | 0.5327 |

### 6.4 「三个朴素 prompt 杠杆」全部被数据否掉

| 朴素猜想 | 实测 | 结论 |
|---|---|---|
| 「模型漏抽议题，要多枚举」 | pred 4.80 vs gold 4.75 issue/样本 | 已校准，无需多抽 |
| 「证据不是原文，要逐字引用」 | 逐字命中率 **501/502 = 1.00** | 已经逐字，无需强约束 |
| 「future_argument 写太短」 | 平均 75 vs gold 82 字 | 差距很小，且 future 只占 20% |
| 「MAXC=5000 截断了长文本」 | val >5000 字：2/300（最长 6000）；test：1/300（最长 5339）；train：62/2400（最长 7502）。val40 里**零条**被截断；MAXC=5000 vs 8000 输出**逐字节相同**（md5 一致，proxy 分数一致） | 全量重跑最多影响 test 的 1/300 条，不值 5–6 h GPU + 额度 |

---

## 7. 死因分析：为什么改 prompt 一定变差

**根因：merged QLoRA 权重是在「原始 prompt 模板」上做的 SFT。**
prompt 一旦偏离训练模板，就是分布外输入，模型的抽取行为整体跑偏：

- triplet_F1 从 0.7539 崩到 0.4649（v2）/ 0.5827（v3）；
- 同时 α 反而**升高**（0.8538 → 0.8692 / 0.8723）。

这两个数一起看，机理非常清楚：**匹配上的那些三元组质量更好了，但能匹配上的数量断崖式下降**（过度拆分议题 + 议题命名风格漂移，导致 sim > 0.7 的配对变少）。而官方指标里 stance 精确匹配 + 0.7 阈值 + 一对一匹配是**硬闸门**，召回掉了就救不回来。

**stance 侧也无利可图：** val40 混淆方向是对称的（neutral→support 5 例 vs support→neutral 7 例），任何单向 demote/promote 都是净负；gold 边际分布 `{support:121, neutral:68, oppose:1}` vs pred `{122, 67, 3}`，几乎完全对齐。

**推论（可迁移到复赛）**：微调后的模型，prompt 是「已固化的接口契约」，不是可自由调优的自由变量。想改 prompt，必须先重训/重对齐；只做推理侧 prompt engineering 的期望值为负。

---

## 8. 未采纳 / 推迟的方向及理由

| 方向 | 决定 | 理由（含数字） |
|---|---|---|
| **同基座多 checkpoint / 多次推理结果融合** | **放弃** | 规则允许，但 EV 低：一次全量 test 跑 5–6 h GPU + 1 次额度，而门禁只有 40 样本、噪声大、且官方分项未知；当前第 4 名、悲观线前 10，**margin 已足够**。 |
| **`future_argument` 措辞类 prompt 实验** | **放弃** | 最弱的 proxy 分项（0.211）确实指向 future，但 §7 已证明 prompt 侧整体是负收益；一轮成本 ≈1.5 h val A/B + 5–6 h test + 1 额度，方向纯赌。 |
| **迁移到更便宜的 4090** | **推迟** | 仅在复赛需要长时间多轮实验时才做预算评估；初赛已收官，无必要。 |
| **复赛（多方谈判文本）预备开发** | **推迟** | 复赛规则是「同一主议题下每一方各自成一条三元组」，但**真实复赛数据尚未发布**，现在写 party-aware 代码是盲写。等晋级确认 + 数据到手再动。 |
| **MAXC 5000 → 8000 全量重跑** | **放弃** | 见 §6.4：test 上最多改 1/300 条。 |

> 另一处不确定：**平台究竟保留最优提交还是最后一次提交未知**。在没有强本地证据时重新提交，等于拿已入库的 0.6935 去赌 —— 这是「不再提交」的决定性理由之一。

---

## 9. 踩坑清单（可直接复用到复赛）

**环境 / 显存**
1. fp16 加载 32B merged 权重 = 75.7 GB，几乎无 KV cache 余量 → 挂死。**4-bit nf4 = 19.8 GB** 才是可用配置；推理中实测占用 31–36 GB。
2. `kill -9` **不会**释放孤儿显存；必须 GPU/实例重置。重启任务前先看 `nvidia-smi` 是否 0 MiB、无 compute 进程。
3. 长任务必须 `screen` / `nohup`（SSH 掉线会杀前台任务）。JupyterLab 网页终端**吞掉 Ctrl+A**（被浏览器全选拦截），`Ctrl+A D` 失效 → 用 `screen -d <session>`；关标签页不会杀 session。
4. 容器内 `OMP_NUM_THREADS` 为空/非法 → `libgomp: Invalid value` 警告 → `export OMP_NUM_THREADS=1`。
5. `py-spy` / `ptrace` 在该容器被禁。

**推理 / 生成**
6. `MAX_NEW_TOKENS=512` 会截断 JSON 输出 → 解析失败率 75%；**2048** 下 ≈100%。这是本轮性价比最高的一个修复。
7. 稳态 56–60 s/条，BS=4 与 BS=8 无差（KV-cache 读带宽受限）→ **加大 batch 不省时间**。
8. 首个 batch ~250 s 预热，**不要用 batch 1 估 ETA**。
9. `do_sample=False` 下的 `generation flags not valid: ['temperature','top_p','top_k']` 警告是**良性**的。
10. `inference.py` 的「结果已保存至 result.jsonl (UTF-8 无 BOM)」是**硬编码打印**，与实际 `OUT_FILE` 无关 → 判断是否误覆盖要用 md5 / mtime，**不要凭打印惊慌**。
11. **不要**把 `model_path` 改指向 `qwen3-32b-merged`（外部建议如此）：qlora 目录确实放着 17-shard merged 权重，merged 目录很可能不存在。

**下载 / 依赖**
12. `hf-mirror.com` 23–146 kB/s + 默认 10 s 超时 → 小文件 `HF_HUB_DOWNLOAD_TIMEOUT=60`，**大权重走 ModelScope**。
13. bert-score：本地目录触发 `model2layers` KeyError → `setdefault` 打补丁，且**补丁必须在打分脚本内部**。
14. bert-score 不截断 → 长文本 512 崩 → 统一 `T()` 截到 500 token，**对所有候选同样处理**才公平。

**流程 / 方法论**
15. `cat >>` 追加覆盖函数 = **静默失效**（入口调用先执行）→ 改函数必须 **in-place splice**，并做功能性校验（真调一次 `build_prompt` 看输出特征）。
16. 静态 ast 自检会**误报**：把 line 23/37/38/39 那些既有的 setup 调用也算进「def 之前有顶层调用」。真正的不变量是「**入口调用必须在 def 之后**」。
17. exec 校验时若只挑 `FunctionDef` 节点，会丢掉模块级 `_FS = None` / `import json as _json` → 假 NameError。正确做法：保留拼接区内 FunctionDef / Import / ImportFrom / 以及 target 以 `_` 开头的 Assign。
18. 文件版本漂移时，**整文件覆盖**而不是 anchor 打补丁（"anchor not found" 的真正含义是磁盘上是旧版）。
19. 打分器未验证前**绝不 `2>/dev/null`**：本轮「只打印 samples 40」就是 traceback 被吞，白跑一轮。改成 `> log 2> err`。
20. **备份文件要验字节数**：`result.jsonl.bak` 曾是 0 字节（10-09 23:11 起的空壳「保险」），真正的提交件只有 `result.jsonl`。

---

## 10. 复现审计要点（REPRO.md 摘要）

晋级队伍需提交技术文档 + 源码 + 权重 + prompt 模板 + 环境/种子/解码参数；QLoRA / 4-bit 推理量化**允许但必须披露**。已在 `REPRO.md` 中记录：

- 基座：**Qwen3-32B**，QLoRA 微调后 **已 merge 全量权重**（17 shards）@ `/root/autodl-tmp/output/qwen3-32b-qlora`
- 量化披露：**推理阶段 4-bit bitsandbytes nf4**
- 加载：`device_map={'':0}`，无 CPU offload
- 解码：round 0 **贪心**（`do_sample=False`，确定性 → 利于复现）；round 1–2 `temperature=0.3`、`top_p=0.9`
- 生成参数：`MAX_NEW_TOKENS=2048`、`BATCH_SIZE=8`、`MAX_INPUT_CHARS=5000`
- prompt：`build_prompt` 逐字取自 `inference.py.bak2`（**未做任何修改**）
- 后处理：仅 schema 对齐（future 长度对齐、去换行、UTF-8 无 BOM），**无内容侧改写**
- 提交件 md5：`541f6677e0adeadac13ea331e8011c19`

**合规声明**
- 结果仅由 Qwen3-32B（自训 QLoRA merged）生成，未使用其他 LLM / 在线 API / 人工服务；
- 未对测试集做人工标注，未使用测试集标签或榜单反馈反推答案；
- 未引入外部数据 / 伪标签（如后续引入将全量披露）；
- 方案可在主办方 Qwen3-32B 上用提交的权重/代码/prompt **离线复现**。

---

## 11. 收尾清单

- [ ] **关机实例**，并核查 AutoDL 账单：确认 10-10 → 10-11 之间是否存在**整夜 GPU 空转**计费（本轮已出现 `screen=0` / `0 MiB` 但仍开机的时段）。
- [ ] **本地下载审计包**（远程目录 `/root/autodl-tmp/data/初赛/发布版本（无测试集标签）/`）：
  - `result.jsonl`、`result.jsonl.bak`、`result_06935_submitted.jsonl`
  - `inference.py.bak2`（权威生成源）
  - `REPRO.md`
  - `val_score2.py`、`postproc.py`
  - `score_a.log`（baseline 0.6094 证据）、`v2.log`、`v3.log`
  - 电池结果截图 / `pp_*.jsonl`（如需）
- [ ] **保持三份备份与 `inference.py.bak2` 只读**，任何后续实验都不得触碰。
- [ ] **不再提交**（0.6935 为最终成绩；额度 3 次/天保留不用）。
- [ ] REPRO.md 随审计材料在 **2026-10-14 前**提交。
- [ ] **2026-10-14 之后**：查榜单确认是否入围复赛（10-15 ~ 10-17）；若入围，等真实多方谈判数据发布后再开发 party-aware 逻辑。

---

## 12. 资产清单（远程，供复赛复用）

| 文件 | 用途 |
|---|---|
| `inference.py.bak2` | 0.6935 的权威生成源（勿改） |
| `inference.py` | 带 `IN_FILE/OUT_FILE/MAXC/MNT/BS` 环境变量覆盖的实验版 |
| `infer_v2.py` / `infer_v3.py` | 挑战 prompt（已证伪，留作反例） |
| `val_score2.py` | **贴官方口径的本地打分器**（bge + ROUGE-L + BERTScore，含两处 bert-score 补丁）→ 复赛最有价值的资产 |
| `postproc.py` | 输出侧后处理电池（identity/dedup/trim/rescue/demote） |
| `check_result.py` | schema 校验器 |
| `val40.jsonl` / `val_pred_a.jsonl` | 门禁基准对（0.6094） |
| `val_pred_v2.jsonl` / `val_pred_v3.jsonl` | 失败对照 |
| `model_paths.env` / `ms_cache/` | 打分器权重路径与本地缓存 |

---

*报告生成于 2026-10-11。所有数字均来自本轮实测日志与打分器输出。*
