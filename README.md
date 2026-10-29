# Qwen3-32B 谈判长文本观点挖掘：QLoRA 微调与离线推理复盘

2026 全国大数据与计算智能挑战赛 · 赛题 3「谈判对手长文本观点挖掘与预测」的参赛代码与工程复盘。

在**单张租用的 GPU、纯离线、禁止调用第三方 API** 的约束下，完成了 Qwen3-32B 的 4-bit QLoRA 微调、长文本批量推理、仿官方口径的本地评分，以及带门禁的系统化 A/B 实验。

这个仓库的主要价值不是名次，而是**含负结果的实测记录**：哪些改动被数据证伪、为什么、以及如何在不继续烧算力的前提下停止无效优化。公开前另附[勘误页](docs/ERRATA.md)，把比赛期间写下、后来被自己的实测推翻的结论逐条更正。

## 成绩

| 阶段 | 成绩 | 说明 |
| --- | --- | --- |
| 初赛 | **0.6935** | 2026-10-10 提交，当时排名第 4；300 条测试样本一次跑通，schema 全合规 |
| 复赛 | **0.508**（最佳提交） | 复赛重新计分、平台保留最佳提交；首个续训版本 candidate_02 官方分 0.5061723 |

本地评分一律是**代理分**，只用于同流程 A/B 门控，不与官方分直接比较。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `src/inference.py` | 初赛 0.6935 的权威推理源码（原 `inference.py.bak2`，未修改）：4-bit nf4 加载、左填充批处理、四级 JSON 兜底、格式重试、按 sample_id 断点续跑 |
| `src/postproc.py` | 输出侧后处理电池 identity / dedup / trim / rescue / demote（实测前四种为字面 no-op） |
| `src/val_score2.py` | 初赛本地评分器：bge-small-zh-v1.5 + 字符级 ROUGE-L + BERTScore，含两处 bert-score 补丁。**与官方口径存在已知偏差，见勘误第 3 条** |
| `src/verification/` | 复赛结果的结构与来源审计脚本：议题覆盖、future 对应关系、全量 200 条验收 |
| `docs/retrospective-preliminary.md` | 初赛完整复盘：时间线、门禁协议、实测数据表、20 条踩坑清单 |
| `docs/semifinal-prep-audit.md` | 复赛前的资产与数据核查（含对初赛文档的纠正、数据 token 长度实测） |
| `docs/semifinal-final-notes.md` | 复赛续训配置、交付与校验信息 |
| `docs/REPRO.md` | 初赛复现说明：环境、解码参数、prompt 模板 |
| `docs/ERRATA.md` | 对早期文档结论的更正，**阅读原始文档前请先读这页** |

## 不包含什么，以及为什么

- **赛题数据与测试集预测结果**：比赛规则要求测试集不得外传，因此 train/val/test 及测试集预测 jsonl 均不在本仓库。
- **模型权重**：adapter 约 0.5 GB/个，未纳入 git；复现所需的训练配置与校验和见 `docs/`。
- **任何凭据**：本仓库不含密码、token 或实例登录信息。

## 环境

初赛实测：torch 2.8.0+cu128 · transformers 4.55.0 · bitsandbytes 0.50.2 · Python 3.12；硬件为单张 RTX 6000D（85.6 GB VRAM）。

评分器另需 `bert-score`、`BAAI/bge-small-zh-v1.5`、`bert-base-chinese`。国内机器建议大权重走 ModelScope：hf-mirror 实测仅 23–146 kB/s 且默认 10 s 读超时会直接失败。

## 使用

推理（需自备数据与权重，并修改脚本顶部的路径配置）：

```bash
python src/inference.py
```

本地评分：

```bash
GOLD_FILE=val40.jsonl BGE_PATH=/path/to/bge BERT_PATH=/path/to/bert \
  python src/val_score2.py pred.jsonl
# 输出：triplet_F1 / alpha / ext / pred / TOTAL
```

## 实测结论（含负结果）

1. **显存**：fp16 加载 32B 权重 = 75.7 GB，几乎不留 KV cache 余量；4-bit nf4 = 19.8 GB，单卡可跑。`kill -9` 不释放孤儿显存，只能重置 GPU。
2. **性价比最高的修复**：`MAX_NEW_TOKENS` 512 → 2048。512 会截断 JSON，首轮解析失败率 75%；提到 2048 后首过解析率 ≈100%。
3. **加大 batch 不省时间**：BS=4 与 BS=8 稳态均为 56–60 s/条（KV-cache 读带宽受限）；首个 batch 有 ~250 s 预热，不能用来估 ETA。
4. **初赛两轮 prompt 挑战均大幅劣于 baseline**（TOTAL 0.6094 → 0.4038 / 0.4953）。其实测结果成立，但当初的机理归因在复赛核查中被削弱，见勘误第 2 条。
5. **五种零成本后处理中四种是字面 no-op** → 据此关闭优化周期，保住已入库成绩，不再拿成绩赌提交。
6. **任务分布漂移时该动权重而不是动 prompt**：复赛 oppose 占比从 0.5% 漂到 31%，128 步 QLoRA 续训（约 44 分钟）把 oppose 预测占比校正到 35%。
7. **科研诚信**：本地代理分更高、但证据系凭空编造的候选一律拒绝采用——涨分不等于进步。

## License

MIT
