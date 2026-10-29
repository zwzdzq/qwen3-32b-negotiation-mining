# 复赛无卡阶段准备结果

日期：2026-10-15。已成功通过用户提供的凭据接入原实例。本文件不包含认证信息。

远端独立目录：`/root/autodl-tmp/semifinal-20261015-codex`。没有修改初赛原文件或已有 `val_score3.py`，没有启动训练、推理或平台提交。

## 资产核验

| 路径（相对 /root/autodl-tmp） | 实测内容                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| models/Qwen/Qwen3-32B   | 原始基座；索引引用的 17 个分片均存在                                                                        |
| output/qwen3-32b-qlora  | adapter，536,991,984 字节；r=16、alpha=32、目标为七类 attention/MLP 投影；含 checkpoint-500、checkpoint-900 |
| output/qwen3-32b-merged | 完整合并模型；索引引用的 17 个分片均存在                                                                      |
| train_qwen3_qlora.yaml  | 初赛 LLaMA-Factory 配置，3 epoch、学习率 1e-4、梯度累积 8、cutoff_len=16384                                |

上述分片只核验文件存在及索引，未读取全部权重计算校验和或验证参数差异。初赛文档把 qlora 目录描述成完整合并模型，与当前实际目录不符。

基线推理仍沿用初赛 `AutoModelForCausalLM.from_pretrained(adapter目录)` 的加载路径，由现有 Transformers/PEFT 集成处理 adapter；GPU 恢复后须确认实际加载日志和 `adapter_active`。未在已合并模型上另加同一个 adapter。

实际训练 instruction 与初赛推理 build_prompt 不相同，不能把“改 prompt 变差必然是偏离训练模板”当作已证实原因。

实例 cgroup 配额确认为 2 GiB 内存、0.5 核 CPU。主机 os.cpu_count 返回 224，不代表实例可用 CPU 配额。仅使用标准库及 tokenizer 做准备。

## 数据和 token 实测

| 集合    |  样本数 | 完整原文最长字符 | 超过 8000 字符 | 完整原文最长 token | 标签最长 token | 标签超过 2048 token |
| ----- | ---: | -------: | ---------: | -----------: | ---------: | --------------: |
| train | 1600 |    11803 |         26 |         7451 |       3279 |              92 |
| val   |  200 |     9123 |          4 |         5543 |       2755 |               7 |
| test  |  200 |     9458 |          5 |         5655 |        无标签 |             无标签 |

token 使用原始基座 tokenizer.json 统计，原文统计不包含 prompt 或 chat template；标签序列化含 issue_list 和 future_argument。测试集只检查输入结构和长度。

因此基线采用 MAXC=12000、MNT=4096、BS=1 作为启动配置。MAXC 超限直接报错，不静默截断。4096 只是基于标签长度的候选预算，模型实际生成是否触顶必须在 GPU 上检查。

## 已完成的代码与数据

- `val_score3_checked.py`：以远端已有 val_score3 为基础，修复无共同字符时 ROUGE-L 除零；提前校验重复 ID、缺失/多余 ID、标签结构和长度；要求 scipy，避免悄悄退回贪心。缺失预测会中止评分，不会输出虚高成绩。仍是本地代理：micro 汇总、mean pooling、字符 ROUGE-L、BERTScore 截断和匹配细节尚未经官方实现逐项确认。
- `inference_baseline.py`：保留初赛推理 prompt 与清洗规则，显式输入/输出路径；新输出文件保护；离线加载；单 GPU、种子 42、输出预算 4096；运行日志报告 adapter 是否激活。保留原格式重试和兜底逻辑，须在验证结果中统计失败与占位，不能仅以脚本退出成功判断结果可提交。
- `prepare_data.py`：保留原训练 instruction，完整转换 train/val，不去重、不截断、不补造三方标签。转换内容已与原文件逐条核对。
- `data/val24.jsonl`：将 200 条验证样本按文本长度排序分四组，各随机选 6 条，seed=20261015；ID 固定于 `data/manifest.json`。用于快速诊断，不作为最终择优的唯一依据，完整 200 条仍需评估。
- `data/train_alpaca.json`、`data/val_alpaca.json` 和 `dataset_info.json` 已准备；没有把验证集加入训练。

测试：3 项轻量测试通过，覆盖无交集/部分/完整/空字符串 ROUGE-L、ID 重排、缺失、额外、重复以及标签数量不对齐。三个脚本在远端语法检查通过。尚未运行模型推理、完整 embedding/BERTScore 评分或训练。

## GPU 恢复后第一轮基线命令

以下命令只在 GPU 模式恢复并确认显存、当前负载后执行。输出文件必须不存在；重复实验使用新名字。

```bash
cd /root/autodl-tmp/semifinal-20261015-codex
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
IN_FILE=/root/autodl-tmp/semifinal-20261015-codex/data/val24.jsonl \
OUT_FILE=/root/autodl-tmp/semifinal-20261015-codex/val24_baseline.jsonl \
MAXC=12000 MNT=4096 BS=1 \
/root/miniconda3/bin/python -u inference_baseline.py > val24_baseline.log 2>&1
```

实际启动时应放入可持续的后台会话，先检查首个样本和实际 adapter 加载，再继续。评分时 BGE_PATH/BERT_PATH 应指向已存在的本地模型缓存；这些缓存目录的具体路径尚待核验，不使用网络下载默认值。

后续依次：验证生成长度、格式和实际 GPU 显存 → 跑代理评分并记录分项 → 决定复赛微调配置 → 完整验证 → 生成和检查测试提交。算力预算及切回 GPU 的时间尚待用户提供。
