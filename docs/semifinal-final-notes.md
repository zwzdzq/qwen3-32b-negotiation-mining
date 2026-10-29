# 复赛续训与交付记录

整理自 2026-10-17 / 10-18 的交接与审计材料，已去除第三方本机路径。

## 结论

- 复赛最佳提交 **0.508**（复赛重新计分，初赛成绩不累计；平台保留最佳提交）。
- 首个续训版本 **candidate_02** 官方分 **0.5061723098266686**，作为兜底版本保留。
- 0.2426 / 0.2636 / 0.4933 等数字是 val24 或 2 条小测的**本地代理分**，不能与官方分直接比较；2 条小测的提升不等于整体提升。

## candidate_02 续训配置（LLaMA-Factory，实际执行）

| 字段 | 值 |
| --- | --- |
| 基座 | Qwen3-32B |
| 起点 | 初赛 initial_adapter 上继续训练（`create_new_adapter: false`） |
| stage / finetuning_type | sft / lora |
| lora_target / rank / alpha | all / 16 / 32 |
| quantization_bit / bf16 | 4 / true |
| template | qwen3_nothink |
| cutoff_len | 16384 |
| per_device_train_batch_size / gradient_accumulation_steps | 1 / 2 |
| learning_rate / max_steps | 2.0e-5 / 128 |
| lr_scheduler_type / warmup_ratio | cosine / 0.05 |
| gradient_checkpointing / flash_attn | true / sdpa |
| seed / data_seed | 20261016 / 20261016 |
| save_only_model | true |

- 实跑 **128 步**，含加载与保存共 **2658.49 秒（约 44.3 分钟）**。
- 数据池为官方 train 1600 条；按 batch 1 × 累积 2 估算实际处理约 **256 个样本**，并未完整训完一遍 1600 条。
- candidate_02 adapter SHA256：`45250b13aa1cc02e5b25789ccad242bf7bebc222c61f393093e5d49c81756d8a`
- 效果：**oppose 预测占比从初赛模型的约 0.5% 校正到 35%（gold 31%）**；议题数 11.48/样本（gold 11.23）；future 平均 48.5 字（gold 43–45）。

## 交付与校验

- 200 条提交中 196 条一次通过，其余 4 条经分步生成修复；所有答案组件仍由同一 Qwen3-32B + candidate_02 adapter 生成。
- 不新增 party 字段、不按 issue_name 去重、不人工补答案、不截断或补占位 future。
- 提交文件 SHA256：`43720f1bf0d8e3a79895b8b921ff80ffd81c066136cd98d7925b080784fe1816`。
- 已验证：200 个唯一 ID 与测试输入顺序一致、UTF-8 无 BOM、字段合法非空、stance 合法、issue/future 等长、无完全重复三元组，共 2296 条议题。
- **测试集没有标签**：结构校验通过不代表准确率验证。
- **未在官方环境完成 8 小时复现测试**：材料齐全不等于复现已通过。

## 科研诚信

候选 42–45 等版本的本地代理分更高，但逐条溯源发现证据含原文没有的机构名（如凭空出现的国际组织）与跨参与方污染细节，**一律未采用**。
