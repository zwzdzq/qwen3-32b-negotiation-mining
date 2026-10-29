# 复现说明 REPRO — 赛题3 谈判对手长文本观点挖掘与预测（提交成绩 0.6935）

生成时间: 2026-10-10 18:28 

## 1. 模型与权重
- 基座: Qwen3-32B，QLoRA 微调后已合并为完整权重（17 shards）
- 权重路径: /root/autodl-tmp/output/qwen3-32b-qlora
- 推理量化披露: bitsandbytes 4-bit nf4；device_map={'': 0}

## 2. 环境
- torch: 2.8.0+cu128
- transformers: 4.55.0
- bitsandbytes: 0.50.2
- python: 3.12.3
- cuda: True 12.8 

## 3. 解码参数
- round 0: do_sample=False（greedy，确定性）；temperature/top_p/top_k 在 do_sample=False 下被忽略（警告良性）
- round 1-2（仅格式重试）: temperature=0.3, top_p=0.9
- MAX_NEW_TOKENS=2048, BATCH_SIZE=8, MAX_INPUT_CHARS=5000（test 300 条仅 1 条超 5000 字，截断几乎不触发）
- 未固定随机种子

## 4. 推理源码与 prompt 模板
- 产生 0.6935 提交的源码: inference.py.bak2（同目录，未修改）；环境变量覆盖 IN_FILE/OUT_FILE/MAXC/MNT/BS，默认值即原始行为
- build_prompt 原文:
```python
def build_prompt(full_text, retry_round=0):
    instruction = (
        "请阅读以下谈判长文本，抽取议题、立场和论据链，并预测未来观点。"
        "只输出一个 JSON 对象：第一个字符必须是 { ，最后一个字符必须是 } ，"
        "不要 markdown 代码块，不要任何解释文字。"
        '格式：{"issue_list": [{"issue_name": "...", "stance": "support/oppose/neutral", '
        '"argument_chain": ["原文证据1"]}], "future_argument": ["未来论点1"]}。'
        "约束：future_argument 长度必须与 issue_list 完全一致；"
        "stance 只能是 support、oppose、neutral 三者之一。"
    )
    if retry_round > 0:
        instruction += (
            f"\n【重要提醒】第 {retry_round + 1} 次尝试，上次输出格式不合规。"
            "务必只输出纯 JSON，包含 issue_list 与 future_argument 两个字段且长度一致。"
        )
    return f"{instruction}\n\n谈判文本：{full_text}\n\n输出："


```

## 5. 输出后处理
- 去 markdown 围栏；解析失败最多重试 2 次；future_argument 与 issue_list 长度对齐（截多补少）；UTF-8 无 BOM；预测文本不含换行

## 6. 提交文件
- result.jsonl（300 行，md5 541f6677e0adeadac13ea331e8011c19），2026-10-10 06:47 提交，Public/通用指标(jsonl) = 0.6935
