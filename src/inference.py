import json
import os
import re
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 配置区 =================
model_path = "/root/autodl-tmp/output/qwen3-32b-qlora"   # 已验证：合并后完整权重（17 shards）
tokenizer_path = "/root/autodl-tmp/models/Qwen/Qwen3-32B"
input_file = "/root/autodl-tmp/data/初赛/发布版本（无测试集标签）/test.jsonl"
output_file = "/root/autodl-tmp/data/初赛/发布版本（无测试集标签）/result.jsonl"
debug_file = "fail_debug.jsonl"

MAX_INPUT_CHARS = 5000
MAX_NEW_TOKENS = 2048     # 调大防 JSON 被截断（截断=括号不闭合=解析失败）
MAX_RETRY = 2
BATCH_SIZE = 8            # 显存若稳定 <40G 可改 8 再提速
# ==========================================

print("正在加载模型和分词器（4bit 量化）...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
tokenizer.padding_side = "left"   # 批量生成必须左填充

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True,
    quantization_config=bnb_config,
)
model.eval()
print(f"显存占用: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
print(f"device_map: {model.hf_device_map}", flush=True)


# ============ 文本清洗与规范化 ============
def clean_str(s):
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def normalize_stance(stance):
    s = clean_str(stance).lower()
    mapping = {
        "support": "support", "支持": "support", "pro": "support", "favor": "support",
        "oppose": "oppose", "反对": "oppose", "against": "oppose", "con": "oppose",
        "neutral": "neutral", "中立": "neutral", "unknown": "neutral", "": "neutral",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return "neutral"


# ============ JSON 提取（四级兜底）============
def strip_markdown(text):
    text = text.strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def _balanced_braces(text):
    depth, start, in_str, escape = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def extract_json(text):
    if not text:
        return None
    text = strip_markdown(text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        balanced = _balanced_braces(candidate)
        if not balanced:
            return None
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", balanced)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None


def normalize_result(result_data):
    if not isinstance(result_data, dict):
        result_data = {}
    raw_issues = result_data.get("issue_list")
    raw_future = result_data.get("future_argument")

    issues = []
    if isinstance(raw_issues, list):
        for it in raw_issues:
            if not isinstance(it, dict):
                continue
            chain = it.get("argument_chain")
            if isinstance(chain, str):
                chain = [chain]
            if not isinstance(chain, list):
                chain = []
            chain = [clean_str(c) for c in chain if clean_str(c)] or ["无"]
            issues.append({
                "issue_name": clean_str(it.get("issue_name", "")) or "未命名议题",
                "stance": normalize_stance(it.get("stance", "")),
                "argument_chain": chain,
            })
    if not issues:
        issues = [{"issue_name": "解析失败", "stance": "neutral", "argument_chain": ["无"]}]

    future = []
    if isinstance(raw_future, str):
        future = [raw_future]
    elif isinstance(raw_future, list):
        future = [clean_str(x) for x in raw_future]

    target_len = len(issues)
    future = [f if f else "待预测" for f in future]
    if len(future) < target_len:
        future.extend(["待预测"] * (target_len - len(future)))
    else:
        future = future[:target_len]
    return issues, future


# ============ Prompt 与生成 ============
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


def generate_batch(prompts, retry_round=0):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    in_len = inputs.input_ids.shape[1]
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
    if retry_round == 0:
        kwargs.update(do_sample=False)
    else:
        kwargs.update(do_sample=True, temperature=0.3, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return [tokenizer.decode(out[i][in_len:], skip_special_tokens=True)
            for i in range(out.shape[0])]


def build_full_text(item):
    full_text = ""
    for doc in item.get("docs", []):
        full_text += clean_str(doc.get("full_text", "")) + "\n\n"
    return full_text[:MAX_INPUT_CHARS]


# ============ 主流程（断点续跑 + 批处理 + 失败落盘）============
done_ids = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue

items = []
with open(input_file, "r", encoding="utf-8") as f_in:
    for line in f_in:
        if not line.strip():
            continue
        item = json.loads(line)
        if item["sample_id"] not in done_ids:
            items.append(item)

print(f"待处理 {len(items)} 条（已跳过 {len(done_ids)} 条）...", flush=True)

f_out = open(output_file, "a", encoding="utf-8", newline="\n")
f_dbg = open(debug_file, "a", encoding="utf-8", newline="\n")
pbar = tqdm(total=len(items), desc="推理进度")

first_pass_ok = 0
total_ok = 0


def write_result(item, issues, future):
    global total_ok
    f_out.write(json.dumps({
        "sample_id": item["sample_id"],
        "issue_list": issues,
        "future_argument": future,
    }, ensure_ascii=False) + "\n")
    f_out.flush()
    total_ok += 1
    pbar.update(1)


def log_fail(item, rnd, resp):
    f_dbg.write(json.dumps({
        "sample_id": item["sample_id"], "round": rnd, "resp": resp,
    }, ensure_ascii=False) + "\n")
    f_dbg.flush()


pending = [(it, build_full_text(it)) for it in items]
for rnd in range(MAX_RETRY + 1):
    if not pending:
        break
    new_pending = []
    for b in range(0, len(pending), BATCH_SIZE):
        chunk = pending[b:b + BATCH_SIZE]
        t0 = time.time()
        texts = generate_batch([build_prompt(ft, rnd) for _, ft in chunk], rnd)
        dt = time.time() - t0
        still = []
        for (item, ft), resp in zip(chunk, texts):
            data = extract_json(resp)
            if isinstance(data, dict) and isinstance(data.get("issue_list"), list) and data["issue_list"]:
                issues, future = normalize_result(data)
                write_result(item, issues, future)
                if rnd == 0:
                    first_pass_ok += 1
            else:
                log_fail(item, rnd, resp)
                still.append((item, ft))
        print(f"[round {rnd}] batch {b // BATCH_SIZE + 1}: "
              f"成功 {len(chunk) - len(still)}/{len(chunk)}, {dt:.1f}s", flush=True)
        new_pending.extend(still)
    pending = new_pending

for item, ft in pending:
    issues, future = normalize_result(None)
    write_result(item, issues, future)

f_out.close()
f_dbg.close()
pbar.close()
denom = max(1, len(items))
print(f"✅ 完成：一次通过 {first_pass_ok}/{denom}，最终写入 {total_ok}/{denom}", flush=True)
print("结果已保存至 result.jsonl（UTF-8 无 BOM）", flush=True)