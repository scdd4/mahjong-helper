"""在服务器上评测微调模型（transformers 直接加载，无需 8080 服务）。

真人基准：300 题，受限解码 A-N，对比真人弃牌。
同时可跑旧 59 题基准做双轨对比。

用法（服务器）：
    python eval_human_model.py --model /home/user/mahjong_train/out-human-4b/merged \
        --bench /home/user/mahjong_train/data/human-benchmark.json \
        --out /home/user/mahjong_train/eval-human-model.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

TOKEN_BASE = 65  # 'A'


class AllowedTokensProcessor(LogitsProcessor):
    """受限解码：只允许候选 A-N 对应的 token id（与生产 allowed_token_ids 一致）。"""

    def __init__(self, allowed):
        self.allowed = allowed

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed] = 0
        return scores + mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bench', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    bench = json.load(open(args.bench, encoding='utf-8'))
    cases = bench['cases']
    if args.limit:
        cases = cases[:args.limit]

    print('加载模型 %s ...' % args.model, flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto')
    model.eval()
    print('模型加载 %.1fs' % (time.time() - t0), flush=True)

    # 找 A-N 的 token id
    letters = [chr(TOKEN_BASE + i) for i in range(14)]
    token_ids = []
    for ch in letters:
        ids = tok.encode(ch, add_special_tokens=False)
        token_ids.append(ids[0] if ids else -1)
    print('A-N token ids:', token_ids[:5], '...', flush=True)

    results = []
    t0 = time.time()
    for i, c in enumerate(cases):
        n = len(c['candidates'])
        allowed = token_ids[:n]
        prompt = c['prompt']
        # 用与训练一致的 chat template（system + user），关思考
        messages = [
            {'role': 'system', 'content': '你是四人立直麻将决策模型。用户给出结构化局面事实与编号候选，你只输出一个字母代表选择。\n【铁律】只输出 A-N 中的一个大写字母，不输出牌名、理由、JSON、标点或任何其它字符。\n依据局面事实（安全集、立直家、役牌、宝牌、点棒、巡目、向听与进张）独立选择最优弃牌：\n有对手立直且我方无听牌威胁时优先安全牌；末巡落后需点数时保持进攻；牌效率作为基础参考而非唯一标准。\n候选含义见输入，字母与牌的对应绝不改变。'},
            {'role': 'user', 'content': prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        inputs = tok(text, return_tensors='pt').to(model.device)
        processor_list = LogitsProcessorList([AllowedTokensProcessor(allowed)])
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                logits_processor=processor_list,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        # 取第一个新 token
        new_id = out[0][inputs['input_ids'].shape[1]].item()
        got = chr(TOKEN_BASE + allowed.index(new_id)) if new_id in allowed else '?'
        passed = (got == c['answer_letter'])
        if (i + 1) % 25 == 0:
            acc = sum(1 for x in results if x['passed']) / len(results)
            print('[%d/%d] acc=%.1f%% %.0fs' % (i + 1, len(cases), acc * 100, time.time() - t0), flush=True)
        results.append({'id': c['id'], 'kind': c['kind'], 'got': got,
                        'expected': c['answer_letter'], 'passed': passed})

    acc = sum(1 for x in results if x['passed']) / len(results)
    bykind = {}
    for x in results:
        bykind.setdefault(x['kind'], [0, 0])
        bykind[x['kind']][0] += x['passed']
        bykind[x['kind']][1] += 1
    print('===== 真人基准汇总 =====')
    print('总: %d/%d = %.1f%%' % (sum(1 for x in results if x['passed']), len(results), acc * 100))
    for k, (p, n) in bykind.items():
        print('  %s: %d/%d = %.1f%%' % (k, p, n, 100 * p / n))
    json.dump({'accuracy': acc, 'bykind': bykind, 'results': results},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
