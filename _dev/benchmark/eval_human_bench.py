"""真人基准集评测：用受限解码（A-N）跑 300 题，对比真人弃牌。

支持三种模式：
  --mode jev       生产 SYSTEM + 关思考（当前 81% 方案在真人尺子下的分数）
  --mode human_lite 注入人类策略规则的 SYSTEM（A/B 85% 的规则）
  --mode base       裸规则（对照）

用法：
    python eval_human_bench.py --mode jev --out eval-human-jev.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_dev'))

from desktop_runtime import load_settings
from jev_decision import SYSTEM as JEV_SYSTEM, TOKEN_BASE

BENCH = ROOT / '_dev' / 'benchmark' / 'human-benchmark.json'

# human_lite 规则（来自 A/B 获胜 prompt 的核心，改写成单字母输出版）
HUMAN_LITE_RULES = """
【人类高手策略】
- 安全度：现物(对手牌河里打过的牌)最安全 > 已见2-3张的字牌 > 筋牌(2/5/8对两面) > 无筋1/9 > 无筋4/5/6最危险。
- 宝牌与危险牌：宝牌打出去风险高2-3级，非必要不早打。
- 攻守：和了收益(番数×点数)明显大于放铳损失时才进攻；有对手立直且我方无听牌，优先现物/筋安全牌。
- 弃和要彻底：决定防守就一直打安全牌，不要半攻半守打危险筋牌试探。
- 牌效率是基础但非唯一：早期无威胁时按进张最广打；有人立直时安全优先于进张。
- 孤张宝牌/役牌：无威胁时保留，但不要为保留它们牺牲向听。

只输出 A-N 中的一个大写字母，不输出其它任何字符。
"""


def make_client():
    s = load_settings()
    from spark_strategy import SparkClient
    return SparkClient(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['jev', 'human_lite'], default='jev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=64)
    args = ap.parse_args()

    bench = json.load(open(BENCH, encoding='utf-8'))
    cases = bench['cases']
    if args.limit:
        cases = cases[:args.limit]

    system = JEV_SYSTEM if args.mode == 'jev' else JEV_SYSTEM + '\n' + HUMAN_LITE_RULES

    client = make_client()
    api = client.settings
    import requests
    results = []
    t0 = time.time()
    for i, c in enumerate(cases):
        letters = [chr(TOKEN_BASE + j) for j in range(len(c['candidates']))]
        allowed = [TOKEN_BASE + j for j in range(len(c['candidates']))]
        payload = {
            'model': api.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'text', 'text': c['prompt']}]},
            ],
            'allowed_token_ids': allowed,
            'logprobs': True, 'top_logprobs': len(letters),
            'max_tokens': args.max_tokens, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        got = ''
        try:
            url = api.endpoint if api.endpoint.rstrip('/').endswith('/chat/completions') else api.endpoint.rstrip('/') + '/chat/completions'
            r = requests.post(url, json=payload, timeout=60,
                              headers={'Authorization': 'Bearer ' + (api.api_key or '')})
            j = r.json()
            txt = j['choices'][0]['message']['content'] or ''
            for ch in txt:
                if ch in letters:
                    got = ch
                    break
        except Exception as e:
            got = 'ERR:' + str(e)[:40]
        passed = (got == c['answer_letter'])
        if (i + 1) % 25 == 0:
            acc = sum(1 for x in results if x['passed']) / len(results)
            print('[%d/%d] acc=%.1f%% elapsed=%.0fs' % (i + 1, len(cases), acc * 100, time.time() - t0), flush=True)
        results.append({'id': c['id'], 'kind': c['kind'], 'got': got,
                        'expected': c['answer_letter'], 'expected_tile': c['answer_tile'],
                        'passed': passed, 'n_riichi': c['n_riichi']})

    acc = sum(1 for x in results if x['passed']) / len(results)
    bykind = {}
    for x in results:
        bykind.setdefault(x['kind'], [0, 0])
        bykind[x['kind']][0] += x['passed']
        bykind[x['kind']][1] += 1
    print('===== 真人基准汇总 (%s) =====' % args.mode)
    print('总: %d/%d = %.1f%%  耗时 %.0fs' % (sum(1 for x in results if x['passed']), len(results), acc * 100, time.time() - t0))
    for k, (p, n) in bykind.items():
        print('  %s: %d/%d = %.1f%%' % (k, p, n, 100 * p / n))
    json.dump({'mode': args.mode, 'accuracy': acc, 'bykind': bykind, 'results': results},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
