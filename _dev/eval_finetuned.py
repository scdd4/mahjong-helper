"""微调模型评测：把训练后的模型部署到独立端口，跑 59 场景基准对比。

用法：
    # 服务器上启动微调模型（strata 或 vLLM）
    # 然后本机：
    python _dev/eval_finetuned.py --base-url http://10.201.137.10:8081/v1 --name qwen38-4b-mj
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _dev.benchmark.run_benchmark import load_scenarios, build_payload, score
from desktop_runtime import Settings

SYSTEM_SFT = ('你是雀魂麻将决策模型。只输出一个牌名和一句≤12字理由，格式：牌名|理由。'
              '依据局面事实（安全集、立直、役牌、宝牌、点棒、巡目、向听与进张）选择最优弃牌。')


def ask(base_url, api_key, model_name, state, timeout=60):
    import urllib.request
    payload = {'model': model_name,
               'messages': [{'role': 'system', 'content': SYSTEM_SFT},
                            {'role': 'user', 'content': '局面事实：' + json.dumps(state, ensure_ascii=False)}],
               'max_tokens': 64, 'temperature': 0, 'stream': False}
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer ' + api_key,
                                          'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    text = data['choices'][0]['message']['content'] or ''
    return text, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--api-key', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    scenarios = load_scenarios()
    results = []
    for sc in scenarios:
        state = dict(sc['state'])
        # 用与训练数据对齐的格式
        state_text = {'hand': state.get('hand'), 'riichi': state.get('riichi'),
                      'rivers': state.get('rivers'), 'local_calculations':
                      state.get('local_calculations') or (
                          {'candidates': [{'tile': r['drop'], 'shanten': r['shanten'],
                                           'ukeire': r['ukeire']}
                                          for r in sorted(sc.get('solver_table', []),
                                                          key=lambda x: (x['shanten'], -x['ukeire']))[:8]]}
                          if sc.get('solver_table') else None)}
        try:
            text, elapsed = ask(args.base_url, args.api_key, args.model, state_text)
            discard = text.split('|')[0].strip()
            if sc['kind'] == 'value':
                exp = sc['expected']
                passed = discard == exp['tile'] or discard in exp.get('alternatives', [])
                detail = '期望%s 得到%s' % (exp['tile'], discard)
            else:
                answer = {'discard': discard, 'stance': '均衡', 'reason': '', 'offense': '', 'defense': ''}
                passed, detail = score(sc, answer)
        except Exception as error:
            elapsed, passed, detail, text = 0, False, 'ERROR: %s' % str(error)[:60], ''
        results.append({'id': sc['id'], 'kind': sc['kind'], 'passed': passed,
                        'seconds': round(elapsed, 2), 'text': text, 'detail': detail})
        print('[%s] %-6s %-10s %.1fs %s' %
              ('OK' if passed else 'BAD', sc['id'], sc['kind'], elapsed, detail[:60]), flush=True)

    ok = sum(r['passed'] for r in results)
    lat = [r['seconds'] for r in results if r['seconds'] > 0]
    print('\n===== %s: %d/%d (%.0f%%) median %.2fs =====' %
          (args.model, ok, len(results), 100 * ok / len(results),
           statistics.median(lat) if lat else 0), flush=True)
    out = args.out or ('_dev/benchmark/eval-%s.json' % args.model)
    Path(out).write_text(json.dumps({'model': args.model, 'passed': ok, 'total': len(results),
                                     'accuracy': ok / len(results),
                                     'median_s': statistics.median(lat) if lat else 0,
                                     'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written', out, flush=True)


if __name__ == '__main__':
    main()
