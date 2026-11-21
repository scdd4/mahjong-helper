"""Jev-like vs 当前 fast 模式：59 场景准确率/延迟/置信度校准对比。

复用 run_benchmark 的场景与评分，但改用 jev_decision 的受限 token 决策。
"""
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from _dev.benchmark.run_benchmark import load_scenarios, score, state_hand, build_payload
from desktop_runtime import load_settings
from spark_strategy import SparkClient
from jev_decision import candidates_from_calculations, build_prompt, parse_logprobs

settings = load_settings()


def run_jev(sc, client):
    """把场景转成 jev 决策并调用。"""
    # 复用基准的 build_payload，拿到与生产同口径的完整 state（含 safety_facts、
    # unsafe_hand_tiles、common_safe、dora_tiles、yakuhai_tiles 等注入事实）。
    full_payload = build_payload(sc, settings)
    state = json.loads(full_payload['messages'][1]['content'].split('：', 1)[1])
    hand = list(state['hand'])
    verified = hand
    calc = state.get('local_calculations') or {}
    candidates = candidates_from_calculations(calc, verified, state)
    if not candidates:
        candidates = list(dict.fromkeys(hand))[:14]
    evidence = dict(state)
    evidence['hand'] = {'reliable': True, 'tiles': [{'name': t} for t in hand]}
    evidence['warnings'] = []
    user_text = build_prompt(verified, calc, evidence, candidates)
    from jev_decision import SYSTEM, TOKEN_FIRST_ID, MAX_CHOICES
    payload = {'model': settings.model,
               'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': [{'type': 'text', 'text': user_text}]}],
               'stream': False, 'max_tokens': 64,
               'logprobs': True, 'top_logprobs': MAX_CHOICES,
               'allowed_token_ids': [TOKEN_FIRST_ID + i for i in range(len(candidates))],
               'reasoning_effort': 'none',
               'chat_template_kwargs': {'enable_thinking': False}, 'strata_checkpoint': False}
    from PIL import Image
    with Image.new('RGB', (32, 32)) as image:
        t0 = time.perf_counter()
        reply = client._jev_request(payload, None)
        tile, confidence, probs = parse_logprobs(reply['choices'][0], candidates)
        elapsed = time.perf_counter() - t0
    return tile, confidence, elapsed


def main():
    scenarios = load_scenarios()
    import os
    limit = int(os.environ.get('BENCH_LIMIT', '0'))
    if limit:
        scenarios = scenarios[:limit]
    client = SparkClient(settings)
    results = []
    for sc in scenarios:
        try:
            tile, conf, elapsed = run_jev(sc, client)
            # 用基准评分：构造 answer 契约
            if sc['kind'] == 'value':
                # Jev 只输出牌，不输出理由；value 类只判牌是否正确，不判理由关键词。
                exp = sc['expected']
                passed = tile == exp['tile'] or tile in exp.get('alternatives', [])
                detail = '期望 %s，模型 %s（Jev 不输出理由）' % (exp['tile'], tile)
            else:
                answer = {'discard': tile, 'stance': '均衡', 'reason': 'jev', 'offense': '', 'defense': ''}
                passed, detail = score(sc, answer)
        except Exception as error:
            tile, conf, elapsed, passed, detail = None, 0, 0, False, 'ERROR: %s' % str(error)[:80]
        results.append({'id': sc['id'], 'kind': sc['kind'], 'category': sc.get('category'),
                        'passed': passed, 'seconds': round(elapsed, 2),
                        'confidence': round(conf, 3), 'tile': tile, 'detail': detail})
        mark = 'PASS' if passed else 'FAIL'
        print('[%s] %-6s %-8s tile=%-4s conf=%.2f %.1fs %s' %
              (mark, sc['id'], sc['kind'], str(tile)[:4], conf, elapsed, detail[:60]), flush=True)
    ok = sum(1 for r in results if r['passed'])
    lat = [r['seconds'] for r in results if r['seconds'] > 0]
    print('\n===== Jev-like: %d/%d (%.0f%%) median %.2fs =====' %
          (ok, len(results), 100 * ok / len(results), statistics.median(lat) if lat else 0))
    # 置信度校准：分桶统计正确率
    bins = {'high': [], 'medium': [], 'low': []}
    for r in results:
        c = r['confidence']
        b = 'high' if c >= 0.6 else ('medium' if c >= 0.3 else 'low')
        bins[b].append(r['passed'])
    print('置信度校准：')
    for b, vals in bins.items():
        if vals:
            print('  %-7s n=%2d 准确率 %.0f%%' % (b, len(vals), 100 * sum(vals) / len(vals)))
    out = HERE / 'jev-benchmark.json'
    out.write_text(json.dumps({'summary': {'passed': ok, 'total': len(results),
                                           'accuracy': ok / len(results),
                                           'median_s': statistics.median(lat) if lat else 0},
                               'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
