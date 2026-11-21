"""Jev-like 决策实验：封闭候选 + 截断思考 + 只输出一张牌。

对比三种配置在 59 场景上的 discard 正确率与延迟：
  off    关思考（当前 v1.9.5 生产配置）
  on-64  开思考 + max_tokens=64（截断思考）
  on-128 开思考 + max_tokens=128
  on-full 开思考 + 完整预算（基准基线）

判分只看 discard 是否落在正确集合（用户只要"打什么牌"）。
"""
import http.client
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from _dev.benchmark.run_benchmark import load_scenarios, build_payload, state_hand
from desktop_runtime import load_settings

settings = load_settings()
from urllib.parse import urlsplit
u = urlsplit(settings.endpoint)


def jev_system(state):
    choices = '/'.join(sorted(set(state['hand'])))
    return ('你是麻将决策模型。可打牌：' + choices + '。'
            '依据局面事实（安全集、立直、役牌、宝牌、点棒、巡目）独立选择一张最优牌。'
            '只输出牌名本身，不要解释、不要JSON、不要标点、不要思维过程。')


def raw_ask(payload, deadline=25):
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    t0 = time.perf_counter()
    first = None
    out = []
    conn.connect()
    conn.request('POST', u.path, json.dumps(payload).encode(),
                 {'Authorization': 'Bearer ' + settings.api_key,
                  'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    resp = conn.getresponse()
    for raw in resp:
        line = raw.decode('utf-8', 'ignore').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        for c in ev.get('choices', []):
            delta = c.get('delta', {})
            if delta.get('content'):
                if first is None:
                    first = time.perf_counter() - t0
                out.append(delta['content'])
        if time.perf_counter() - t0 > deadline:
            break
    conn.close()
    return ''.join(out).strip(), round(time.perf_counter() - t0, 2), first


def tile_ok(sc, discard):
    exp = sc['expected']
    kind = sc['kind']
    if kind in ('efficiency', 'value'):
        return discard == exp['tile'] or discard in exp.get('alternatives', [])
    if kind == 'defense':
        return discard in exp['safe'] and discard not in exp.get('must_avoid', [])
    if kind == 'push_fold':
        pool = exp['tenpai_discards'] if exp['direction'] == 'push' else exp['safe']
        return discard in pool
    return False


def main():
    scenarios = load_scenarios()
    import os
    limit = int(os.environ.get('PROBE_LIMIT', '0'))
    if limit:
        scenarios = scenarios[:limit]
    configs = [('off', False, 512), ('on-64', True, 64), ('on-128', True, 128)]
    report = {}
    for name, thinking, budget in configs:
        ok, total, lat, firsts = 0, 0, [], []
        detail = []
        for sc in scenarios:
            payload = build_payload(sc, settings)
            payload['messages'][0]['content'] = jev_system(sc['state'])
            payload['max_tokens'] = budget
            payload['reasoning_effort'] = 'low' if thinking else 'none'
            payload['chat_template_kwargs'] = {'enable_thinking': thinking}
            answer, seconds, first = raw_ask(payload)
            good = tile_ok(sc, answer)
            ok += good
            total += 1
            lat.append(seconds)
            if first:
                firsts.append(first)
            detail.append({'id': sc['id'], 'kind': sc['kind'], 'discard': answer,
                           'expected': sc['expected'].get('tile'), 'ok': good, 's': seconds})
            print('[%s] %-5s %-8s got=%-4s exp=%-4s %.1fs' %
                  ('OK' if good else 'BAD', name, sc['id'], answer[:4], str(sc['expected'].get('tile'))[:4], seconds),
                  flush=True)
        report[name] = {
            'accuracy': round(ok / total, 4), 'passed': ok, 'total': total,
            'median_s': round(statistics.median(lat), 2),
            'median_first_token_s': round(statistics.median(firsts), 2) if firsts else None,
            'detail': detail,
        }
        print('=== %s: %d/%d (%.0f%%) median %.2fs first %.2fs ===' %
              (name, ok, total, 100 * ok / total, statistics.median(lat),
               statistics.median(firsts) if firsts else 0), flush=True)
    out = HERE / 'jev-like-probe.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
