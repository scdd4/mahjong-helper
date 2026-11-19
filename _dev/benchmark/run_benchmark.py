#!/usr/bin/env python
"""打牌能力基准运行器。

给定完美结构化事实（无图、无识别噪声），评估模型的出牌决策质量。
场景答案来自本地 ExactSolver（效率类）与确定性安全事实（防守类），
push/fold 与价值类为标注方向 + 约束校验。

用法：
  python _dev/benchmark/run_benchmark.py --validate   # 只校验场景与求解器答案
  python _dev/benchmark/run_benchmark.py --run        # 调用真实模型评测全部场景
  python _dev/benchmark/run_benchmark.py --run --ids eff-001,def-001
  python _dev/benchmark/run_benchmark.py --run --limit 3

注意：本基准测的是「给定事实后的决策能力」，不测识图；生产混合模式
的识图质量由桌面回归测试覆盖。
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from desktop_runtime import load_settings, tile_index
from spark_strategy import SparkClient

HERE = Path(__file__).parent
SCENARIOS = HERE / 'scenarios.json'

# 与生产 STRUCT_SYSTEM_PROMPT 同一规则约束，但不附图（本基准喂的是完美事实）。
BENCH_SYSTEM_PROMPT = '''你是四人立直麻将策略教练。输入是结构化局面JSON（本地程序核验过的事实）。
只输出一个紧凑JSON对象，不要思维过程、不要Markdown。字段全部一句话且尽量短：
{"stance":"进攻/防守/均衡","discard":"牌名或null","reason":"≤16字","offense":"≤12字","defense":"≤12字","next":"≤12字","dora":"≤10字"}。
事实约束：hand/melds/dora_indicators是核验事实，直接采用，不要改写。
暗牌张数+副露张数=14时必须打牌且牌必须在hand中；=13时discard必须为null，等待摸牌。
牌名m/p/s/z，0=赤五。有明副露（吃/碰/明杠）的门清已破坏，不能立直；暗杠不算副露仍可立直。
防守时现物最安全，其次是筋与客风/客三元字牌；无筋中张数牌危险。
ukeire（若提供）未扣除牌河可见牌，只是参考上限，可结合牌河定性取舍。'''


def load_scenarios():
    data = json.loads(SCENARIOS.read_text(encoding='utf-8'))
    return data['scenarios']


def validate(scenarios):
    """物理与答案自检；效率类期望与求解器逐场景核对。"""
    from desktop_runtime import ExactSolver
    solver = ExactSolver()
    problems = []
    for sc in scenarios:
        state = sc['state']
        counts = {}
        piles = [state['hand']]
        for seat in ('own', 'right', 'across', 'left'):
            for group in (state.get('melds', {}).get(seat) or []):
                piles.append(group['tiles'] if isinstance(group, dict) else group)
            piles.append(state.get('rivers', {}).get(seat) or [])
        piles.append(state.get('dora_indicators', []))
        for pile in piles:
            for name in pile:
                counts[name] = counts.get(name, 0) + 1
        overflow = {k: v for k, v in counts.items() if v > 4}
        if overflow:
            problems.append(f"{sc['id']}: 物理超编 {overflow}")
        if state.get('drawn'):
            problems.append(f"{sc['id']}: 应使用14张hand，而非hand+drawn")
        if sc['kind'] == 'efficiency':
            seen = []
            for seat in ('own', 'right', 'across', 'left'):
                seen += state.get('rivers', {}).get(seat) or []
            seen += state.get('dora_indicators', [])
            expected = sc['expected']
            table = {row['drop']: (row['shanten'], row['ukeire']) for row in sc['solver_table']}
            best = min(table.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
            if best[0] != expected['tile'] or best[0] not in [expected['tile']] + expected['alternatives']:
                problems.append(f"{sc['id']}: 期望 {expected['tile']} 与求解器最优 {best[0]} 不符")
        try:
            for name in state['hand']:
                tile_index(name)
        except (ValueError, TypeError):
            problems.append(f"{sc['id']}: 手牌含非法牌名")
    solver.close()
    return problems


def build_payload(sc, settings):
    state = dict(sc['state'])
    user_text = '当前局面（本地程序核验与规则核算）：' + json.dumps(state, ensure_ascii=False)
    payload = {'model': settings.model,
               'messages': [{'role': 'system', 'content': BENCH_SYSTEM_PROMPT},
                            {'role': 'user', 'content': user_text}],
               'stream': True, 'max_tokens': 300, 'reasoning_effort': 'low',
               'chat_template_kwargs': {'enable_thinking': True}, 'strata_checkpoint': False}
    return payload


def score(sc, answer):
    """返回 (passed, detail)。答案缺字段按失败计。"""
    discard = answer.get('discard')
    stance = answer.get('stance') or ''
    reason = str(answer.get('reason', '')) + str(answer.get('offense', '')) + str(answer.get('defense', ''))
    exp = sc['expected']
    if sc['kind'] == 'efficiency':
        if not discard:
            return False, '未给出discard'
        if discard not in state_hand(sc):
            return False, f'discard {discard} 不在手牌内'
        ok = discard == exp['tile'] or discard in exp['alternatives']
        detail = f"期望 {exp['tile']}(备选 {exp['alternatives']})，模型 {discard}"
        return ok, detail
    if sc['kind'] == 'defense':
        if not discard:
            return False, '未给出discard'
        hit_avoid = discard in exp['must_avoid']
        safe = discard in exp['safe']
        detail = f"安全集 {exp['safe']}，模型 {discard}"
        return safe and not hit_avoid, detail
    if sc['kind'] == 'push_fold':
        if exp['direction'] == 'fold':
            safe = discard in exp['safe'] if discard else False
            stance_ok = stance == exp['stance']
            detail = f"期望 {exp['direction']}({exp['stance']}，安全集 {exp['safe']})，模型 {stance}/{discard}"
            return safe and stance_ok, detail
        keep = exp['tenpai_discards']
        push_ok = discard in keep if discard else False
        stance_ok = stance == exp['stance']
        detail = f"期望 {exp['direction']}({exp['stance']}，保持听牌弃牌 {keep})，模型 {stance}/{discard}"
        return push_ok and stance_ok, detail
    if sc['kind'] == 'value':
        ok = discard == exp['tile'] or discard in exp['alternatives']
        keywords = [k for k in exp.get('rationale_keywords', []) if k in reason]
        detail = f"期望 {exp['tile']}(备选 {exp['alternatives']})，模型 {discard}；理由关键词 {keywords}"
        return ok, detail
    return False, '未知场景类型'


def state_hand(sc):
    return sc['state']['hand']


def run(scenarios, ids, limit, settings):
    client = SparkClient(settings)
    selected = [s for s in scenarios if not ids or s['id'] in ids][:limit or None]
    results = []
    for sc in selected:
        payload = build_payload(sc, settings)
        started = time.perf_counter()
        try:
            reply = client._request_with_retry(payload, None, parse_mode='fast')
            answer = reply['report']
            elapsed = time.perf_counter() - started
            passed, detail = score(sc, answer)
            results.append({'id': sc['id'], 'category': sc['category'], 'difficulty': sc['difficulty'],
                            'passed': passed, 'seconds': round(elapsed, 1), 'detail': detail,
                            'answer': answer, 'expected': sc['expected']})
            mark = 'PASS' if passed else 'FAIL'
            print(f"[{mark}] {sc['id']} ({elapsed:.1f}s) {detail}")
        except Exception as error:
            elapsed = time.perf_counter() - started
            results.append({'id': sc['id'], 'category': sc['category'], 'difficulty': sc['difficulty'],
                            'passed': False, 'seconds': round(elapsed, 1),
                            'detail': f'调用失败: {type(error).__name__}: {error}', 'answer': None})
            print(f"[ERR ] {sc['id']} ({elapsed:.1f}s) {type(error).__name__}: {error}")
    return results


def report(results):
    total = [r for r in results if 'passed' in r and not str(r['detail']).startswith('调用失败')]
    errored = [r for r in results if str(r['detail']).startswith('调用失败')]
    passed = [r for r in total if r['passed']]
    print('\n===== 汇总 =====')
    print(f"有效场景 {len(total)}，通过 {len(passed)}，准确率 {len(passed)/max(1,len(total))*100:.0f}%"
          + (f"，调用失败 {len(errored)}" if errored else ''))
    by_cat = {}
    for r in total:
        by_cat.setdefault(r['category'], []).append(r)
    for cat, rows in sorted(by_cat.items()):
        ok = sum(1 for r in rows if r['passed'])
        lat = statistics.median(r['seconds'] for r in rows)
        print(f"  {cat:12} {ok}/{len(rows)} ({ok/len(rows)*100:.0f}%)  中位耗时 {lat:.1f}s")
    out = HERE / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({'results': results}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'明细已写入 {out}')


def main():
    parser = argparse.ArgumentParser(description='打牌能力基准')
    parser.add_argument('--validate', action='store_true', help='只校验场景，不调用模型')
    parser.add_argument('--run', action='store_true', help='调用真实模型评测')
    parser.add_argument('--ids', help='逗号分隔的场景 id')
    parser.add_argument('--limit', type=int, help='只跑前 N 个')
    args = parser.parse_args()
    scenarios = load_scenarios()
    ids = set(args.ids.split(',')) if args.ids else None
    if args.validate or not args.run:
        problems = validate(scenarios)
        print(f'校验 {len(scenarios)} 场景：' + ('全部通过' if not problems else f'{len(problems)} 个问题'))
        for p in problems:
            print('  ', p)
        if problems:
            sys.exit(1)
    if args.run:
        settings = load_settings()
        if not settings.api_key:
            print('尚未配置 Spark API 密钥')
            sys.exit(1)
        results = run(scenarios, ids, args.limit, settings)
        report(results)


if __name__ == '__main__':
    main()
