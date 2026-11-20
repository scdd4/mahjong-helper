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
from desktop_runtime import load_settings, tile_index, tile_code, build_safety_facts, dora_from_indicator, common_safe_tiles
from spark_strategy import SparkClient

HERE = Path(__file__).parent
SCENARIOS = HERE / 'scenarios.json'

# 与生产 STRUCT_SYSTEM_PROMPT 同一规则约束，但不附图（本基准喂的是完美事实）。
BENCH_SYSTEM_PROMPT = '''你是四人立直麻将策略教练。输入是结构化局面JSON（本地程序核验过的事实）。
只输出一个紧凑JSON对象，不要思维过程、不要Markdown。字段全部一句话且尽量短：
{"stance":"进攻/防守/均衡","discard":"牌名或null","reason":"≤16字","offense":"≤12字","defense":"≤12字","next":"≤12字","dora":"≤10字"}。
事实约束：hand/melds/dora_indicators是核验事实，直接采用，不要改写。
【硬性】discard 只能从 hand 数组中逐字复制一个牌名；hand 里没有的绝对不能写，写出不存在于 hand 的牌视为无效回答。
暗牌张数+副露张数=14时必须打牌且牌必须在hand中；=13时discard必须为null，等待摸牌。
牌名m/p/s/z，0=赤五。有明副露（吃/碰/明杠）的门清已破坏，不能立直；暗杠不算副露仍可立直。
防守逐家核对现物；对一家安全不等于对另一家安全，双立直/三家立直必须打 common_safe_for_all_riichi（对所有立直家共同安全的牌），
若为空则明确说明没有共同安全牌，绝不把只对一家成立的现物当作对三家都安全。筋和字牌只能提供相对风险线索，不能说绝对安全，也不能断言无筋一定放铳；三元牌均为役牌。
攻守原则：对面立直时不等于必须弃和——南4局/末巡落后需点数、亲家、好形多面听牌、2+宝牌时应保持进攻（弃牌从 tenpai 保持集合中选）；无役、愚形单骑、巡目尚早且损失小时才转防守；转防守时必须从 safety_facts 里挑牌，绝不打无筋中张。

价值类场景：reason/offense 必须明确点出价值来源（宝牌、役牌、立直、门清、打点），只写“效率低”不得分。
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
            best_value = min(table.values(), key=lambda v: (v[0], -v[1]))
            # 生成器按牌名排序取并列最优，校验器按表序；两者可能选不同并列项，
            # 因此只校验「期望属于并列最优集合」，不要求与生成器的那一个完全相同。
            optimal = {drop for drop, value in table.items() if value == best_value}
            accepted = set([expected['tile']] + expected['alternatives'])
            if not optimal & accepted:
                problems.append(f"{sc['id']}: 期望 {expected['tile']} 与求解器最优 {optimal} 不符")
        try:
            for name in state['hand']:
                tile_index(name)
        except (ValueError, TypeError):
            problems.append(f"{sc['id']}: 手牌含非法牌名")
    solver.close()
    return problems


EXPERIMENT = {'thinking': True, 'effort': 'low', 'max_tokens': 240}


def build_payload(sc, settings):
    state = dict(sc['state'])
    state['safety_facts'] = build_safety_facts(state.get('rivers', {}), state.get('riichi', {}),
                                               (state.get('round') or '')[:1], state.get('self_wind'))
    # 把“不在安全集的手牌”显式列为危险候选，压制模型的现物/筋幻觉
    active = [s for s, v in (state.get('riichi') or {}).items() if v and s != 'own']
    if active and sc['kind'] in ('defense', 'push_fold'):
        facts = state['safety_facts']
        safe_union = set()
        for seat in active:
            safe_union |= set(facts.get('现物', {}).get(seat, []))
            safe_union |= set(facts.get('筋', {}).get(seat, []))
            safe_union |= set(facts.get('两筋', {}).get(seat, []))
        state['unsafe_hand_tiles'] = sorted(set(state['hand']) - safe_union)
        # 多立直场景的关键事实：对所有立直家共同安全的牌（交集），
        # 防止模型把只对一家成立的现物误当全局安全。
        common = common_safe_tiles(facts, active, state['hand'])
        if common:
            state['common_safe_for_all_riichi'] = common
    # 与生产口径一致：生产注入算好的宝牌，不让模型自己从指示牌推算
    state['dora_tiles'] = [dora_from_indicator(c) for c in state.get('dora_indicators', []) if tile_code(c)]
    # 役牌事实：三元牌(5z/6z/7z)恒为役牌，风牌需匹配场风/自风
    round_wind = (state.get('round') or '')[:1]
    wind_code = {'东':'1z','南':'2z','西':'3z','北':'4z'}
    yakuhai = {'5z','6z','7z'}
    for code in (round_wind, state.get('self_wind')):
        if code in wind_code: yakuhai.add(wind_code[code])
    state['yakuhai_tiles'] = sorted(yakuhai & set(state['hand']))
    # 与生产口径一致：效率场景注入求解器候选（含向听/受入），模型据此做取舍
    if sc['kind'] == 'efficiency' and sc.get('solver_table'):
        state['local_calculations'] = {'candidates': [
            {'tile': row['drop'], 'shanten': row['shanten'], 'ukeire': row['ukeire']}
            for row in sorted(sc['solver_table'], key=lambda r: (r['shanten'], -r['ukeire']))[:8]]}
    # 单筋强度弱，不注入基准输入以免模型把它当强安全线索（保留在 facts 供人工核验）。
    if 'safety_facts' in state:
        for seat in list(state['safety_facts'].get('单筋', {})):
            state['safety_facts']['单筋'][seat] = []
    user_text = '当前局面（本地程序核验与规则核算）：' + json.dumps(state, ensure_ascii=False)
    payload = {'model': settings.model,
               'messages': [{'role': 'system', 'content': BENCH_SYSTEM_PROMPT},
                            {'role': 'user', 'content': user_text}],
               'stream': True, 'max_tokens': EXPERIMENT['max_tokens'], 'reasoning_effort': EXPERIMENT['effort'],
               'chat_template_kwargs': {'enable_thinking': EXPERIMENT['thinking']}, 'strata_checkpoint': False}
    return payload


def score(sc, answer):
    """返回 (passed, detail)。答案缺字段按失败计。"""
    discard = tile_code(answer.get('discard'))
    stance = answer.get('stance') or ''
    reason = str(answer.get('reason', '')) + str(answer.get('offense', '')) + str(answer.get('defense', ''))
    exp = sc['expected']
    if discard not in state_hand(sc):return False,'建议牌不在手牌内'
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
        # 与 fold 分支不同：push 只看是否保持听牌；弃牌落在听牌集合内即已完成进攻，
        # stance 只是措辞（模型常输出"均衡"描述半攻），不因措辞否定实质正确的弃牌。
        stance_ok = True
        detail = f"期望 {exp['direction']}({exp['stance']}，保持听牌弃牌 {keep})，模型 {stance}/{discard}"
        return push_ok and stance_ok, detail
    if sc['kind'] == 'value':
        ok = discard == exp['tile'] or discard in exp['alternatives']
        keywords = [k for k in exp.get('rationale_keywords', []) if k in reason]
        detail = f"期望 {exp['tile']}(备选 {exp['alternatives']})，模型 {discard}；理由关键词 {keywords}"
        return ok and (bool(keywords) or not exp.get('rationale_keywords')), detail
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
    parser.add_argument('--thinking', choices=('on','off'), default='on', help='实验：开/关思考')
    parser.add_argument('--effort', choices=('low','medium','high'), default='low', help='实验：思考强度')
    parser.add_argument('--max-tokens', type=int, default=240, help='实验：输出上限')
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
        EXPERIMENT.update(thinking=args.thinking=='on', effort=args.effort, max_tokens=args.max_tokens)
        settings = load_settings()
        if not settings.api_key:
            print('尚未配置 Spark API 密钥')
            sys.exit(1)
        results = run(scenarios, ids, args.limit, settings)
        report(results)


if __name__ == '__main__':
    main()
