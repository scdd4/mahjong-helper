"""从真人牌谱决策样本构建基准集（修尺子：用真人实际弃牌当 gold label）。

与旧 59 题基准的区别：
  - 旧基准：38/59 题答案是手写标注，无真人支撑
  - 真人基准：所有 gold = 凤凰桌高段真人实际弃牌，唯一客观标准

评测口径：模型输出的 A-N 字母映射回牌名 == 真人弃牌 → pass。
分两类：
  - riichi 类：决策时刻场上有立直家（攻守/防守，旧基准短板）
  - early 类：无立直家（牌效率/打点）
并做去掉"位置捷径"的处理：候选顺序在 prompt 里本就是 solver 排序，
真人弃牌位置分布天然分散（已验证 A≈16%），无需额外打乱。

用法：
    python make_human_benchmark.py <decisions.jsonl> --out <bench.json> --n-riichi 200 --n-early 100
"""
import json
import sys
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_dev'))

from desktop_runtime import ExactSolver, tile_name, build_safety_facts, common_safe_tiles
from jev_decision import build_prompt, candidates_from_calculations, TOKEN_BASE

WIND_MAP = {'E': '东', 'S': '南', 'W': '西', 'N': '北'}
WIND_CN = ('东', '南', '西', '北')


def build_case(row, solver, case_id):
    hand, discard = row['hand'], row['discard']
    if len(hand) != 14 or discard not in hand:
        return None
    res = solver.analyze(hand)
    if not res:
        return None
    cands = []
    for d in (res.get('discards') or []):
        ukeire = sum(x.get('remaining', 0) for x in (d.get('ukeireTiles') or []))
        cands.append({'tile': tile_name(d['tile']),
                      'shanten': d.get('shanten'), 'ukeire': ukeire})
    calc = {'candidates': cands, 'shanten': res.get('shanten')}
    candidates = candidates_from_calculations(calc, hand, None)
    if discard not in candidates:
        return None
    riichi = row.get('riichi') or {}
    rivers_raw = row.get('rivers') or {}
    rivers = {s: [t.get('name') if isinstance(t, dict) else t for t in items]
              for s, items in rivers_raw.items()}
    round_wind = WIND_MAP.get(row.get('round_wind', 'E'), '东')
    idx = row.get('self_wind_idx')
    self_wind = WIND_CN[idx] if isinstance(idx, int) and 0 <= idx < 4 else 'right'
    facts = build_safety_facts(rivers, riichi, round_wind[:1], self_wind)
    evidence = dict(row)
    evidence['safety_facts'] = facts
    riichi_seats = [s for s, on in riichi.items() if on and s != 'own']
    if riichi_seats:
        common = common_safe_tiles(facts, riichi_seats, hand)
        if common:
            evidence['common_safe_for_all_riichi'] = common
        safe_union = set()
        for seat in riichi_seats:
            for key in ('现物', '筋', '两筋'):
                safe_union |= set((facts.get(key) or {}).get(seat) or [])
        unsafe = [t for t in hand if t not in safe_union]
        if unsafe:
            evidence['unsafe_hand_tiles'] = unsafe
    evidence['round'] = round_wind
    evidence['self_wind'] = self_wind
    evidence['remaining'] = row.get('remaining', 60)
    prompt = build_prompt(hand, calc, evidence, candidates)
    answer_letter = chr(TOKEN_BASE + candidates.index(discard))
    return {
        'id': case_id,
        'kind': 'human_riichi' if riichi_seats else 'human_early',
        'prompt': prompt,
        'answer_letter': answer_letter,
        'answer_tile': discard,
        'candidates': candidates,
        'n_riichi': len(riichi_seats),
        'gold_source': 'tenhou_houou_pro_player',
        'game_id': row.get('game_id', ''),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-riichi', type=int, default=200)
    ap.add_argument('--n-early', type=int, default=100)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.input, encoding='utf-8')]
    riichi = [r for r in rows if any(r['riichi'].values())]
    early = [r for r in rows if not any(r['riichi'].values())]
    rng.shuffle(riichi); rng.shuffle(early)
    print('pool riichi=%d early=%d' % (len(riichi), len(early)))

    solver = ExactSolver()
    cases = []
    for pool, n, prefix in [(riichi, args.n_riichi, 'hr'), (early, args.n_early, 'he')]:
        made = 0
        for row in pool:
            if made >= n:
                break
            c = build_case(row, solver, '%s-%03d' % (prefix, made + 1))
            if c:
                cases.append(c)
                made += 1
    solver.close()
    json.dump({'name': '真人基准集(天凤凤凰桌高段)', 'version': 1,
               'source': 'nikketryhard/tenhou-houou-mjai',
               'scoring': 'model A-N letter -> tile == answer_tile',
               'cases': cases},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    import collections
    print('wrote %d cases -> %s' % (len(cases), args.out))
    print('kinds:', collections.Counter(c['kind'] for c in cases))
    print('answer letter dist:', dict(sorted(collections.Counter(c['answer_letter'] for c in cases).items())))


if __name__ == '__main__':
    main()
