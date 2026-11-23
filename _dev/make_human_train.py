"""从真人牌谱决策样本生成微调数据（只学第四层：给定结构化事实选 A-N）。

训练/推理同分布保证：
  - prompt 完全复用 jev_decision.build_prompt
  - calculations 完全复用 candidates_from_calculations + ExactSolver
  - evidence 安全事实完全复刻 spark_strategy.decide_jev 的现场构造
  - 标签 = 真人实际弃牌映射的 A-N 字母（不是任何规则）

用法：
    python make_human_train.py <decisions.jsonl> --out <train.jsonl> --max 20000
"""
import json
import sys
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_dev'))

from desktop_runtime import (ExactSolver, tile_name, build_safety_facts,
                             common_safe_tiles, dora_from_indicator)
from jev_decision import build_prompt, candidates_from_calculations, SYSTEM, TOKEN_BASE

# 场风/自风：MJAI 的 bakaze (E/S/W) 直接是轮 wind；自风按亲家推定
WIND_MAP = {'E': '东', 'S': '南', 'W': '西', 'N': '北'}
SEAT_WIND = ['东', '南', '西', '北']  # 0=亲家东


def make_example(row, solver):
    """把一个真人决策样本转成 (prompt, answer_letter)。失败返回 None。"""
    hand = row['hand']
    discard = row['discard']
    if len(hand) != 14 or discard not in hand:
        return None

    res = solver.analyze(hand)
    if not res:
        return None
    # solver tile index -> 项目牌名，计算进张总数
    cands = []
    for d in (res.get('discards') or []):
        ukeire = sum(x.get('remaining', 0) for x in (d.get('ukeireTiles') or []))
        cands.append({'tile': tile_name(d['tile']),
                      'shanten': d.get('shanten'), 'ukeire': ukeire})
    calc = {'candidates': cands, 'shanten': res.get('shanten')}
    candidates = candidates_from_calculations(calc, hand, None)
    if discard not in candidates:
        return None

    # 复刻 decide_jev 的 evidence 现场构造
    riichi = row.get('riichi') or {}
    rivers_raw = row.get('rivers') or {}
    rivers = {seat: [t.get('name') if isinstance(t, dict) else t for t in items]
              for seat, items in rivers_raw.items()}
    # 真宝牌/役牌（生产 build_prompt 从 evidence 读 dora_tiles/yakuhai_tiles）
    dora_ind = row.get('dora_indicators') or []
    try:
        dora_tiles = [dora_from_indicator(d) for d in dora_ind]
    except Exception:
        dora_tiles = []
    yakuhai = [t for t in hand if t in ('1z', '2z', '3z', '4z', '5z', '6z', '7z')]

    round_wind = WIND_MAP.get(row.get('round_wind', 'E'), '东')
    # 自风：解析器已按亲家相对位置算出索引（0=东/亲 1=南 2=西 3=北）
    WIND_CN = ('东', '南', '西', '北')
    idx = row.get('self_wind_idx')
    self_wind = WIND_CN[idx] if isinstance(idx, int) and 0 <= idx < 4 else (row.get('self_wind') or 'right')

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
    evidence['dora_tiles'] = dora_tiles
    evidence['yakuhai_tiles'] = yakuhai
    evidence['round'] = round_wind
    evidence['self_wind'] = self_wind
    # 余牌：MJAI 未直接给，用固定保守值（生产也是从 read_scene 取，对决策影响小）
    evidence['remaining'] = row.get('remaining', 60)

    prompt = build_prompt(hand, calc, evidence, candidates)
    letter = chr(TOKEN_BASE + candidates.index(discard))
    return {'prompt': prompt, 'letter': letter, 'discard': discard,
            'candidates': candidates, 'n_riichi': len(riichi_seats),
            'game_id': row.get('game_id', ''),
            'kind': 'human_riichi' if riichi_seats else 'human_early'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=20000)
    ap.add_argument('--riichi-ratio', type=float, default=0.0, help='含立直家样本目标占比（0=不平衡）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.input, encoding='utf-8')]
    rng.shuffle(rows)

    # 攻守/防守样本天然稀缺（含立直家仅约5%），oversample 到目标比例，
    # 否则模型几乎学不到"有人立直时该怎么办"——这正是旧基准最弱的类别。
    if getattr(args, 'riichi_ratio', 0) > 0:
        riichi_rows = [r for r in rows if any(r['riichi'].values())]
        early_rows = [r for r in rows if not any(r['riichi'].values())]
        rng.shuffle(riichi_rows); rng.shuffle(early_rows)
        target_riichi = int(args.max * args.riichi_ratio)
        # 含立直家样本可能不够，重复采样补足
        if len(riichi_rows) < target_riichi:
            riichi_rows = (riichi_rows * (target_riichi // max(len(riichi_rows), 1) + 1))[:target_riichi]
        rows = riichi_rows[:target_riichi] + early_rows[:args.max - target_riichi]
        rng.shuffle(rows)
        print('平衡采样：riichi=%d early=%d' % (len(riichi_rows[:target_riichi]), len(early_rows[:args.max - target_riichi])), flush=True)

    solver = ExactSolver()
    out = open(args.out, 'w', encoding='utf-8')
    n = 0
    kinds = {}
    letters = {}
    for row in rows:
        if n >= args.max:
            break
        ex = make_example(row, solver)
        if not ex:
            continue
        out.write(json.dumps(ex, ensure_ascii=False) + '\n')
        n += 1
        kinds[ex['kind']] = kinds.get(ex['kind'], 0) + 1
        letters[ex['letter']] = letters.get(ex['letter'], 0) + 1
    out.close()
    solver.close()
    print('wrote %d examples -> %s' % (n, args.out))
    print('kinds:', kinds)
    print('letters:', dict(sorted(letters.items())))


if __name__ == '__main__':
    main()
