"""生成雀魂麻将微调训练数据集（5 万条可验证样本）。

设计原则：
- 答案由 ExactSolver + 安全规则引擎生成，可验证、无主观标注噪声
- 输入格式与生产 Jev prompt 完全对齐
- 覆盖 8 类决策场景，难度分层
- 输出 "牌名|≤12字理由"，保证快速响应

用法：
    python _dev/gen_training_data.py --total 50000 --out _dev/train_data
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_runtime import ExactSolver, build_safety_facts, common_safe_tiles, tile_index, tile_code

TILES = [str(n) + s for s in 'mps' for n in range(1, 10)] + [str(n) + 'z' for n in range(1, 8)]
HONORS = {str(n) + 'z' for n in range(1, 8)}
SEATS = ('own', 'right', 'across', 'left')
SYSTEM = ('你是雀魂麻将决策模型。只输出一个牌名和一句≤12字理由，格式：牌名|理由。'
          '依据局面事实（安全集、立直、役牌、宝牌、点棒、巡目、向听与进张）选择最优弃牌。')


def random_hand(rng, n=14):
    """随机合法手牌（每种牌≤4张）。"""
    bag = []
    for tile in TILES:
        bag.extend([tile] * 4)
    rng.shuffle(bag)
    return bag[:n]


def code_of(idx):
    """ExactSolver 返回 tile index (0-33)，转 m/p/s/z 牌名。"""
    if idx < 27:
        return str(idx % 9 + 1) + 'mps'[idx // 9]
    return str(idx - 26) + 'z'


def solver_candidates(solver, hand):
    """ExactSolver 核算结果，返回 (shanten, 候选列表)。"""
    try:
        result = solver.analyze(hand)
    except (ValueError, Exception):
        return None, []
    candidates = []
    for d in result.get('discards', [])[:8]:
        candidates.append({'tile': code_of(d['tile']), 'shanten': d['shanten'],
                           'ukeire': d.get('ukeireTotal', 0), 'kinds': d.get('ukeireKinds', 0),
                           'tenpai': d.get('tenpai', False)})
    return result, candidates


def make_rivers(rng, hand, n_riichi=0):
    """构造牌河与立直状态，保证现物来自已打出的牌。"""
    bag = [t for t in TILES if t not in hand] * 4
    rng.shuffle(bag)
    rivers = {s: [] for s in SEATS if s != 'own'}
    riichi = {s: False for s in SEATS}
    idx = 0
    for seat in SEATS[1:]:
        count = rng.randint(3, 12)
        rivers[seat] = bag[idx:idx + count]
        idx += count
    riichi_seats = rng.sample(SEATS[1:], n_riichi) if n_riichi else []
    for seat in riichi_seats:
        riichi[seat] = True
    return rivers, riichi, riichi_seats


def format_prompt(hand, shanten, candidates, rivers, riichi, safety, common_safe,
                  dora=None, round_wind=None, self_wind=None, scores=None, turn=None):
    """与生产 Jev build_prompt 对齐的输入格式。"""
    lines = ['手牌(%d张)：%s' % (len(hand), ' '.join(hand))]
    if shanten is not None:
        lines.append('当前向听：%s' % shanten)
    for i, c in enumerate(candidates):
        letter = chr(ord('A') + i)
        note = []
        if c.get('shanten') is not None:
            note.append('%s向听' % c['shanten'])
        if c.get('ukeire'):
            note.append('%s张进张' % c['ukeire'])
        if c.get('tenpai'):
            note.append('听牌')
        lines.append('%s=%s（%s）' % (letter, c['tile'], '，'.join(note)) if note else '%s=%s' % (letter, c['tile']))
    for seat, on in riichi.items():
        if on and seat != 'own':
            lines.append('%s立直' % seat)
    for seat in SEATS[1:]:
        if rivers.get(seat):
            lines.append('%s牌河：%s' % (seat, ' '.join(rivers[seat][:12])))
    if common_safe:
        lines.append('对所有立直家共同安全：' + ' '.join(common_safe))
    for kind in ('现物', '筋'):
        for seat, tiles in (safety.get(kind) or {}).items():
            if tiles:
                lines.append('%s对%s：%s' % (kind, seat, ' '.join(tiles[:8])))
    if dora:
        lines.append('宝牌指示牌：' + ' '.join(dora))
    if round_wind:
        lines.append('场风：%s' % round_wind)
    if self_wind:
        lines.append('自风：%s' % self_wind)
    if scores:
        lines.append('点棒：' + ' '.join('%s=%s' % (k, v) for k, v in scores.items()))
    if turn:
        lines.append('巡目：%s' % turn)
    lines.append('轮到你出牌。只输出牌名|理由。')
    return '\n'.join(lines)


def make_record(prompt, discard, reason):
    return {'messages': [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': '%s|%s' % (discard, reason)}]}


def gen_efficiency(rng, solver, n):
    """牌效率：ExactSolver 最优弃牌。"""
    records = []
    for _ in range(n):
        hand = random_hand(rng)
        result, cands = solver_candidates(solver, hand)
        if not cands or not result:
            continue
        best = cands[0]
        rivers, riichi, _ = make_rivers(rng, hand)
        if best['tile'] not in hand:
            continue
        prompt = format_prompt(hand, result['shanten'], cands, rivers, riichi, {}, [])
        reason = '进张%s张最广' % best['ukeire'] if best.get('ukeire') else '牌效率最优'
        records.append(make_record(prompt, best['tile'], reason))
    return records


def gen_defense(rng, solver, n):
    """防守：有立直时打现物/共同安全牌。"""
    records = []
    for _ in range(n):
        hand = random_hand(rng)
        n_riichi = rng.choice([1, 1, 2, 3])
        rivers, riichi, riichi_seats = make_rivers(rng, hand, n_riichi)
        safety = build_safety_facts(rivers, riichi, '东', '西')
        common = common_safe_tiles(safety, riichi_seats, hand)
        # 优先共同安全 → 单个现物 → 筋
        safe_in_hand = [t for t in common if t in hand]
        if not safe_in_hand:
            for seat in riichi_seats:
                genbutsu = [t for t in (safety.get('现物', {}).get(seat) or []) if t in hand]
                if genbutsu:
                    safe_in_hand = genbutsu
                    break
        if not safe_in_hand:
            for seat in riichi_seats:
                suji = [t for t in (safety.get('筋', {}).get(seat) or []) if t in hand]
                if suji:
                    safe_in_hand = suji
                    break
        if not safe_in_hand:
            continue
        result, cands = solver_candidates(solver, hand)
        discard = rng.choice(safe_in_hand)
        reason = '共同安全牌' if discard in common else '现物安全'
        prompt = format_prompt(hand, result['shanten'] if result else None, cands,
                               rivers, riichi, safety, common)
        records.append(make_record(prompt, discard, reason))
    return records


def gen_push_fold(rng, solver, n):
    """攻守：末巡落后/听牌好形进攻，否则弃和防守。"""
    records = []
    for _ in range(n):
        hand = random_hand(rng)
        result, cands = solver_candidates(solver, hand)
        if not result or not cands:
            continue
        rivers, riichi, riichi_seats = make_rivers(rng, hand, rng.choice([1, 1, 2]))
        safety = build_safety_facts(rivers, riichi, '东', '西')
        common = common_safe_tiles(safety, riichi_seats, hand)
        # 决策规则：末巡(>=10) + 落后 + 听牌好形 → 进攻；否则防守
        late = rng.random() < 0.4
        behind = rng.random() < 0.5
        tenpai = result.get('tenpai') or any(c.get('tenpai') for c in cands)
        attack = late and behind and tenpai
        if attack:
            tenpai_discards = [c['tile'] for c in cands if c.get('tenpai')]
            if not tenpai_discards:
                continue
            discard = rng.choice(tenpai_discards)
            reason = '听牌好形保持进攻'
        else:
            safe = [t for t in common if t in hand] or [t for seat in riichi_seats
                    for t in (safety.get('现物', {}).get(seat) or []) if t in hand]
            if not safe:
                continue
            discard = rng.choice(safe)
            reason = '无役且巡目早先弃和'
        scores = {'own': rng.choice([15000, 25000, 35000]),
                  'right': rng.choice([25000, 35000, 45000])}
        turn = rng.randint(6, 14)
        prompt = format_prompt(hand, result['shanten'], cands, rivers, riichi,
                               safety, common, round_wind='东', self_wind='西',
                               scores=scores, turn=turn)
        records.append(make_record(prompt, discard, reason))
    return records


def gen_value(rng, solver, n):
    """打点：宝牌/役牌保留，打低价值浮牌。"""
    records = []
    for _ in range(n):
        hand = random_hand(rng)
        result, cands = solver_candidates(solver, hand)
        if not cands:
            continue
        rivers, riichi, _ = make_rivers(rng, hand, 0)
        dora = rng.sample([t for t in TILES], rng.randint(1, 3))
        # 保留宝牌/役牌，从候选中打非宝牌非役牌
        # 只把手牌里真实存在的宝牌/役牌计为"有价值"，否则会误把浮牌当宝牌
        actual_dora = set(dora) & set(hand)
        valuable = actual_dora | (HONORS & set(hand))
        low_value = [c['tile'] for c in cands if c['tile'] not in valuable]
        if not low_value:
            continue
        discard = low_value[0]
        reason = '保留宝牌役牌打浮牌' if valuable else '牌效率优先打浮牌'
        prompt = format_prompt(hand, result['shanten'] if result else None, cands,
                               rivers, riichi, {}, [], dora=dora)
        records.append(make_record(prompt, discard, reason))
    return records


def gen_simple(rng, solver, n, kind):
    """鸣牌/立直/终局/多立直：规则驱动的简化样本。"""
    records = []
    for _ in range(n):
        hand = random_hand(rng)
        result, cands = solver_candidates(solver, hand)
        if not cands:
            continue
        n_riichi = rng.choice([2, 3]) if kind == '多立直极限' else rng.choice([0, 1])
        rivers, riichi, riichi_seats = make_rivers(rng, hand, n_riichi)
        safety = build_safety_facts(rivers, riichi, '东', '西')
        common = common_safe_tiles(safety, riichi_seats, hand) if riichi_seats else []
        if kind == '多立直极限':
            safe = [t for t in common if t in hand]
            if safe:
                discard, reason = safe[0], '多家立直打共同安全'
            else:
                # 没有共同安全牌时也要训练：选最不危险的（现物覆盖最多家）
                coverage = {}
                for t in hand:
                    coverage[t] = sum(1 for seat in riichi_seats
                                      if t in (safety.get('现物', {}).get(seat) or []))
                best_cover = max(coverage.values()) if coverage else 0
                candidates_safe = [t for t, c in coverage.items() if c == best_cover]
                if not candidates_safe:
                    continue
                discard = candidates_safe[0]
                reason = '无共同安全牌选相对安全'
        elif kind == '鸣牌判断':
            # 门清手：打进张最优的牌保持门清（理由必须解释为何打这张）
            discard = cands[0]['tile']
            reason = '进张%s张最广' % cands[0].get('ukeire') if cands[0].get('ukeire') else '保持门清效率'
        elif kind == '立直判断':
            # 只有听牌（0向听）才谈立直；非听牌跳过，避免"3向听愚形不立直"的噪声
            tenpai_discards = [c for c in cands if c.get('tenpai')]
            if not tenpai_discards or (result and result.get('shanten', 9) > 0):
                continue
            discard = tenpai_discards[0]['tile']
            waits = tenpai_discards[0]
            reason = '听牌%s张待牌' % len(waits.get('waits', [])) if waits.get('waits') else '好形立直'
        else:  # 终局/顺位：弃牌理由必须对应实际选择
            discard = cands[0]['tile']
            # 有立直威胁时打安全牌保点棒；否则打进张最优保顺位
            if riichi_seats:
                safe = [t for t in common if t in hand] or [
                    t for seat in riichi_seats for t in (safety.get('现物', {}).get(seat) or [])
                    if t in hand]
                if safe:
                    discard = safe[0]
                    reason = '避免放铳保点棒'
                else:
                    reason = '进张%s张保顺位' % cands[0].get('ukeire') if cands[0].get('ukeire') else '末巡保顺位'
            else:
                reason = '进张%s张保顺位' % cands[0].get('ukeire') if cands[0].get('ukeire') else '末巡保顺位'
        if discard not in hand:
            continue
        prompt = format_prompt(hand, result['shanten'] if result else None, cands,
                               rivers, riichi, safety, common,
                               round_wind='东', self_wind='西', turn=rng.randint(8, 15))
        records.append(make_record(prompt, discard, reason))
    return records


def validate(records):
    """质量校验：牌必须在手牌中、格式正确。"""
    good = []
    for r in records:
        content = r['messages'][-1]['content']
        if '|' not in content:
            continue
        discard, reason = content.split('|', 1)
        user = r['messages'][1]['content']
        if '手牌(' not in user or discard not in user.split('\n')[0]:
            continue
        if len(reason) > 20:
            continue
        good.append(r)
    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total', type=int, default=50000)
    parser.add_argument('--out', default='_dev/train_data')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    solver = ExactSolver()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 按方案比例分配
    quota = {
        '牌效率': int(args.total * 0.30),
        '防守': int(args.total * 0.20),
        '攻守': int(args.total * 0.16),
        '打点': int(args.total * 0.10),
        '鸣牌判断': int(args.total * 0.08),
        '立直判断': int(args.total * 0.06),
        '终局顺位': int(args.total * 0.06),
        '多立直极限': int(args.total * 0.04),
    }
    all_records = []
    for kind, n in quota.items():
        print('生成 %s %d 条...' % (kind, n), flush=True)
        if kind == '牌效率':
            recs = gen_efficiency(rng, solver, n)
        elif kind == '防守':
            recs = gen_defense(rng, solver, n)
        elif kind == '攻守':
            recs = gen_push_fold(rng, solver, n)
        elif kind == '打点':
            recs = gen_value(rng, solver, n)
        else:
            recs = gen_simple(rng, solver, n, kind)
        recs = validate(recs)
        print('  有效 %d 条' % len(recs), flush=True)
        all_records.extend(recs)

    rng.shuffle(all_records)
    split = int(len(all_records) * 0.9)
    train, eval_ = all_records[:split], all_records[split:]
    for name, data in (('train.jsonl', train), ('eval.jsonl', eval_)):
        with open(out / name, 'w', encoding='utf-8') as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('写入 %s: %d 条' % (name, len(data)), flush=True)
    print('总计 %d 条' % len(all_records), flush=True)
    solver.close()


if __name__ == '__main__':
    main()
