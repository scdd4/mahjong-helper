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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safety_core import dora_from_indicator

TILES = [str(n) + s for s in 'mps' for n in range(1, 10)] + [str(n) + 'z' for n in range(1, 8)]
HONORS = {str(n) + 'z' for n in range(1, 8)}
SEATS = ('own', 'right', 'across', 'left')
_cand_rng = random.Random(20261123)  # 候选乱序专用，切断"答案恒在A"的位置捷径
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
    candidates = _cand_rng.sample(candidates, len(candidates)) if candidates else candidates
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


def prompt_cands(cands, rng):
    """乱序候选列表后写入 prompt。

    v1 数据答案 100% 排在第 1 位（300/300），模型学的是"永远选 A"的位置捷径：
    4B 靠它虚高（eval 恰好答案也几乎都在 A），27B 记忆能力更强反而暴露捷径失效，
    efficiency 从 90% 崩到 62%。重训前必须切断位置捷径。
    """
    lst = list(cands)
    rng.shuffle(lst)
    return lst


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


def tenpai_hand(rng, solver):
    """构造 14 张听牌手牌 + 一个'打了仍听牌'的安全弃牌。

    方法：先组一副完整和牌型（4 面子 + 1 对，14 张），去掉其中一张 → 13 张必听牌；
    再摸进一张杂牌 d 凑成 14 张，则打出 d 后回到那 13 张听牌态（保证保持听牌）。
    v1 的 push_fold"进攻"类只有 1 条样本（随机手牌几乎不可能听牌），导致全模型攻守崩，
    此函数保证进攻样本可量产。返回 (hand14, keep_tenpai_discard)。
    """
    for _ in range(300):
        tiles = []
        ok = True
        for _ in range(4):
            if rng.random() < 0.75:
                s = rng.choice('mps'); n = rng.randint(1, 7)
                tiles += [str(n) + s, str(n + 1) + s, str(n + 2) + s]
            else:
                h = str(rng.randint(1, 7)) + 'z'
                tiles += [h, h, h]
        if rng.random() < 0.5:
            ps = rng.choice('mps'); pn = rng.randint(1, 9)
            pair = str(pn) + ps
        else:
            pair = str(rng.randint(1, 7)) + 'z'
        tiles += [pair, pair]
        if any(tiles.count(t) > 4 for t in set(tiles)):
            continue
        # 去掉一张得 13 张听牌
        drop_idx = rng.randrange(14)
        hand13 = tiles[:drop_idx] + tiles[drop_idx + 1:]
        try:
            if solver.analyze(hand13).get('shanten') != 0:
                continue
        except Exception:
            continue
        # 摸一张杂牌 d（不使任一张超 4），打出 d 后 = hand13 听牌
        d = rng.choice(TILES)
        if hand13.count(d) >= 4:
            continue
        hand14 = hand13 + [d]
        rng.shuffle(hand14)
        return hand14, d
    return None, None


def gen_push_fold(rng, solver, n):
    """攻守：末巡+落后+听牌好形 → 进攻保持听牌；否则弃和打安全牌。

    v2：约 40% 样本用 tenpai_hand 强制构造听牌进攻场景，修复"进攻类仅 1 条"。
    """
    records = []
    for _ in range(n):
        want_attack = rng.random() < 0.4
        if want_attack:
            hand, _ = tenpai_hand(rng, solver)
            if not hand:
                continue
            result, cands = solver_candidates(solver, hand)
            tenpai_discards = [c['tile'] for c in cands if c.get('tenpai')]
            if not tenpai_discards:
                continue
            discard = tenpai_discards[0]
            reason = '听牌保持进攻'
            n_riichi = rng.choice([1, 2])  # 进攻题必有威胁，否则 push 无意义
        else:
            hand = random_hand(rng)
            result, cands = solver_candidates(solver, hand)
            if not result or not cands:
                continue
            tenpai_discards = []
            discard = None
            n_riichi = rng.choice([1, 1, 2])
        rivers, riichi, riichi_seats = make_rivers(rng, hand, n_riichi)
        safety = build_safety_facts(rivers, riichi, '东', '西')
        common = common_safe_tiles(safety, riichi_seats, hand)
        if not want_attack:
            safe = [t for t in common if t in hand] or [
                t for seat in riichi_seats
                for t in (safety.get('现物', {}).get(seat) or []) if t in hand]
            if not safe:
                continue
            discard = safe[0]
            reason = '有威胁弃和打安全'
        # 进攻场景点棒落后且末巡，弃和场景点棒领先或中巡（与理由因果对齐）
        if want_attack:
            scores = {'own': rng.choice([15000, 22000]),
                      'right': 45000, 'across': 35000, 'left': 30000}
            turn = rng.randint(11, 15)
        else:
            scores = {'own': rng.choice([40000, 45000]),
                      'right': 22000, 'across': 20000, 'left': 18000}
            turn = rng.randint(3, 9)
        prompt = format_prompt(hand, result['shanten'], cands, rivers, riichi,
                               safety, common, round_wind='东', self_wind='西',
                               scores=scores, turn=turn)
        records.append(make_record(prompt, discard, reason))
    return records


def gen_value(rng, solver, n):
    """打点：宝牌/役牌保留，从打点与效率兼顾处选弃牌。

    v1 缺陷（导致 4B value 只有 14%）：把宝牌**指示牌**当成宝牌本体保留
    （actual_dora = dora_indicator ∩ hand），教的是错误概念。
    v2 修复：指示牌 → dora_from_indicator → 真宝牌；理由与"保留宝牌"因果对齐。
    并且不再随机丢浮牌——只在候选里挑"打了不损向听、且不拆真宝牌/役牌对子"的最优牌。
    """
    records = []
    for _ in range(n):
        hand = random_hand(rng)
        result, cands = solver_candidates(solver, hand)
        if not cands:
            continue
        rivers, riichi, _ = make_rivers(rng, hand, 0)
        indicators = rng.sample([t for t in TILES], rng.randint(1, 3))
        real_dora = {dora_from_indicator(t) for t in indicators} & set(hand)
        # 役牌：场风/自风 + 三元牌；只认手牌里成对/成刻的才算"有价值"，孤张役牌可打
        honors_kept = {t for t in HONORS & set(hand) if hand.count(t) >= 2}
        valuable = real_dora | honors_kept
        best = min(c['shanten'] for c in cands)
        # 只在最优向听候选里挑，且不拆有价值牌
        safe_drops = [c for c in cands
                      if c['shanten'] == best and c['tile'] not in valuable]
        if not safe_drops:
            continue
        # 有价值牌确实存在时才归为"打点"决策，否则是纯效率（避免噪声标签）
        if not valuable:
            continue
        discard = max(safe_drops, key=lambda c: c.get('ukeire', 0))['tile']
        reason = ('保留宝牌打浮牌' if real_dora else '保留役牌打浮牌')
        prompt = format_prompt(hand, result['shanten'] if result else None, cands,
                               rivers, riichi, {}, [], dora=indicators)
        records.append(make_record(prompt, discard, reason))
    return records


def gen_simple(rng, solver, n, kind):
    """鸣牌/立直/终局/多立直：规则驱动的简化样本。"""
    records = []
    for _ in range(n):
        if kind == '立直判断':
            # 随机手牌几乎不可能听牌（v1 该类 2490 条只出 2 条有效），改用听牌构造器
            hand, _ = tenpai_hand(rng, solver)
            if not hand:
                continue
        else:
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
