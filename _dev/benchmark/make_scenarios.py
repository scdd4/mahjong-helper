#!/usr/bin/env python
"""打牌能力基准——场景生成与校验。

用本地 ExactSolver 为 efficiency 类场景计算每个弃牌的向听/进张，
把最优解（含并列）写为期望答案；defense/push_fold/value 类的期望答案
为人工标注（现物/筋/客风为确定性安全事实，push/fold 的"保持听牌"
约束由求解器复核）。输出 scenarios.json 与校验报告。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from desktop_runtime import ExactSolver, tile_index, build_safety_facts

SOLVER = ExactSolver()

def idx_name(index):
    return str(index % 9 + 1) + 'mps'[index // 9] if index < 27 else str(index - 26) + 'z'

def name_code(name):
    return tile_index(name)

def efficiency_expected(hand14, seen=()):
    """弃牌后 13 张的 (向听, 受入总数, 受入种数)；返回最优弃牌集合。

    受入 = ukeire（向听推进张）+ waits（听牌时的和牌张），统一量纲；
    seen 为牌河/副露/指示牌等可见牌，使 remaining 扣除已见张数。
    analyze 的 discards 仅在特定状态填充，不作依据。
    """
    seen_codes = [c for c in seen if isinstance(c, str)]
    table = []
    for i, drop in enumerate(hand14):
        rest = hand14[:i] + hand14[i + 1:]
        result = SOLVER.analyze(rest, seen_codes)
        ukeire = result.get('ukeire', [])
        waits = result.get('waits', [])
        total = sum(u['remaining'] for u in ukeire) + sum(w['remaining'] for w in waits)
        kinds = len(ukeire) + len(waits)
        table.append({'drop': drop, 'shanten': result['shanten'], 'total': total, 'kinds': kinds})
    best_shanten = min(t['shanten'] for t in table)
    candidates = [t for t in table if t['shanten'] == best_shanten]
    max_total = max(t['total'] for t in candidates)
    finalists = [t for t in candidates if t['total'] == max_total]
    max_kinds = max(t['kinds'] for t in finalists)
    best = sorted({t['drop'] for t in finalists if t['kinds'] == max_kinds})
    return {'best_shanten': best_shanten, 'max_ukeire': max_total,
            'expected': best[0], 'alternatives': best[1:], 'table': table}

def tenpai_keeping(hand14, seen=()):
    """push 场景：返回全部能保持最优向听的弃牌（不只受入最大者）。"""
    calc = efficiency_expected(hand14, seen)
    all_tenpai = sorted({t['drop'] for t in calc['table'] if t['shanten'] == calc['best_shanten']})
    return all_tenpai

def physics_check(scenario):
    counts = {}
    state = scenario['state']
    piles = [state['hand'] + ([state['drawn']] if state.get('drawn') else [])]
    for seat in ('own', 'right', 'across', 'left'):
        melds = state.get('melds', {}).get(seat) or []
        for group in melds:
            piles.append(group['tiles'] if isinstance(group, dict) else group)
        piles.append(state.get('rivers', {}).get(seat) or [])
    piles.append(state.get('dora_indicators', []))
    for pile in piles:
        for name in pile:
            counts[name] = counts.get(name, 0) + 1
    overflow = {k: v for k, v in counts.items() if v > 4}
    return overflow


def accepted_defense_tiles(sc,hand):
    """Expert alternatives plus genbutsu common to every confirmed threat.

    Suji and honors are relative-risk annotations, never an absolute-safe union.
    """
    state=sc['state'];active=[s for s,v in state.get('riichi',{}).items() if v is True and s!='own']
    common=set.intersection(*(set(state.get('rivers',{}).get(s,[])) for s in active)) if active else set()
    return sorted((set(sc.get('safe',[]))|common)&set(hand)-set(sc.get('must_avoid',[])))

SCENARIOS = [
    # ============ 牌效率 · basic ============
    dict(id='eff-001', category='efficiency', difficulty='basic',
         description='三组完整顺子+雀头后，四张孤张客风字牌只能留两张：孤字牌取舍。',
         state=dict(hand=['1m','2m','3m','4p','5p','6p','7s','8s','9s','1z','1z','2z','3z'],
                    drawn='4z', dora_indicators=[], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=70),
         kind='efficiency'),
    dict(id='eff-002', category='efficiency', difficulty='basic',
         description='两组两面进张完成后，孤字牌雀头听牌：打字牌即听。',
         state=dict(hand=['2m','3m','4m','5m','6m','7m','2p','3p','4p','5p','6p','7p','1z'],
                    drawn='9p', dora_indicators=[], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=65),
         kind='efficiency'),
    dict(id='eff-003', category='efficiency', difficulty='basic',
         description='四面张两端延伸后多出孤立端牌：识别多余端牌。',
         state=dict(hand=['1m','2m','3m','4m','5m','7m','8m','2p','3p','4p','7p','8p','9p'],
                    drawn='6m', dora_indicators=[], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=60),
         kind='efficiency'),
    dict(id='eff-004', category='efficiency', difficulty='basic',
         description='进阶取舍：打一张宝牌(4m)换取立刻听牌（另一张宝牌保留在456m顺子中），而不是抱着宝牌对子停留在1向听。',
         state=dict(hand=['4m','4m','5m','6m','7m','8m','2p','3p','4p','5p','6p','7p','1z'],
                    drawn='1z', dora_indicators=['3m'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=55),
         kind='efficiency'),
    dict(id='eff-005', category='efficiency', difficulty='basic',
         description='两刻子候选与孤字牌并存：留对子价值高于孤张。',
         state=dict(hand=['2m','2m','5m','6m','7m','3p','4p','5p','2s','3s','4s','1z','4z'],
                    drawn='6z', dora_indicators=[], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=68),
         kind='efficiency'),
    # ============ 牌效率 · advanced ============
    dict(id='eff-006', category='efficiency', difficulty='advanced',
         description='宝牌暗刻+多搭子：唯一无用张为孤立客风字牌。',
         state=dict(hand=['2m','3m','4m','4m','4m','6m','7m','2p','3p','4p','6p','7p','1z'],
                    drawn='8m', dora_indicators=['3m'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=52),
         kind='efficiency'),
    dict(id='eff-007', category='efficiency', difficulty='advanced',
         description='单骑听牌改良：结合牌河剩余张数选保留哪张浮牌（9索仅剩2张，5索剩3张）。',
         state=dict(hand=['1m','2m','3m','4m','5m','6m','2p','3p','4p','5p','6p','7p','5s'],
                    drawn='9s', dora_indicators=['1m'],
                    rivers={'across': ['9s','4m']}, melds={},
                    round='东1局', self_wind='东', remaining=50),
         kind='efficiency'),
    dict(id='eff-008', category='efficiency', difficulty='advanced',
         description='字牌浮子取舍：三组顺子完成后字牌价值比较（连风 vs 客风）。',
         state=dict(hand=['1m','2m','3m','4p','5p','6p','7s','8s','9s','5z','6z','6z','7z'],
                    drawn='5z', dora_indicators=[], rivers={}, melds={},
                    round='南1局', self_wind='西', remaining=48),
         kind='efficiency'),
    dict(id='eff-009', category='efficiency', difficulty='advanced',
         description='坎张与两面并存时的进张宽度比较。',
         state=dict(hand=['1m','2m','4m','5m','6m','8m','2p','3p','4p','6p','7p','3s','4s'],
                    drawn='9m', dora_indicators=['6s'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=58),
         kind='efficiency'),
    dict(id='eff-010', category='efficiency', difficulty='advanced',
         description='听牌后改良：单骑改两面（打字牌留两面）。',
         state=dict(hand=['2m','3m','4m','5m','6m','7m','2p','3p','4p','7p','8p','2z','2z'],
                    drawn='9m', dora_indicators=[], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=46),
         kind='efficiency'),
    # ============ 防守 · basic（对面立直，安全牌为确定性事实）============
    dict(id='def-001', category='defense', difficulty='basic',
         description='现物安全：手牌中的字牌恰为立直者牌河现物，直接打现物。',
         state=dict(hand=['1z','2m','3m','5m','6m','2p','4p','6p','8p','3s','5s','7s','9s'],
                    drawn='4m', dora_indicators=['1m'],
                    rivers={'across': ['1z','3p','5p','7m','9s']},
                    riichi={'across': True},
                    round='东1局', self_wind='东', remaining=45),
         kind='defense', safe=['1z'], must_avoid=['4m','5m']),
    dict(id='def-002', category='defense', difficulty='basic',
         description='筋安全：立直者早打4万，则1万/7万为筋；无筋5万危险。',
         state=dict(hand=['1m','7m','5m','2m','3m','2p','3p','5s','6s','7s','8s','4z','4z'],
                    drawn='9m', dora_indicators=[],
                    rivers={'across': ['4m','6s','2p','8p','1z']},
                    riichi={'across': True},
                    round='东1局', self_wind='南', remaining=40),
         kind='defense', safe=['1m','7m','4z'], must_avoid=['5m']),
    dict(id='def-003', category='defense', difficulty='basic',
         description='客风字牌安全度高于无筋数牌；立直者牌河内的字牌是现物。',
         state=dict(hand=['1s','2s','3s','4s','5s','6s','7s','8s','9s','3z','4z','2m','3m'],
                    drawn='5m', dora_indicators=['2s'],
                    rivers={'across': ['9m','2s','6p','3z']},
                    riichi={'across': True},
                    round='东1局', self_wind='南', remaining=38),
         kind='defense', safe=['3z','4z'], must_avoid=['2m','3m','5m']),
    dict(id='def-004', category='defense', difficulty='basic',
         description='双立直：同时是两家现物的牌才最安全。',
         state=dict(hand=['2z','1m','2m','4m','5m','6m','3p','5p','7p','2s','3s','4s','9s'],
                    drawn='8s', dora_indicators=['5p'],
                    rivers={'across': ['2z','4p','8m'], 'right': ['2z','6p','3s']},
                    riichi={'across': True, 'right': True},
                    round='东1局', self_wind='西', remaining=35),
         kind='defense', safe=['2z'], must_avoid=['5p','8s']),
    dict(id='def-005', category='defense', difficulty='basic',
         description='自风风牌并不安全（单骑/对对风险），客风与客三元更安全。',
         state=dict(hand=['2m','4m','5m','7m','8m','1p','3p','4p','6p','9p','1z','2z','4z'],
                    drawn='6z', dora_indicators=['4z'],
                    rivers={'across': ['3m','7s','2p','8p','6m']},
                    riichi={'across': True},
                    round='东1局', self_wind='东', remaining=42),
         kind='defense', safe=['2z','4z','6z'], must_avoid=['1z','2m','4m','5m','7m','8m']),
    # ============ 防守 · advanced ============
    dict(id='def-006', category='defense', difficulty='advanced',
         description='现物优先级最高：手牌同时含筋牌与现物时打现物。',
         state=dict(hand=['1m','2m','5m','9m','3p','4p','5p','6p','7p','2s','3s','4s','8s'],
                    drawn='9p', dora_indicators=['3p'],
                    rivers={'across': ['1m','4m','7m','2p']},
                    riichi={'across': True},
                    round='东1局', self_wind='东', remaining=33),
         kind='defense', safe=['1m'], must_avoid=['2m','5m','9m']),
    dict(id='def-007', category='defense', difficulty='advanced',
         description='立直即立（一发风险）：唯一现物是唯一安全牌。',
         state=dict(hand=['9p','1m','2m','4m','5m','6m','7m','8m','3s','4s','5s','6s','7s'],
                    drawn='9m', dora_indicators=['7m'],
                    rivers={'across': ['9p']},
                    riichi={'across': True},
                    round='东1局', self_wind='东', remaining=48),
         kind='defense', safe=['9p'], must_avoid=['9m','1m','2m','4m','5m','6m','7m','8m','3s','4s','5s','6s','7s']),
    dict(id='def-008', category='defense', difficulty='advanced',
         description='无现物时安全度排序：客三元/客风字牌 > 筋牌 > 无筋中张。',
         state=dict(hand=['2m','8m','3z','4z','5z','6z','7z','1p','9p','2s','4s','6s','8s'],
                    drawn='3m', dora_indicators=['8s'],
                    rivers={'across': ['5m']},
                    riichi={'across': True},
                    round='东南局', self_wind='北', remaining=30),
         kind='defense', safe=['3z','4z','5z','6z','7z','2m','8m'], must_avoid=['3m','1p','9p','2s','4s','6s','8s']),
    # ============ 攻守判断 push/fold ============
    dict(id='pf-001', category='push_fold', difficulty='advanced',
         description='好形听牌+3宝牌 vs 一家立直：宝牌多、听牌好形应进攻。',
         state=dict(hand=['2m','3m','4m','5m','6m','7m','7p','8p','9p','3s','4s','5s','5z'],
                    drawn='1m', dora_indicators=['4m'],
                    rivers={'across': ['1z','4s','9p']},
                    riichi={'across': True},
                    round='东1局', self_wind='东', remaining=18,
                    scores={'own': 27000, 'right': 24000, 'across': 26000, 'left': 23000}),
         kind='push_fold', direction='push', stance='进攻'),
    dict(id='pf-002', category='push_fold', difficulty='advanced',
         description='离听牌较远且无宝牌 vs 立直：进度差、价值低应防守。',
         state=dict(hand=['1m','2m','4m','6m','8m','2p','5p','9p','2s','6s','8s','3z','4z'],
                    drawn='7m', dora_indicators=[],
                    rivers={'across': ['1z','5p','8s','3m']},
                    riichi={'across': True},
                    round='东2局', self_wind='南', remaining=30,
                    scores={'own': 25000, 'right': 27000, 'across': 24000, 'left': 24000}),
         kind='push_fold', direction='fold', stance='防守',
         safe=['1z','3m','3z','4z'], must_avoid=['7m','2p']),
    dict(id='pf-003', category='push_fold', difficulty='advanced',
         description='亲家好形听牌 vs 立直：亲家被牌损失大、和牌收益高，应进攻。',
         state=dict(hand=['3m','4m','5m','6m','7m','8m','4p','5p','6p','2s','3s','4s','7z'],
                    drawn='9m', dora_indicators=['3s'],
                    rivers={'across': ['1z','2z','9s']},
                    riichi={'across': True},
                    round='东1局', self_wind='东', remaining=25,
                    scores={'own': 30000, 'right': 23000, 'across': 24000, 'left': 23000}),
         kind='push_fold', direction='push', stance='进攻'),
    dict(id='pf-004', category='push_fold', difficulty='advanced',
         description='南4局末巡落后需点数：即使对面立直，末位也需进攻争取和牌。',
         state=dict(hand=['2m','3m','4m','6m','7m','8m','3p','4p','5p','6p','7p','8p','5z'],
                    drawn='9m', dora_indicators=['5p'],
                    rivers={'across': ['1z','7m','3s']},
                    riichi={'across': True},
                    round='南4局', self_wind='西', remaining=8,
                    scores={'own': 28000, 'right': 31000, 'across': 46000, 'left': 32000}),
         kind='push_fold', direction='push', stance='进攻'),
    # ============ 打点/价值 ============
    dict(id='val-001', category='value', difficulty='advanced',
         description='3宝牌暗刻听牌：打孤张保持满贯潜力，弃宝牌相邻张不可取。',
         state=dict(hand=['4m','4m','4m','6m','7m','8m','2p','3p','4p','5p','6p','7p','9p'],
                    drawn='9m', dora_indicators=['3m'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=44),
         kind='value', expected='9m', alternatives=['9p'],
         rationale_keywords=['宝牌', '听牌']),
    dict(id='val-002', category='value', difficulty='advanced',
         description='宝牌对子保留：即使打宝牌相邻张能改善进张，也应保留宝牌价值。',
         state=dict(hand=['2m','3m','4m','5m','6m','7m','4p','4p','4p','6p','7p','2z','2z'],
                    drawn='9s', dora_indicators=['3p'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=40),
         kind='value', expected='9s', alternatives=[],
         rationale_keywords=['宝牌']),
]

def main():
    out = []
    problems = []
    for sc in SCENARIOS:
        state = dict(sc['state'])
        hand14 = state['hand'] + [state['drawn']]
        overflow = physics_check(sc)
        if overflow:
            problems.append(f"{sc['id']}: 物理超编 {overflow}")
        if sc['kind'] == 'efficiency':
            hand14 = state['hand'] + [state['drawn']]
            if SOLVER.analyze(hand14).get('win'):
                problems.append(f"{sc['id']}: 摸到的牌恰好和牌，场景无效")
        entry = {'id': sc['id'], 'category': sc['category'], 'difficulty': sc['difficulty'],
                 'description': sc['description'], 'state': {
                     'hand': hand14, 'dora_indicators': state.get('dora_indicators', []),
                     'rivers': state.get('rivers', {}), 'melds': state.get('melds', {}),
                     'riichi': state.get('riichi', {}), 'round': state.get('round'),
                     'self_wind': state.get('self_wind'), 'remaining': state.get('remaining'),
                     'scores': state.get('scores', {})},
                 'kind': sc['kind']}
        if sc['kind'] == 'efficiency':
            seen = []
            for seat in ('own', 'right', 'across', 'left'):
                seen += state.get('rivers', {}).get(seat) or []
                for group in (state.get('melds', {}).get(seat) or []):
                    seen += group['tiles'] if isinstance(group, dict) else group
            seen += state.get('dora_indicators', [])
            calc = efficiency_expected(hand14, seen)
            entry['expected'] = {'tile': calc['expected'], 'alternatives': calc['alternatives'],
                                 'best_shanten': calc['best_shanten'], 'max_ukeire': calc['max_ukeire']}
            entry['solver_table'] = [{'drop': t['drop'], 'shanten': t['shanten'], 'ukeire': t['total']} for t in calc['table']]
        elif sc['kind'] == 'defense':
            entry['expected'] = {'safe': accepted_defense_tiles(sc,hand14), 'must_avoid': sc['must_avoid']}
        elif sc['kind'] == 'push_fold':
            seen = []
            for seat in ('own', 'right', 'across', 'left'):
                seen += state.get('rivers', {}).get(seat) or []
                for group in (state.get('melds', {}).get(seat) or []):
                    seen += group['tiles'] if isinstance(group, dict) else group
            seen += state.get('dora_indicators', [])
            entry['expected'] = {'direction': sc['direction'], 'stance': sc['stance']}
            if sc['direction'] == 'fold':
                entry['expected']['safe'] = accepted_defense_tiles(sc,hand14)
                entry['expected']['must_avoid'] = sc['must_avoid']
            else:
                entry['expected']['tenpai_discards'] = tenpai_keeping(hand14, seen)
        elif sc['kind'] == 'value':
            entry['expected'] = {'tile': sc['expected'], 'alternatives': sc.get('alternatives', []),
                                 'rationale_keywords': sc.get('rationale_keywords', [])}
        out.append(entry)
    target = Path(__file__).parent / 'scenarios.json'
    target.write_text(json.dumps({'version': '1.1', 'scenarios': out}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'写 {len(out)} 场景 → {target}')
    if problems:
        print('!! 物理校验问题:')
        for p in problems:
            print('  ', p)
        sys.exit(1)
    print('物理校验通过（无超编）')

if __name__ == '__main__':
    main()
