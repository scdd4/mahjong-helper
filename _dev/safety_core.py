"""纯 Python 安全事实核心（生产 desktop_runtime 的无依赖副本）。

服务器无法 import desktop_runtime（含 ctypes.wintypes 等 Windows 依赖），
故把评测/训练共用的安全计算逐字复制到此，保证本机与服务器口径完全一致。

来源：desktop_runtime.py 的 TILE_RE / SUJI_FROM_MIDDLE / SUJI_SINGLE /
tile_name_to_canonical / build_safety_facts / common_safe_tiles，逻辑不得改动。
"""
import re

TILE_RE = re.compile(r'^(?:[0-9][mps]|[1-7]z)$')
SUJI_FROM_MIDDLE = {4: (1, 7), 5: (2, 8), 6: (3, 9)}
SUJI_SINGLE = {1: (4,), 2: (5,), 3: (6,), 7: (4,), 8: (5,), 9: (6,)}


def tile_name_to_canonical(code):
    return '5' + code[1] if code[0] == '0' else code


def build_safety_facts(rivers, riichi, round_wind=None, self_wind=None):
    """Per-player visible genbutsu and limited suji evidence, never a global safe union."""
    facts = {'现物': {}, '筋': {}, '单筋': {}, '两筋': {}, '客风客三元': [],
             '说明': '现物只对对应玩家成立；筋仍可能被嵌张、边张、双碰或单骑命中。'}
    for seat, declared in (riichi or {}).items():
        if not declared:
            continue
        river = [t for t in (rivers.get(seat) or []) if isinstance(t, str) and TILE_RE.fullmatch(t)]
        river = [tile_name_to_canonical(t) for t in river]
        facts['现物'][seat] = sorted(set(river))
        suji, double = set(), set()
        for tile in river:
            if tile.endswith('z'):
                continue
            number = 5 if tile[0] == '0' else int(tile[0])
            for side in SUJI_FROM_MIDDLE.get(number, ()):
                code = str(side) + tile[1]
                if code not in facts['现物'][seat]:
                    suji.add(code)
        single = set()
        for tile in river:
            if tile.endswith('z'):
                continue
            number = 5 if tile[0] == '0' else int(tile[0])
            for side in SUJI_SINGLE.get(number, ()):
                code = str(side) + tile[1]
                if code not in facts['现物'][seat] and code not in suji:
                    single.add(code)
        facts['单筋'][seat] = sorted(single - set(facts['现物'][seat]))
        for suit in 'mps':
            for middle in (4, 5, 6):
                if str(middle - 3) + suit in river and str(middle + 3) + suit in river:
                    code = str(middle) + suit
                    if code not in river:
                        double.add(code)
        facts['筋'][seat] = sorted(suji - set(facts['现物'][seat]))
        facts['两筋'][seat] = sorted(double)
    return facts


def dora_from_indicator(code):
    """指示牌 → 宝牌。数牌循环 9→1；字牌双循环 东南西北 / 白发中。
    与生产 desktop_runtime.dora_from_indicator 逐字一致。"""
    number = 5 if code[0] == '0' else int(code[0])
    if code[1] != 'z':
        return str(number % 9 + 1) + code[1]
    return str({1: 2, 2: 3, 3: 4, 4: 1, 5: 6, 6: 7, 7: 5}[number]) + 'z'


def common_safe_tiles(facts, seats, hand=None):
    """Tiles safe against every declared riichi player (intersection, not union)."""
    seats = [s for s in seats if s in (facts.get('现物') or {})]
    if not seats:
        return []
    pools = []
    for seat in seats:
        pool = (set(facts['现物'].get(seat, []))
                | set(facts.get('筋', {}).get(seat, []))
                | set(facts.get('两筋', {}).get(seat, [])))
        pools.append(pool)
    common = set.intersection(*pools)
    if hand:
        common &= set(hand)
    return sorted(common)
