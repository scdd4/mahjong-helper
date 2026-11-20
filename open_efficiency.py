"""Regular-hand efficiency for called hands using MahjongRepository/mahjong."""
from collections import Counter
from functools import lru_cache
from desktop_runtime import tile_index,group_tiles,effective_hand_total
from mahjong.shanten import Shanten


def _code(index):
    return str(index%9+1)+'mps'[index//9] if index<27 else str(index-26)+'z'


@lru_cache(maxsize=8192)
def _shanten(counts):
    return Shanten.calculate_shanten_for_regular_hand(counts)


def analyze_open(hand,groups,seen=None):
    total=effective_hand_total(hand,groups)
    if total not in (13,14) or not groups:raise ValueError('副露后的有效张数需要为13或14张')
    counts=[0]*34
    for code in hand:counts[tile_index(code)]+=1
    visible=Counter(tile_index(t) for t in seen or [])
    own=Counter()
    for group in groups:
        tiles=group_tiles(group)
        if len(tiles) not in (3,4):raise ValueError('副露组尚未完整识别')
        for tile in tiles:
            code=tile.get('name') if isinstance(tile,dict) else tile
            own[tile_index(code)]+=1
    # max 而非 +=：build_seen 可能已把高分自家副露计入 seen，取较大值
    # 保证同种牌只扣一次；低置信未进 seen 时由这里补足。
    for index,n in own.items():
        visible[index]=max(visible[index], n)
    if any(counts[i]+own[i]>4 for i in range(34)):raise ValueError('手牌与自家副露存在超过四张的同种牌')
    remaining=[max(0,4-counts[i]-visible[i]) for i in range(34)]

    def draws(rest,current):
        values=[]
        for index,n in enumerate(remaining):
            if not n or rest[index]>=4:continue
            rest[index]+=1
            improved=_shanten(tuple(rest))<current
            rest[index]-=1
            if improved:values.append({'tile':_code(index),'remaining':n})
        return values

    current=_shanten(tuple(counts))
    result={'shanten':current,'win':current==-1,'candidates':[],'waits':[],
            'scope':'副露普通型向听与进张；役种、符数、点数和振听需另行核验'}
    if total==14:
        candidates=[]
        for index,n in enumerate(counts):
            if not n:continue
            rest=counts.copy();rest[index]-=1
            shanten=_shanten(tuple(rest));ukeire=draws(rest,shanten)
            candidates.append({'tile':_code(index),'shanten':shanten,
                               'ukeire':sum(t['remaining'] for t in ukeire),'kinds':len(ukeire)})
        candidates.sort(key=lambda c:(c['shanten'],-c['ukeire'],-c['kinds']))
        result['candidates']=candidates[:8]
    elif current==0:
        result['waits']=draws(counts,current)
    return result


def possible_pon_plans(hand,groups,seen=None):
    if effective_hand_total(hand,groups)!=13:return []
    counts=Counter(tile_index(t) for t in hand)
    visible=Counter(tile_index(t) for t in seen or [])
    plans=[]
    for index,n in counts.items():
        if n<2 or n+visible[index]>=4:continue
        name=_code(index);rest=hand.copy();taken=[]
        for _ in range(2):
            tile=next(t for t in rest if tile_index(t)==index)
            rest.remove(tile);taken.append(tile)
        meld=[{'name':tile} for tile in taken+[name]]
        future=analyze_open(rest,list(groups)+[meld],list(seen or [])+[name]*3)
        plans.append({'tile':name,'best_shanten_after_call':future['candidates'][0]['shanten'],
                      'discard_candidates':future['candidates'][:2],
                      'yakuhai':'三元牌刻子有役牌' if index>=31 else ('数牌刻子本身无役' if index<27 else '风牌是否为役牌需核对场风和自风'),
                      'note':'假设未来他家打出此牌可碰，碰后仍须弃牌；是否鸣牌由模型结合有役、速度和防守决定'})
    return plans


def closed_route_facts(hand,groups):
    """Separate regular/special forms so their shanten cannot be interchanged."""
    if groups:return {'note':'已有副露，七对子与国士路线不可用'}
    counts=[0]*34
    for tile in hand:counts[tile_index(tile)]+=1
    def forms(values):
        return {'普通型':_shanten(tuple(values)),
                '七对子':Shanten.calculate_shanten_for_chiitoitsu_hand(values),
                '国士':Shanten.calculate_shanten_for_kokushi_hand(values)}
    result={'current':forms(counts),'after_discard':[],
            'seven_pairs':{'pair_kinds':sum(n>=2 for n in counts),
                           'new_pair_draws':[_code(i) for i,n in enumerate(counts) if n==1],
                           'note':'摸一张只能把现有单张变为新对子；第三张同牌不增加七对子对子数'},
            'note':'每条路线分开核算；一向听不能立直，须门清听牌并符合点棒与余牌条件'}
    if len(hand)==14:
        for index,n in enumerate(counts):
            if n:
                rest=counts.copy();rest[index]-=1
                result['after_discard'].append({'tile':_code(index),'forms':forms(rest)})
    return result
