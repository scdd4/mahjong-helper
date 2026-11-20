"""能力上限极限专项：与常规基准分离，category='limit'。

原则：
1. 只放高区分度场景——反复出错的模式（现物幻觉/筋边界/多威胁）。
2. 期望必须可自动验证：效率由 ExactSolver、安全由 build_safety_facts。
3. 不靠"刁钻但无意义"的牌型，每题对应一个明确的可迁移规则。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LIMIT_SCENARIOS = [
    # ===== 防守：现物幻觉与多威胁 =====
    dict(id='lim-001', category='limit', difficulty='extreme',
         description='双立直：3z同时出现在两家牌河是共同现物；1m只对右家、9p只在对家，打单家现物会被另一家铳。',
         state=dict(hand=['1m','9p','3z','2m','4m','5m','7p','8p','3s','4s','5s','6s','6z'],
                    drawn='9m', dora_indicators=[],
                    rivers={'right':['1m','5p','7s','3z'],'across':['9p','6p','3s','3z']},
                    riichi={'right':True,'across':True}, melds={},
                    round='南2局', self_wind='北', remaining=52),
         kind='defense', safe=['3z'], must_avoid=['9m','5m']),

    dict(id='lim-002', category='limit', difficulty='extreme',
         description='现物优先级：手牌有现物2p、强筋1m/7m、无筋5m，必须打现物2p。',
         state=dict(hand=['2p','1m','7m','5m','3p','4p','6s','7s','8s','2s','3s','4s','1z'],
                    drawn='9m', dora_indicators=[],
                    rivers={'across':['4m','2p','8p','1z','9s']},
                    riichi={'across':True}, melds={},
                    round='东2局', self_wind='东', remaining=58),
         kind='defense', safe=['2p','1z'], must_avoid=['5m','9m']),

    dict(id='lim-003', category='limit', difficulty='extreme',
         description='三家立直：4z在三家牌河都是现物；任何只出现在单家牌河的现物都不足以安全。',
         state=dict(hand=['4z','1m','5m','2p','8p','3s','7s','1z','2z','3z','5z','6z','7z'],
                    drawn='9m', dora_indicators=[],
                    rivers={'right':['1m','6p','8s','4z'],'across':['5m','2p','3s','4z'],
                            'left':['8p','7s','4z','3z']},
                    riichi={'right':True,'across':True,'left':True}, melds={},
                    round='南3局', self_wind='东', remaining=44),
         kind='defense', safe=['4z'], must_avoid=['9m','5m','1m']),

    dict(id='lim-004', category='limit', difficulty='extreme',
         description='筋边界：立直者切8m，5m只是单侧筋（弱线索）；现物2p才是确定安全。',
         state=dict(hand=['2p','5m','3p','4p','6p','2s','3s','4s','5s','7z','7z','1m','9m'],
                    drawn='8p', dora_indicators=[],
                    rivers={'across':['8m','2p','6s','1z']},
                    riichi={'across':True}, melds={},
                    round='东3局', self_wind='南', remaining=55),
         kind='defense', safe=['2p','1z'], must_avoid=['9m','8p']),

    dict(id='lim-005', category='limit', difficulty='extreme',
         description='现物幻觉陷阱：牌河只有9s/1p，手牌没有5m；不得凭空报5m现物。',
         state=dict(hand=['9p','3m','4m','5m','2p','3p','4p','6s','7s','8s','2z','3z','4z'],
                    drawn='1m', dora_indicators=[],
                    rivers={'across':['9s','1p','7m','5z']},
                    riichi={'across':True}, melds={},
                    round='东1局', self_wind='西', remaining=63),
         kind='defense', safe=['2z','4z'], must_avoid=['1m','5m']),

    # ===== 牌效率：多路线与物理约束 =====
    dict(id='lim-006', category='limit', difficulty='extreme',
         description='普通型/七对/国士三路线竞争：求解器判定最优弃牌，不要混用路线数字。',
         state=dict(hand=['1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z'],
                    drawn='2m', dora_indicators=[], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=66),
         kind='efficiency'),

    dict(id='lim-007', category='limit', difficulty='extreme',
         description='牌河扣减反转：9s已见三张，继续留9s的进张被高估；求解器按可见牌重算。',
         state=dict(hand=['1m','2m','3m','4m','5m','6m','2p','3p','4p','5s','6s','7s','9s'],
                    drawn='8s', dora_indicators=[],
                    rivers={'own':['9s','9s','9s'],'across':['8s','8s']}, melds={},
                    round='东2局', self_wind='东', remaining=54),
         kind='efficiency'),

    dict(id='lim-008', category='limit', difficulty='extreme',
         description='赤五+多面听牌+物理4张：0m/5m同牌种已四张，进张计算须排除。',
         state=dict(hand=['0m','5m','2m','3m','4m','6m','2p','3p','4p','5p','6s','7s','8s'],
                    drawn='9s', dora_indicators=['4m'], rivers={'across':['5m','0m']}, melds={},
                    round='东1局', self_wind='东', remaining=60),
         kind='efficiency'),

    # ===== 攻守：极端点差与多威胁 =====
    dict(id='lim-009', category='limit', difficulty='extreme',
         description='南4局落后26000、双立直、只剩6张：好形听牌必须进攻，弃和等于输。',
         state=dict(hand=['1m','2m','3m','4m','5m','6m','7p','8p','9p','2s','3s','4s','5z'],
                    drawn='6s', dora_indicators=['8p'],
                    rivers={'right':['1z','6p','9s'],'across':['2z','5p','3s']},
                    riichi={'right':True,'across':True}, melds={},
                    round='南4局', self_wind='东', remaining=6,
                    scores={'own':9000,'right':28000,'across':35000,'left':28000}),
         kind='push_fold', direction='push', stance='进攻'),

    dict(id='lim-010', category='limit', difficulty='extreme',
         description='东1局早巡、无役、愚形、三立直：果断弃和，打唯一客风。',
         state=dict(hand=['1m','4m','7m','2p','5p','8p','1s','4s','7s','2z','3z','4z','5z'],
                    drawn='6z', dora_indicators=[],
                    rivers={'right':['7m','2p','9s','1z'],'across':['6m','2p','9s','2z'],
                            'left':['4m','8p','3s','3z']},
                    riichi={'right':True,'across':True,'left':True}, melds={},
                    round='东1局', self_wind='南', remaining=66,
                    scores={'own':25000,'right':25000,'across':25000,'left':25000}),
         kind='push_fold', direction='fold', stance='防守',
         safe=['4z','5z','6z'], must_avoid=['6z','7m']),

    # ===== 价值：多要素冲突 =====
    dict(id='lim-011', category='limit', difficulty='extreme',
         description='役牌对子+宝牌+门清三要素：打孤立客风，理由须点明役牌或宝牌价值。',
         state=dict(hand=['2m','3m','4m','5m','6m','5s','5s','4p','4p','4p','7z','7z','1z'],
                    drawn='9s', dora_indicators=['3p'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=64),
         kind='value', expected='9s', alternatives=['1z'],
         rationale_keywords=['役牌','宝牌']),

    dict(id='lim-012', category='limit', difficulty='extreme',
         description='速度vs打点：打4m立刻听牌但失去对子，保对子停留在1向听；正确是听牌，理由须说明速度/听牌。',
         state=dict(hand=['4m','4m','5m','6m','7m','8m','2p','3p','4p','5p','6p','7p','1z'],
                    drawn='1z', dora_indicators=['3m'], rivers={}, melds={},
                    round='东1局', self_wind='东', remaining=66),
         kind='value', expected='4m', alternatives=[],
         rationale_keywords=['听牌','速度']),
]
