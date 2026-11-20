"""Visible table evidence and concise strategy reference, independent of decisions."""
from collections import Counter
import re
from desktop_runtime import tile_index,tile_name,tile_code,group_tiles,build_safety_facts

REFERENCE = '''攻守参考框架：
逐家看威胁：已确认立直、庄家、多副露、公开役牌与可见宝牌、巡目；副露不等于已经听牌。
进攻比较候选的向听、有效进张、两面等好形、可成立的役与打点；宝牌不能单独成役。
七对子比普通型更近时应比较七对路线，不能把对子一律碰掉；鸣牌要参考程序给出的碰后向听。
攻守结合自身进度、听牌形状、价值、威胁家数、余牌与顺位；不能见立直就一律弃和，也不能好牌必推。
防守先核对该玩家牌河现物；对一家安全不等于对另一家安全。筋、壁、字牌均是有限风险线索，不能称绝对安全。
鸣牌要说明速度收益、成立的役、门清与防守损失；立直须门清且听牌，暗杠不破坏门清。
一向听仍未听牌，不能立直。白、发、中成刻或杠均有役牌，碰它们会破门清但不会失去该役牌。
普通型与七对子向听要分别引用，不能把总向听当普通型。主选若牺牲进张须说明原因，不能反称进张最多。
当前退守弃牌只能从手里选；对手牌河的现物若手里没有，就说明目前没有该现物可用。
立直还需至少一千点和足够余牌；立直后的杠必须符合不改变听牌等限制。
提出主选和备选，分别写何种摸牌或他家行动会改变选择。读不清的分数、立直与牌河写不确定，不编造精确放铳率或对手暗牌。
'''


def meld_description(group):
    names=[t.get('name') if isinstance(t,dict) else t for t in group_tiles(group)]
    named=[t for t in names if t]
    concealed=any(isinstance(t,dict) and t.get('kan') for t in group_tiles(group))
    kind='unknown'
    try:
        kinds=[tile_index(t) for t in named]
        if len(names)==4 and len(set(kinds))==1:kind='kan'
        elif len(names)==3 and len(kinds)==3:
            if len(set(kinds))==1:kind='pon'
            elif max(kinds)<27 and min(kinds)//9==max(kinds)//9 and sorted(kinds)==list(range(min(kinds),min(kinds)+3)):kind='chi'
    except (ValueError,TypeError):pass
    return {'type':kind,'tiles':named,'confidence':'low','concealed':concealed}


def public_tiles(value):
    if isinstance(value,str):return tile_name(value) if tile_code(value)==value else value
    if isinstance(value,list):return [public_tiles(v) for v in value]
    if isinstance(value,dict):return {public_tiles(k):public_tiles(v) for k,v in value.items()}
    return value


def table_profile(hand,rivers,melds,indicators,riichi=None):
    per_seat={}
    for seat in ('right','across','left'):
        groups=melds.get(seat) or []
        per_seat[seat]={'visible_discards':rivers.get(seat,[]),'discard_count':len(rivers.get(seat,[])),
                        'meld_count':len(groups),'melds':groups,
                        'riichi':(riichi or {}).get(seat),
                        'note':'牌河与副露来自有置信门槛的候选；没有读出立直不等于确认未立直'}
    # 只为画面中确实确认立直的座位生成现物/筋；未读出立直不等于立直，
    # 把三家一律当立直会喂给模型错误的“安全牌”事实。
    declared={s: bool((riichi or {}).get(s)) for s in per_seat}
    safety=build_safety_facts(rivers, declared)
    safety.pop('客风客三元',None)
    safety['说明']='仅对上表 riichi=true 的座位生成现物/筋；未确认立直的座位没有安全事实，不要套用他家现物。'
    return {'players':per_seat,'safety_by_player':safety,
            'instructions':'必须逐家结合可见牌河、副露与图中立直、点棒、局数判断；不能只分析自己手牌。'}


def shape_facts(hand):
    counts=Counter(tile_index(t) for t in hand)
    facts=[]
    for index,n in counts.items():
        if n>=2:facts.append({'tiles':[hand[next(i for i,t in enumerate(hand) if tile_index(t)==index)]]*2,'shape':'对子'})
        if index>=27:continue
        for gap in (1,2):
            other=index+gap
            if other//9!=index//9 or other not in counts:continue
            first=str(index%9+1)+'mps'[index//9]
            second=str(other%9+1)+'mps'[other//9]
            kind='嵌张' if gap==2 else ('边张' if index%9 in (0,7) else '两面')
            facts.append({'tiles':[first,second],'shape':kind})
    return {'patterns':facts,'note':'搭子可能重叠，不能把它们重复计为可同时使用的面子。'}


def repair_call_timing(strategy,hand):
    """Correct only an impossible action description, preserving model choices."""
    counts=Counter(tile_index(t) for t in hand)
    corrections=[]
    for field in ('offense','defense','call_plan','placement','next_turn_plan'):
        original=strategy.get(field)
        texts=original if isinstance(original,list) else [original]
        fixed=[]
        for text in texts:
            if not isinstance(text,str):fixed.append(text);continue
            for index,n in counts.items():
                name=tile_name(next(t for t in hand if tile_index(t)==index))
                pattern=r'摸到?'+re.escape(name)+r'[^。；;，,\n]{0,12}?(?:可以碰|可碰|碰出)'
                def replace(match):
                    if any(word in match.group() for word in ('不能','不可','不应','不要')):return match.group()
                    corrections.append('摸到自己的牌不能直接碰；碰牌须由他家打出。')
                    if n>=3:return '摸'+name+'形成四张时需核验杠的条件，不能把自摸称作碰牌'
                    if n==2:return '摸'+name+'可成暗刻；若仍持有对子，只有他家打出'+name+'时才可碰'
                    return '摸'+name+'后仍需先持有对子，并等他家打出'+name+'才可碰'
                text=re.sub(pattern,replace,text)
            fixed.append(text)
        strategy[field]=fixed if isinstance(original,list) else (fixed[0] if fixed else original)
    return list(dict.fromkeys(corrections))


def repair_rule_claims(strategy,hand):
    """Repair narrow rule contradictions; never replace the model's decision."""
    corrections=repair_call_timing(strategy,hand)
    def fix(text):
        if not isinstance(text,str):return text
        riichi=r'((?:[一二三四五六七八九]|[1-9])向听)(?:后|时|即可|就|直接|即|便|可|可以|能|能够|已)*立直'
        def riichi_fix(match):
            corrections.append('一向听及以上尚未听牌，不能立直；须门清听牌并符合点棒、余牌等条件。')
            return match.group(1)+'，仍须推进到门清听牌并核验点棒与余牌后才能立直'
        text=re.sub(riichi,riichi_fix,text)
        def dragon_fix(match):
            corrections.append('白、发、中成刻或杠有役牌；碰牌破门清，不会失去该役牌。')
            return '碰'+match.group(1)+'可获得役牌，但会破坏门清'
        text=re.sub(r'碰([白发中])(?:会|将|就|则)?(?:失去|失|丢失)(?:了)?役牌',dragon_fix,text)
        def pon_yaku_fix(match):
            names=re.split(r'[/、和或]',match.group(1))
            if not any(tile_code(v) and tile_index(tile_code(v))<27 for v in names):return match.group()
            corrections.append('碰普通数牌的刻子本身不提供役牌，鸣牌仍需另有成立的役。')
            return '；'.join('碰'+name+('本身不成役，须另有役' if tile_index(tile_code(name))<27 else
                            ('可得役牌' if tile_index(tile_code(name))>=31 else '是否有役需核对场风与自风')) for name in names)
        text=re.sub(r'碰((?:[一二三四五六七八九][万筒条]|[东南西北白发中])(?:[/、和或](?:[一二三四五六七八九][万筒条]|[东南西北白发中]))*)(?:得|获得|成)(?:一番)?役牌',pon_yaku_fix,text)
        return text
    def walk(value):
        if isinstance(value,list):return [walk(v) for v in value]
        if isinstance(value,dict):return {k:(v if k=='tile' else walk(v)) for k,v in value.items()}
        return fix(value)
    for key in list(strategy):strategy[key]=walk(strategy[key])
    return list(dict.fromkeys(corrections))


def check_table_claims(strategy,observed):
    """Check arithmetic and unconfirmed statuses without selecting a discard."""
    corrections=[]
    scores=observed.get('scores') or {}
    if set(scores)=={'own','right','across','left'} and sum(scores.values())%1000==0:
        own=scores['own'];rank=1+sum(v>own for v in scores.values())
        tied=sum(v==own for v in scores.values())>1
        higher=[v for v in scores.values() if v>own]
        fact='按本轮点棒读数，自己%d，%s第%d'%(own,'并列' if tied else '暂列',rank)
        if higher:fact+='，距前一档%d点'%(min(higher)-own)
        original=strategy.get('placement','')
        # Remove the model's contradictory ordinal, retaining its intended goal.
        pattern=r'(?:本家|自己|我方)(?:\s*\d+)?(?:分|点)?(?:居|为|处于|暂列|排在|排名|位列|排)?第[一二三四1234](?:位|名)?'
        cleaned=re.sub(pattern,'',original).lstrip('，,；;。 ')
        if cleaned!=original:corrections.append('顺位与点差按本轮读取的四家点棒重新核算；同分座次仍需规则确认。')
        strategy['placement']=fact+'。'+cleaned
    labels={'right':'下家','across':'对家','left':'上家'}
    def walk(value):
        if isinstance(value,list):return [walk(v) for v in value]
        if isinstance(value,dict):return {k:(v if k=='tile' else walk(v)) for k,v in value.items()}
        if not isinstance(value,str):return value
        for seat,label in labels.items():
            if (observed.get('riichi') or {}).get(seat) is None:
                updated=re.sub(label+r'(?:无立直|未立直|没有立直)',label+'立直状态未确认',value)
                if updated!=value:corrections.append('未读到立直不代表确认未立直，相关说法已降级为未确认。')
                value=updated
        if any((observed.get('riichi') or {}).get(seat) is None for seat in labels):
            updated=re.sub(r'各家(?:无立直|未立直|没有立直)','各家立直状态未确认',value)
            if updated!=value:corrections.append('未读到立直不代表确认未立直，相关说法已降级为未确认。')
            value=updated
        return value
    for key in list(strategy):strategy[key]=walk(strategy[key])
    return list(dict.fromkeys(corrections))
