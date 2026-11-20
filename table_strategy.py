"""Serial thinking requests: fast advice, then concrete whole-table strategy."""
from collections import Counter
import json
import re
import time
from desktop_runtime import tile_name,tile_code,tile_index,humanize_tiles,dora_from_indicator,resolve_tile_choice
from strategy_context import REFERENCE,public_tiles,table_profile,meld_description,shape_facts,repair_rule_claims,check_table_claims
from scoreboard_view import scoreboard_panel

QUICK_SYSTEM = '''你是四人立直麻将教练。先快速核对全桌并给初步建议，只输出紧凑JSON：
{"discard":"中文牌名或null","reason":"具体理由，15字内","stance":"进攻/防守/均衡/等待/待核验"}。不要输出其他字段，场面详解将在下一阶段完成。
手牌以输入核验值为准。有效13张必须等待摸牌，14张才从手牌选弃牌；杠算3张有效牌。
结合各家牌河、副露、宝牌、立直威胁和图中局数点差判断，不只看自己手牌。现物只对对应玩家成立，不能把字牌或筋当绝对安全。
只用万、筒、条和汉字字牌，不用英文牌码；简短思考，不写内部推理。'''

DETAIL_SYSTEM = '''你是四人立直麻将教练，依据本地核验事实、逐家牌河/副露候选与已按座位标注的点棒放大图，补全具体攻守方案。
本家可靠手牌必须锁定；对手牌河与副露来自初判阶段的结构化候选，低置信项已标注，只能当“图上疑似”，不能当成定论。不要编造未知信息。只输出紧凑JSON，正文合计约220个中文字：
{"discard":"中文牌名或null","reason":"关键取舍，20字内","stance":"进攻/防守/均衡/等待/待核验","offense":"具体成役、速度打点取舍，30字内","defense":"逐家威胁与可用安全牌，40字内","next":"两种具体摸牌或对手行动后的改选，35字内","alternatives":[{"tile":"中文牌名","when":"改选条件","reason":"取舍"}],"call_plan":"鸣牌及立直条件，25字内","placement":"点差目标；未知则说明","threats":["逐家场面证据，最多3条"],"table":{"round":null,"self_wind":null,"remaining":null,"scores":{},"riichi":{}}}。
点棒放大图已按自己/下家/对家/上家标注并旋转，分数按标签读取，风位看单独标注的本家风位；红色轮转标记不等于本家。scores/riichi座位键own/right/across/left。未知写null，不能填历史或猜测。向听和鸣牌后的数值只用程序提供的核算，不自行编写新数字。route_facts逐路线核算，不能混用普通型和七对子数字；一向听不能立直，碰白发中可得役牌。防守候选要在自己手中；没有现物须说没有，不能让用户打手里没有的牌。
有效13张只能等待摸牌；14张才考虑弃牌。local_calculations.win=true只是和型完成，须核验自摸/荣和与役，不应强迫拆牌。牌名只用万、筒、条和汉字字牌。思考简洁，不输出内部推理。
七对子摸牌举例只能从route_facts.seven_pairs.new_pair_draws的现有单张中选；手里没有的牌摸一张不能直接成对子。未确认立直只能说未知；点棒顺位按四家数字比较。
'''+REFERENCE


def _text(value):
    if isinstance(value,str):return humanize_tiles(value)
    if isinstance(value,list):return [_text(v) for v in value]
    if isinstance(value,dict):return {k:_text(v) for k,v in value.items()}
    return value


def analyze_table(client,image,evidence,calculations=None,history=None,progress=None):
    from spark_strategy import Cancelled,hand_tile_total
    if not client.settings.api_key:raise ValueError('请先在模型与截图设置中填写密钥')
    if not client._lock.acquire(blocking=False):raise RuntimeError('请等待上一次分析结束')
    if not getattr(client, '_shared_cancel', False):
        client._cancel.clear()
    started=time.perf_counter()
    try:
        own=evidence.get('hand') or {}
        if not own.get('reliable'):
            raise ValueError('手牌尚未核验，请让牌面完整清晰后重新截图。')
        verified=[t['name'] for t in own.get('tiles',[]) if t.get('name')]
        total=hand_tile_total(verified,evidence)
        notes=list(evidence.get('warnings') or [])
        rivers={seat:[v['name'] for v in items if v.get('name') and (v.get('score') or 0)>=.65]
                for seat,items in (evidence.get('rivers') or {}).items()}
        melds={seat:[meld_description(group) for group in (data.get('groups') or [])]
               for seat,data in (evidence.get('melds') or {}).items()}
        indicators=[t for t in (evidence.get('dora_indicators') or []) if tile_code(t)]
        dora=[dora_from_indicator(t) for t in indicators]
        # 'candidate' = 本地几何推测的立直，同样纳入安全事实（保守），
        # 但 notes 已标注需复核；不完全依赖它做弃和决定。
        riichi={s:True for s,v in (evidence.get('riichi') or {}).items() if v is True or v=='candidate'}
        if riichi:notes.append('本地检测到立直棒，需结合图复核')
        physical=Counter(tile_index(t) for t in verified+indicators)
        for river in rivers.values():physical.update(tile_index(t) for t in river)
        for groups in melds.values():
            for group in groups:physical.update(tile_index(t) for t in group['tiles'])
        overflow=any(n>4 for n in physical.values())
        context={'hand':verified,'hand_total':total,'effective_total':total,'dora_indicators':indicators,'dora_tiles':dora,
                 'shape_facts':shape_facts(verified),
                 'local_calculations':calculations,'table':table_profile(verified,rivers,melds,indicators,riichi),
                 'riichi':riichi,'physical_overflow':overflow,
                 'recent_context':(history or [])[-2:],
                 'uncertainties':notes[:3]}
        from open_efficiency import possible_pon_plans,possible_chi_plans,possible_kan_plans,closed_route_facts
        seen=[code for code,n in ((calculations or {}).get('seen') or {}).get('by_tile',{}).items() for _ in range(n)]
        own_groups=(evidence.get('melds') or {}).get('own',{}).get('groups',[])
        context['pon_options']=possible_pon_plans(verified,own_groups,seen)
        context['chi_options']=possible_chi_plans(verified,own_groups,seen)
        context['kan_options']=possible_kan_plans(verified,own_groups,seen)
        context['route_facts']=closed_route_facts(verified,(evidence.get('melds') or {}).get('own',{}).get('groups',[]))
        content=[{'type':'text','text':json.dumps(public_tiles(context),ensure_ascii=False,separators=(',',':'))}]
        viewport=evidence.get('viewport') or [0,0,image.width,image.height]
        picture=image.crop(tuple(viewport))
        try:
            content.append(client._image_part(picture,max_width=960))
            with scoreboard_panel(picture) as panel:
                content.append({'type':'text','text':'点棒与风位放大图：以图上座位标签为准。'})
                content.append(client._image_part(panel,max_width=600))
        finally:picture.close()
        first_advice=None
        def on_progress(event):
            nonlocal first_advice
            partial=event.get('partial_text','')
            if first_advice is None and not overflow:
                match=re.search(r'"discard"\s*:\s*(null|"(?:[^"\\]|\\.)*")',partial)
                reason=re.search(r'"reason"\s*:\s*("(?:[^"\\]|\\.)*")',partial)
                if match and reason:
                    candidate=resolve_tile_choice(json.loads(match.group(1)),verified) if match.group(1)!='null' else None
                    if total==13 or (total==14 and candidate in verified):
                        candidate=candidate if total==14 else None
                        first_advice=round(time.perf_counter()-started,2)
                        if progress:progress({'phase':'advice','tile':candidate,
                                              'title':'初步建议：'+('打'+tile_name(candidate) if candidate else '等待摸牌'),
                                              'reason':humanize_tiles(json.loads(reason.group(1)))})
            if progress:progress({k:v for k,v in event.items() if k!='partial_text'})
        if progress:progress({'phase':'structured'})
        payload={'model':client.settings.model,'messages':[{'role':'system','content':QUICK_SYSTEM},{'role':'user','content':content[:2]}],
                 'stream':True,'max_tokens':160,'reasoning_effort':'low',
                 'chat_template_kwargs':{'enable_thinking':True},'strata_prefix':{'messages':1}}
        try:quick=client._request_with_retry(payload,on_progress,parse_mode='quick')
        except RuntimeError as error:
            if '输出达到上限' not in str(error):raise
            payload['max_tokens']=256
            quick=client._request_with_retry(payload,on_progress,parse_mode='quick')
        if client._cancel.is_set():raise Cancelled('分析已取消')
        environment=quick['report'].get('table') or {}
        detail_context={**context,'environment':environment,'first_decision':quick['report']}
        detail_payload={'model':client.settings.model,'messages':[{'role':'system','content':DETAIL_SYSTEM},
                        # 速度优化：quick 阶段已看过整桌图，详解只需点棒放大图(content[2:])；
                        # 整桌事实已由结构化 context 与 strata_prefix 的 KV 缓存承载。
                        {'role':'user','content':[{'type':'text','text':json.dumps(public_tiles(detail_context),ensure_ascii=False,separators=(',',':'))}]+content[2:]}],
                        'stream':True,'max_tokens':1100,'reasoning_effort':'low',
                        'chat_template_kwargs':{'enable_thinking':True},'strata_prefix':{'messages':1}}
        detail_started=time.perf_counter()
        detail_failed=False
        try:
            reply=client._request_with_retry(detail_payload,progress,parse_mode='fast')
        except Cancelled:raise
        except (RuntimeError,ValueError,TimeoutError) as error:
            detail_failed=True
            notes.append('场面详解尚未完成，当前保留初步建议：'+str(error))
            reply={**quick,'report':{**quick['report'],'offense':'详解生成未完成','defense':quick['report'].get('defense','场况待确认'),
                                    'next':'牌面变化后重新截图'}}
        answer=reply['report']
        discard=resolve_tile_choice(answer.get('discard'),verified)
        if total==13:discard=None
        if discard and discard not in verified:
            notes.append('建议牌不在手中，当前行动已隐藏');discard=None
        observed={'hand':verified,'round':None,'self_wind':None,'remaining':None,'scores':{},
                  'riichi':riichi,'rivers':rivers,'melds':melds,'dora_indicators':indicators,'dora_tiles':dora}
        table=answer.get('table') or environment
        if not isinstance(table,dict):table={}
        for field in ('round','self_wind'):
            if isinstance(table.get(field),str):observed[field]=table[field][:24]
        remaining=table.get('remaining')
        if isinstance(remaining,int) and not isinstance(remaining,bool) and 0<=remaining<=100:observed['remaining']=remaining
        for field in ('scores','riichi'):
            for seat,value in (table.get(field) or {}).items() if isinstance(table.get(field),dict) else []:
                if seat not in ('own','right','across','left'):continue
                if field=='scores' and isinstance(value,int) and not isinstance(value,bool) and -100000<=value<=1000000:
                    observed[field][seat]=value
                if field=='riichi' and (isinstance(value,bool) or value is None):observed[field][seat]=value
        alternatives=[]
        for item in answer.get('alternatives') or []:
            if isinstance(item,dict) and tile_code(item.get('tile')) in verified:
                alternatives.append({**_text(item),'tile':tile_code(item['tile'])})
        strategy={'stance':'待核验' if overflow else answer.get('stance','待核验'),
                  'recommendation':{'action':'打牌' if discard else ('等待摸牌' if total==13 else '待核验'),
                                    'tile':discard,'reason':_text(answer.get('reason',''))},
                  'offense':_text(answer.get('offense','')),'defense':_text(answer.get('defense','')),
                  'alternatives':alternatives,'next_turn_plan':_text(answer.get('next',[])),
                  'call_plan':_text(answer.get('call_plan','本轮模型未给出鸣牌条件')),
                  'placement':_text(answer.get('placement','点差尚未确认')),
                  'table_threats':_text(answer.get('threats',[])),
                  'uncertainties':notes,'confidence':'low' if overflow else 'medium'}
        if isinstance(strategy['next_turn_plan'],str):strategy['next_turn_plan']=[strategy['next_turn_plan']]
        corrections=repair_rule_claims(strategy,verified)
        corrections+=check_table_claims(strategy,observed)
        if corrections:reply['rule_corrections']=corrections
        if (calculations or {}).get('win') and total==14:
            strategy['recommendation']={'action':'和牌核验','tile':None,'reason':'已成和型，请结合自摸或荣和按钮及役种核验。'}
        if len(observed['scores'])==4:
            stick_count=sum(1 for v in (evidence.get('riichi') or {}).values() if v is True or v=='candidate')
            expected_total=100000-1000*stick_count
            actual_total=sum(observed['scores'].values())
            if actual_total!=expected_total:
                strategy['placement']='点棒合计与立直棒数不符（应为%d，读得%d），暂不据此判断顺位。' % (expected_total,actual_total)
                notes.append('四家点棒合计异常，点差判断待核对。')
        reply.update(report={'observed':observed,'strategy':strategy},hand_source='本地核验',
                     vision_uncertainties=notes,mode='structured',thinking_stages={'structured':'low'},
                     stage_seconds={'quick':quick['elapsed_seconds'],'detail':round(time.perf_counter()-detail_started,2)},
                     stage_usage={'quick':quick['usage'],'detail':reply.get('usage')},
                     first_advice_seconds=first_advice,detail_incomplete=detail_failed)
        reply['elapsed_seconds']=round(time.perf_counter()-started,2)
        return reply
    finally:
        client._socket=None;client._lock.release()
