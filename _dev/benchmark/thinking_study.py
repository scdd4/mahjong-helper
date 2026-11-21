"""高思考档位 & 策略规则注入：Jev 模式准确率对比实验。

配置矩阵（每种跑 59 场景）：
  off        关思考，当前 SYSTEM（v1.10.0 基线）
  low        开低思考，当前 SYSTEM
  high       开高思考，当前 SYSTEM
  high+rules 开高思考 + 优秀策略规则注入 SYSTEM
  off+rules  关思考 + 优秀策略规则注入 SYSTEM

用环境变量 STUDY_CONFIG 指定单个配置，便于并行/分批跑。
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from _dev.benchmark.run_benchmark import load_scenarios, build_payload, score
from desktop_runtime import load_settings
from spark_strategy import SparkClient
from jev_decision import (SYSTEM, TOKEN_FIRST_ID, MAX_CHOICES, candidates_from_calculations,
                          build_prompt, parse_logprobs)
from PIL import Image

settings = load_settings()

# 优秀策略规则（来自历史 A/B 研究：关思考退化为"见立直就保命"，
# 损失集中在攻守判断与打点。这些规则把 System 2 的结论显式化给 System One）。
RULES = '''
【优秀牌手策略规则】
1. 攻守不是"见立直就弃和"：南4局/末巡落后需要点数、自己是亲家、好形多面听牌、
   手上有2+宝牌时，即使对手立直也要保持进攻，弃牌从"保持听牌"的候选里选。
2. 只有无役、愚形单骑、巡目尚早且损失很小时才转防守；转防守必须从安全集（现物>筋>字牌）里挑，
   绝不打无筋中张。
3. 多立直时只认"对所有立直家共同安全"的牌；只对一家成立的现物不是全局安全。
   若无共同安全牌，必须明确承认，不能硬选。
4. 现物只对对应玩家成立；筋/客风/幺九只是相对风险线索，不是绝对安全。
5. 牌效率只是基础：向听相同看进张张数，但打点（宝牌、役牌、门清、立直）能改变取舍。
6. 自己已经听牌时，不要为了防守轻易拆听；保持听牌的攻守价值高于单张安全收益。
'''

CONFIGS = {
    'off':       (False, 'none', 64,   SYSTEM),
    'low':       (True,  'low',  256,  SYSTEM),
    'high':      (True,  'high', 1024, SYSTEM),
    'high+rules':(True,  'high', 1024, SYSTEM + RULES),
    'off+rules': (False, 'none', 64,   SYSTEM + RULES),
}


def run_one(sc, client, cfg):
    thinking, effort, budget, system = cfg
    full = build_payload(sc, settings)
    state = json.loads(full['messages'][1]['content'].split('：', 1)[1])
    hand = list(state['hand'])
    calc = state.get('local_calculations') or {}
    # 生产口径：ExactSolver 总给出按进张排序的候选表。基准场景自带 solver_table，
    # 之前只给 efficiency 类注入，导致攻守类拿到全手牌 14 张，人为拉低准确率。
    if not calc.get('candidates') and sc.get('solver_table'):
        calc = dict(calc)
        calc['candidates'] = [
            {'tile': row['drop'], 'shanten': row['shanten'], 'ukeire': row['ukeire']}
            for row in sorted(sc['solver_table'], key=lambda r: (r['shanten'], -r['ukeire']))[:8]]
    candidates = candidates_from_calculations(calc, hand, state) or list(dict.fromkeys(hand))[:14]
    evidence = dict(state)
    evidence['hand'] = {'reliable': True, 'tiles': [{'name': t} for t in hand]}
    evidence['warnings'] = []
    user_text = build_prompt(hand, calc, evidence, candidates)
    payload = {'model': settings.model,
               'messages': [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': [{'type': 'text', 'text': user_text}]}],
               'stream': False, 'max_tokens': budget,
               'logprobs': True, 'top_logprobs': MAX_CHOICES,
               'allowed_token_ids': [TOKEN_FIRST_ID + i for i in range(len(candidates))],
               'reasoning_effort': effort,
               'chat_template_kwargs': {'enable_thinking': thinking},
               'strata_checkpoint': False}
    with Image.new('RGB', (32, 32)):
        t0 = time.perf_counter()
        reply = client._jev_request(payload, None)
        tile, conf, _ = parse_logprobs(reply['choices'][0], candidates)
        elapsed = time.perf_counter() - t0
    return tile, conf, elapsed


def judge(sc, tile):
    if sc['kind'] == 'value':
        exp = sc['expected']
        return tile == exp['tile'] or tile in exp.get('alternatives', [])
    answer = {'discard': tile, 'stance': '均衡', 'reason': 'jev', 'offense': '', 'defense': ''}
    passed, _ = score(sc, answer)
    return passed


def main():
    cfg_name = os.environ.get('STUDY_CONFIG', 'off')
    cfg = CONFIGS[cfg_name]
    scenarios = load_scenarios()
    ids = os.environ.get('STUDY_IDS', '').strip()
    if ids:
        wanted = set(ids.split(','))
        scenarios = [s for s in scenarios if s['id'] in wanted]
    limit = int(os.environ.get('STUDY_LIMIT', '0'))
    if limit:
        scenarios = scenarios[:limit]
    client = SparkClient(settings)
    results = []
    for sc in scenarios:
        try:
            tile, conf, elapsed = run_one(sc, client, cfg)
            passed = judge(sc, tile)
        except Exception as error:
            tile, conf, elapsed, passed = None, 0, 0, False
            print('[%s] ERROR %s: %s' % (cfg_name, sc['id'], str(error)[:80]), flush=True)
        results.append({'id': sc['id'], 'kind': sc['kind'], 'passed': passed,
                        'seconds': round(elapsed, 2), 'confidence': round(conf, 3), 'tile': tile})
        print('[%s] %-5s %-6s %-10s tile=%-4s conf=%.2f %.1fs' %
              ('OK' if passed else 'BAD', cfg_name, sc['id'], sc['kind'],
               str(tile)[:4], conf, elapsed), flush=True)
    ok = sum(1 for r in results if r['passed'])
    lat = [r['seconds'] for r in results if r['seconds'] > 0]
    print('\n===== %s: %d/%d (%.0f%%) median %.2fs =====' %
          (cfg_name, ok, len(results), 100 * ok / len(results),
           statistics.median(lat) if lat else 0), flush=True)
    out = HERE / ('study-%s.json' % cfg_name.replace('+', '_'))
    out.write_text(json.dumps({'config': cfg_name, 'passed': ok, 'total': len(results),
                               'accuracy': ok / len(results),
                               'median_s': statistics.median(lat) if lat else 0,
                               'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written', out, flush=True)


if __name__ == '__main__':
    main()
