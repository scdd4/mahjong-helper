"""A/B 实验：把人类麻将策略的量化规则注入 benchmark system prompt。

不修改生产代码、不改评分、不改场景。只替换 system prompt，用同一套 59 题对比：
  base      — 当前 BENCH_SYSTEM_PROMPT（基线，预期 ~81%）
  human     — 注入安全度排序 + 巡目/宝牌修正 + 攻守番数门槛
  human_lite— 只注入不与基准安全集定义冲突的部分

用法：
  python _dev/benchmark/ab_human_strategy.py --variant human
  python _dev/benchmark/ab_human_strategy.py --variant human_lite
  python _dev/benchmark/ab_human_strategy.py --variant base
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_benchmark as RB
from desktop_runtime import load_settings

BASE = RB.BENCH_SYSTEM_PROMPT

HUMAN_RULES = '''
【人类高手量化策略（天凤凤桌统计 / 《科学する麻雀》）】
安全度从高到低（对同一立直家）：现物(0%) ≈ 场上见3张的字牌 > 见2张字牌 > 筋19/两筋456(约2%) > 筋28/筋37(4-6%) > 生张字牌(晚巡危险,勿当安全) > 无筋19(7-10%) > 无筋28/37 > 无筋456(13%,最危险)。
· 筋只对两面有效，对嵌张/边张/双碰/单骑无效——无筋不等于必铳，筋也不等于绝对安全。
· 宝牌及其周边风险上调2-3级；立直家若打了红5，无筋4/6反而相对安全。
· 巡目≥10的晚巡：生张字牌与无筋456危险度显著上升；现物始终最安全。
攻守判别（ D = 和率×(和了点+1500) − 1500 − 放铳风险 ）：
· 门槛：自己是亲家1番即可对攻；子家对子家需2番；子家对亲家需3番；愚形(嵌张/边张/单骑)+1番；需切危险牌+1番；两家以上立直需4-5番。
· 和率−铳率≥10 才值得攻；否则弃和。决定弃和就一路打现物/高安全牌到底，禁止半攻半守打筋牌试探。
· 好形多面听、2+宝牌、末巡落后追点、ALL LAST争位 → 即使被立直也进攻，弃牌从 tenpai 保持集合选。
打点：孤张宝牌不要为保留而牺牲效率；有役(役牌/立直/门清)+宝牌的复合手才值得为打点放慢速度。
'''

HUMAN_LITE = '''
【人类策略补充（不覆盖上方安全集定义）】
· 安全度细排：现物 > 见3张字牌 > 见2张字牌 > 筋19 > 筋28/筋37 > 无筋19 > 无筋456。筋仅对两面有效。
· 宝牌及周边风险上调；晚巡生张字牌危险度上升，但【本基准的 safe 集合字段优先】，选牌仍须落在 safe 列表内。
· 攻守：亲家1番、子家2番、对亲家3番可对攻；愚形+1、切危险牌+1、多家立直需4-5番。和率−铳率≥10 才攻。
· 弃和必须彻底，只打安全牌；进攻时弃牌从 tenpai 保持集合选。
· 打点：保留有役+宝牌的复合手，孤张宝牌不为打点牺牲效率。
'''

VARIANTS = {'base': BASE, 'human': BASE + '\n' + HUMAN_RULES,
            'human_lite': BASE + '\n' + HUMAN_LITE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=list(VARIANTS), required=True)
    ap.add_argument('--thinking', choices=('on', 'off'), default='off')
    ap.add_argument('--effort', choices=('low', 'medium', 'high'), default='low')
    ap.add_argument('--max-tokens', type=int, default=240)
    args = ap.parse_args()

    RB.BENCH_SYSTEM_PROMPT = VARIANTS[args.variant]
    RB.EXPERIMENT.update(thinking=args.thinking == 'on', effort=args.effort,
                         max_tokens=args.max_tokens)
    settings = load_settings()
    # 实验期间 Spark 服务只绑服务器 127.0.0.1，走本地 SSH 隧道（_dev/tunnel_forward.py）。
    if os.environ.get('AB_USE_TUNNEL'):
        settings.base_url = 'http://127.0.0.1:18080/v1'
    scenarios = RB.load_scenarios()
    print(f'[AB] variant={args.variant} thinking={args.thinking} effort={args.effort} '
          f'max_tokens={args.max_tokens} model={settings.model}', flush=True)
    t0 = time.perf_counter()
    results = RB.run(scenarios, None, None, settings)
    elapsed = time.perf_counter() - t0
    out = HERE / f'ab-{args.variant}-{args.thinking}-{args.max_tokens}.json'
    out.write_text(json.dumps({'variant': args.variant, 'config': dict(RB.EXPERIMENT),
                               'results': results}, ensure_ascii=False, indent=1), encoding='utf-8')
    RB.report(results)
    print(f'[AB] 结果写入 {out} | 总耗时 {elapsed:.1f}s')
    # 覆盖 report 里的 results-<ts>.json，避免污染历史目录
    for p in sorted(HERE.glob('results-2026*.json')):
        if p.stat().st_mtime > t0 + 1e9:
            pass


if __name__ == '__main__':
    main()
