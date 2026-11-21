"""把服务器 eval-XXX.json 的 discards 用统一 score() 评分，产出可比报告。

用法：
    python _dev/score_eval.py _dev/benchmark/eval-4b-server.json --name 4B微调 --baseline 81
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _dev.benchmark.run_benchmark import load_scenarios, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--name', default='model')
    ap.add_argument('--baseline', type=float, default=0.81)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding='utf-8'))
    preds = {r['id']: r['discard'] for r in data['results']}
    times = [r['seconds'] for r in data['results'] if r.get('seconds', 0) > 0]
    scenarios = {s['id']: s for s in load_scenarios()}

    by_kind = defaultdict(lambda: [0, 0])
    details = []
    for sid, sc in scenarios.items():
        discard = preds.get(sid, '')
        if sc['kind'] == 'value':
            exp = sc['expected']
            passed = discard == exp['tile'] or discard in exp.get('alternatives', [])
        else:
            ans = {'discard': discard, 'stance': '均衡', 'reason': '', 'offense': '', 'defense': ''}
            passed, _ = score(sc, ans)
        by_kind[sc['kind']][0] += passed
        by_kind[sc['kind']][1] += 1
        details.append((sid, sc['kind'], passed, discard))

    total = sum(v[0] for v in by_kind.values())
    n = sum(v[1] for v in by_kind.values())
    print('===== %s: %d/%d (%.0f%%) 中位延迟 %.2fs =====' %
          (args.name, total, n, 100 * total / n,
           statistics.median(times) if times else 0))
    for kind, (ok, cnt) in sorted(by_kind.items()):
        print('  %-12s %d/%d %.0f%%' % (kind, ok, cnt, 100 * ok / cnt))
    print('  对比当前 Jev 基线 81%%: %+.0f 个百分点' % (100 * total / n - args.baseline * 100))
    print('\n失败场景:', ' '.join(sid for sid, _, ok, _ in details if not ok))


if __name__ == '__main__':
    main()
