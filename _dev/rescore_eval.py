"""从服务器评测结果的 reply 字段重新提取牌名并评分。

评测脚本首次解析有 bug（skip_special_tokens 剥掉  thinking 的尖括号），
但 reply 字段保留了完整文本，可直接用正则提取牌名，无需重跑评测。

用法：
    python _dev/rescore_eval.py _dev/benchmark/eval-4b-server.json --name 4B微调
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _dev.benchmark.run_benchmark import load_scenarios, score


def extract_tile(reply):
    """优先取竖线前的牌名（训练格式 牌名|理由），否则取任一牌名。"""
    if not reply:
        return ''
    # 先试 "牌名|" 模式，这是训练时的目标格式
    m = re.search(r'([1-9][mpsz])\s*\|', reply)
    if m:
        return m.group(1)
    # 再试任意位置的牌名
    m = re.search(r'([1-9][mpsz])', reply)
    return m.group(1) if m else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--name', default='model')
    ap.add_argument('--baseline', type=float, default=0.81)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding='utf-8'))
    preds = {r['id']: extract_tile(r.get('reply', '') or r.get('discard', ''))
             for r in data['results']}
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
        details.append((sid, sc['kind'], passed, discard, sc['expected'].get('tile') or sc['expected'].get('safe_set') or 'safe'))

    total = sum(v[0] for v in by_kind.values())
    n = sum(v[1] for v in by_kind.values())
    acc = total / n
    print('===== %s: %d/%d (%.0f%%) 中位延迟 %.2fs =====' %
          (args.name, total, n, 100 * acc,
           statistics.median(times) if times else 0))
    for kind, (ok, cnt) in sorted(by_kind.items()):
        print('  %-12s %d/%d %.0f%%' % (kind, ok, cnt, 100 * ok / cnt))
    print('  对比当前 Jev 基线 %.0f%%: %+.0f 个百分点' % (args.baseline * 100, 100 * acc - args.baseline * 100))
    print('\n失败场景（期望→实际）:')
    for sid, kind, ok, got, exp in details:
        if not ok:
            print('  %-9s %-10s 期望 %-3s 得到 %-3s' % (sid, kind, exp, got or '(空)'))

    if args.out:
        Path(args.out).write_text(json.dumps({
            'model': args.name, 'passed': total, 'total': n, 'accuracy': acc,
            'median_s': statistics.median(times) if times else 0,
            'by_kind': {k: {'passed': v[0], 'total': v[1]} for k, v in by_kind.items()},
            'failures': [{'id': s, 'kind': k, 'expected': str(e), 'got': g}
                         for s, k, ok, g, e in details if not ok],
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print('\n写入', args.out)


if __name__ == '__main__':
    main()