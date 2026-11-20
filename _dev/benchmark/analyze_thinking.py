#!/usr/bin/env python
"""配对分析思考开关 A/B：找出四轮结果文件，做同题对比与分类拆解。"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RUNS = sorted(HERE.glob('results-*.json'))


def load_config(path):
    """从文件名时间戳区分四轮；配置由调用方按顺序指定。"""
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(files, labels):
    data = [load_config(f) for f in files]
    maps = []
    for d, label in zip(data, labels):
        m = {r['id']: r for r in d['results'] if not str(r.get('detail','')).startswith('调用失败')}
        maps.append((label, m))
        ok = sum(1 for r in m.values() if r['passed'])
        secs = [r['seconds'] for r in m.values()]
        print(f'{label:12s} {ok}/{len(m)} ({ok/len(m)*100:.0f}%) 中位 {statistics.median(secs):.1f}s')
    print()

    ids = sorted(set(maps[0][1]) & set(maps[1][1]))
    # 前两轮配对：on vs off
    on, off = maps[0][1], maps[1][1]
    on_only = [i for i in ids if on[i]['passed'] and not off[i]['passed']]
    off_only = [i for i in ids if off[i]['passed'] and not on[i]['passed']]
    both = [i for i in ids if on[i]['passed'] and off[i]['passed']]
    neither = [i for i in ids if not on[i]['passed'] and not off[i]['passed']]
    print(f'配对(R1): 开对关错={len(on_only)} {on_only}')
    print(f'          关对开错={len(off_only)} {off_only}')
    print(f'          都对={len(both)} 都错={len(neither)}')

    # 分类拆解：思考对哪类更重要
    cats = {}
    for i in ids:
        c = on[i]['category']
        cats.setdefault(c, {'on': 0, 'off': 0, 'n': 0})
        cats[c]['n'] += 1
        cats[c]['on'] += int(on[i]['passed'])
        cats[c]['off'] += int(off[i]['passed'])
    print('\n分类(R1):')
    for c, v in sorted(cats.items()):
        print(f'  {c:12s} 开 {v["on"]}/{v["n"]}  关 {v["off"]}/{v["n"]}')

    # 稳定性：on 两轮、off 两轮各自内部一致性
    print('\n稳定性:')
    for a, b, label in ((0, 2, 'thinking=on'), (1, 3, 'thinking=off')):
        ma, mb = maps[a][1], maps[b][1]
        common = set(ma) & set(mb)
        flip = [i for i in common if ma[i]['passed'] != mb[i]['passed']]
        print(f'  {label}: 两轮翻转 {len(flip)} 题 {flip}')


if __name__ == '__main__':
    import sys
    files = [HERE / f for f in sys.argv[1:5]]
    labels = sys.argv[5:9] if len(sys.argv) > 8 else ['on-R1', 'off-R1', 'on-R2', 'off-R2']
    analyze(files, labels)
