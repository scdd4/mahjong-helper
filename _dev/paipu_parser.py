"""天凤牌谱（mjloggm v2.3）解析器：重放动作流，重建每个决策时刻的局面。

用途：从高段真人牌谱提取 (局面事实, 真人弃牌) 监督样本，供 Jev-like 微调与真人基准。

牌谱格式要点（已实测）：
- <INIT seed ten oya hai0..hai3>  每局起始手牌（tile index 0-133）
- <N who m>  动作：m<32768=自家摸切牌 index；32768<=m<65536=荣和/加杠等；
             m>=65536=吃碰杠（低字节编码副露）
- <DORA hai> 宝牌指示牌翻开
- <REACH who step> 立直宣言（step=1 声明 / step=2 成立）
- <AGARI>/<RYUUKYOKU> 局结束

tile index -> 牌名：0-33 万、34-67 筒、68-101 索、102-134 字；
index%4==3 且 102<=index<=134 为赤五（0m/0p/0s 对应 index 16/52/88 的第四张）。

用法：
  python _dev/paipu_parser.py _dev/paipu_sample.xml --out _dev/paipu_sample.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

TILE_RE = re.compile(r'^(?:[0-9][mps]|[1-7]z)$')


def tile_name(idx):
    """天凤 tile index -> 本项目牌名（m/p/s/z，0=赤五）。

    天凤编码：index = tile_type*4 + copy，tile_type 0-8 万/9-17 筒/18-26 索/27-33 字。
    赤五固定 index：0m=16, 0p=52, 0s=88。
    """
    idx = int(idx)
    if idx in (16, 52, 88):
        return '0' + 'mps'[(idx // 4) // 9]
    t = idx // 4
    if t < 27:
        return str(t % 9 + 1) + 'mps'[t // 9]
    return str(t - 26) + 'z'


def parse(xml_text, min_rate=2000):
    """返回决策样本列表。每个样本 = 一个玩家在摸牌后、弃牌前的局面 + 真人弃牌。"""
    un = re.search(r'<UN ([^>]+)>', xml_text)
    if not un:
        return [], 'no UN'
    rates = [float(x) for x in (re.search(r'rate="([^"]+)"', un.group(1)) or ['', ''])[1].split(',')]
    if max(rates or [0]) < min_rate:
        return [], f'low rate {rates}'

    samples = []
    inits = re.finditer(r'<INIT ([^>]+)>', xml_text)
    # 按事件顺序处理：INIT 重置局面，N/DORA/REACH/AGARI 更新
    events = []
    for m in re.finditer(r'<(INIT|N|DORA|REACH|AGARI|RYUUKYOKU) ([^>]*)/?>', xml_text):
        events.append((m.group(1), m.group(2)))

    hands = [set() for _ in range(4)]   # 各家手牌 index（13 张基准态）
    rivers = [[] for _ in range(4)]     # 各家牌河牌名
    riichi = [False] * 4
    dora_ind = []
    turn = 0
    meld_seat = set()

    for tag, attrs in events:
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        if tag == 'INIT':
            hands = []
            for seat in range(4):
                hs = a.get(f'hai{seat}', '')
                hands.append(set(int(x) for x in hs.split(',') if x))
            rivers = [[] for _ in range(4)]
            riichi = [False] * 4
            turn = 0
            meld_seat = set()
            dora_ind = [tile_name(int(a['seed'].split(',')[5]))] if a.get('seed') else []
        elif tag == 'DORA':
            dora_ind.append(tile_name(int(a['hai'])))
        elif tag == 'REACH':
            riichi[int(a['who'])] = True
        elif tag == 'N':
            who = int(a['who'])
            m = int(a['m'])
            if m >= 65536:  # 副露/杠：标记弃和样本，本轮无法重建手牌
                meld_seat.add(who)
                continue
            t = m & 255  # 天凤 N 动作的低 8 位是 tile index（实测）
            if t > 134:
                continue
            discard = tile_name(t)
            # 决策时手牌 = 摸牌态（13 张 + 本次打出的牌）。
            # 天凤 N 事件是"摸切"合并表示：本次打出的 t 必然来自摸牌或手牌。
            hand_at_decision = sorted(tile_name(i) for i in (hands[who] | {t}))
            if len(hand_at_decision) == 14 and who not in meld_seat:
                samples.append({
                    'seat': who,
                    'hand': hand_at_decision,
                    'discard': discard,
                    'rivers': {f'seat{i}': list(rivers[i]) for i in range(4)},
                    'riichi': {f'seat{i}': riichi[i] for i in range(4)},
                    'dora_indicators': list(dora_ind),
                    'turn': turn,
                    'melded': who in meld_seat,
                })
            rivers[who].append(discard)
            hands[who].discard(t)
            turn += 1
        elif tag in ('AGARI', 'RYUUKYOKU'):
            pass
    return samples, f'rate={rates}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xml')
    ap.add_argument('--out', default='')
    ap.add_argument('--min-rate', type=int, default=2000)
    args = ap.parse_args()
    text = Path(args.xml).read_text(encoding='utf-8')
    samples, info = parse(text, args.min_rate)
    print(f'解析 {args.xml}: {info}, 决策样本 {len(samples)}')
    if samples:
        s = samples[0]
        print('样例:', json.dumps(s, ensure_ascii=False)[:500])
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print('写入', args.out)


if __name__ == '__main__':
    main()
