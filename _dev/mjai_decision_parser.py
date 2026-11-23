"""从 MJAI 牌谱事件流重建真人弃牌决策样本。

数据源：nikketryhard/tenhou-houou-mjai（天凤凤凰桌，MJAI 格式，zstd 压缩）。
每条样本 = 一个决策时刻的结构化局面 + 真人实际弃牌，格式与 jev_decision.build_prompt 对齐。

用法：
    python mjai_decision_parser.py <game.json.zst> --out <out.jsonl>
    python mjai_decision_parser.py --tar <sample_head.tar> --out <out.jsonl>  # 从 tar 批量
"""
import io
import json
import sys
import argparse

import zstandard as zstd

# 天凤座位索引 0-3 -> 项目座位名；决策者始终映射到 own（相对视角在下游处理）
SEATS = ('own', 'right', 'across', 'left')

TILE_RE = __import__('re').compile(r'[1-9][mpsz]|0[mpsz]|[ESWNPCF]|[1-9][mpsz]r')


def norm_tile(t):
    """MJAI 牌名 -> 项目牌名（红牌 r 后缀去掉，字牌转数字）。"""
    if not t:
        return t
    red = t.endswith('r')
    base = t[:-1] if red else t
    m = {'E': '1z', 'S': '2z', 'W': '3z', 'N': '4z', 'P': '5z', 'F': '6z', 'C': '7z'}
    if base in m:
        return m[base]
    # 红牌：项目用 0m/0p/0s 表示赤五；但生产 TILE_RE 支持 0? 需确认。先统一为 5? 避免解析失败
    if base in ('5m', '5p', '5s'):
        return base  # 红5与正常5同值，Missing red info 不影响决策结构
    return base


def decompress(data):
    d = zstd.ZstdDecompressor()
    with d.stream_reader(io.BytesIO(data)) as r:
        return r.read()


def parse_game(raw, game_id='', min_riichi=0):
    """解析一场对局的 MJAI 事件流，产出决策样本列表。

    决策时刻 = 每次 tsumo 后紧接的 dahai（自己摸牌后打牌）。
    （吃碰后打牌也可纳入，但天凤 MJAI 的 tsumo/dahai 配对最干净，先只取门清摸切。）
    """
    events = [json.loads(l) for l in raw.decode('utf-8').splitlines() if l.strip()]
    samples = []
    hands = [[] for _ in range(4)]      # 各家手牌（牌名列表）
    rivers = [[] for _ in range(4)]     # 各家牌河
    riichi = [False] * 4
    dora_ind = []
    scores = [25000] * 4
    bakaze, self_wind, kyoku = 'E', 'E', 1
    pending_draw = None  # (actor, tile)

    def ctx(seat):
        """当前决策者视角的局面：把 seat 映射到 own，其余顺时针映射。"""
        order = [(seat + i) % 4 for i in range(4)]
        # 亲家 = (kyoku-1) % 4（1/2局东家亲，3/4南家亲...），自风按相对座位推算
        dealer = (kyoku - 1) % 4
        my_wind_idx = (dealer - seat) % 4  # 相对自己：0=东(亲) 1=南 2=西 3=北
        return {
            'hand': sorted(hands[seat]),
            'riichi': {SEATS[i]: bool(riichi[order[i]]) for i in range(4)},
            'rivers': {SEATS[i]: list(rivers[order[i]]) for i in range(4)},
            'dora_indicators': list(dora_ind),
            'scores': {SEATS[i]: scores[order[i]] for i in range(4)},
            'round_wind': bakaze, 'self_wind_idx': my_wind_idx,
            'dealer_rel': (dealer - seat) % 4,
            'kyoku': kyoku, 'seat_abs': seat,
        }

    for ev in events:
        t = ev.get('type')

        if t == 'start_kyoku':
            hands = [[norm_tile(x) for x in h] for h in ev['tehais']]
            rivers = [[] for _ in range(4)]
            riichi = [False] * 4
            dora_ind = [norm_tile(ev['dora_marker'])]
            scores = list(ev.get('scores', [25000] * 4))
            bakaze = ev.get('bakaze', 'E')
            kyoku = ev.get('kyoku', 1)
            oya = ev.get('oya', 0)
            self_wind = None
            pending_draw = None

        elif t == 'tsumo':
            who = ev['actor']
            tile = norm_tile(ev['pai'])
            if 0 <= who < 4 and tile in hands[who]:
                hands[who].remove(tile)
            if 0 <= who < 4:
                hands[who].append(tile)
            pending_draw = (who, tile)

        elif t == 'dahai':
            who = ev['actor']
            tile = norm_tile(ev['pai'])
            # 记录决策样本（摸牌者 = 弃牌者，即门清摸切）
            if pending_draw is not None and pending_draw[0] == who and len(hands[who]) == 14:
                n_riichi = sum(riichi)
                if n_riichi >= min_riichi:
                    st = ctx(who)
                    # 打出的牌若恰好在手里，则去掉后为 13 张合法手牌
                    st['discard'] = tile
                    st['game_id'] = game_id
                    st['tsumogiri'] = bool(ev.get('tsumogiri', False))
                    samples.append(st)
            if tile in hands[who]:
                hands[who].remove(tile)
            rivers[who].append(tile)
            pending_draw = None

        elif t == 'reach':
            who = ev['actor']
            if 0 <= who < 4:
                riichi[who] = True

        elif t == 'dora':
            dora_ind.append(norm_tile(ev.get('dora_marker', '')))

        elif t in ('pon', 'chi', 'daiminkan', 'ankan', 'kakan'):
            # 副露：从手牌移除消耗的牌，加一副露标记；本轮样本仍可用但手牌可能不闭合
            who = ev.get('actor', 0)
            consumed = ev.get('consumed') or []
            for c in consumed:
                c = norm_tile(c)
                if c in hands[who]:
                    hands[who].remove(c)
            pending_draw = None

        elif t == 'end_kyoku':
            pending_draw = None

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', nargs='?', help='单个 .zst 文件')
    ap.add_argument('--tar', help='从 tar 批量读取（可用 HTTP Range 下载的片段）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-riichi', type=int, default=0)
    ap.add_argument('--max-games', type=int, default=100000)
    args = ap.parse_args()

    out = open(args.out, 'w', encoding='utf-8')
    total_samples = total_games = 0

    def handle(data, gid):
        nonlocal total_samples, total_games
        try:
            raw = decompress(data)
            samples = parse_game(raw, gid, args.min_riichi)
        except Exception as e:
            return
        if not samples:
            return
        total_games += 1
        for s in samples:
            out.write(json.dumps(s, ensure_ascii=False) + '\n')
            total_samples += 1

    if args.tar:
        import tarfile
        tf = tarfile.open(args.tar, 'r|')  # 流式读取，容忍末尾截断
        while True:
            try:
                m = tf.next()
            except Exception:
                break  # HTTP Range 片段末尾截断
            if m is None:
                break
            if not m.isfile() or not m.name.endswith('.zst'):
                continue
            if total_games >= args.max_games:
                break
            gid = m.name.split('/')[-1].replace('.mjai.json.zst', '')
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue
            handle(data, gid)
    else:
        data = open(args.input, 'rb').read()
        gid = args.input.split('/')[-1].replace('.zst', '')
        handle(data, gid)

    out.close()
    print('games=%d samples=%d out=%s' % (total_games, total_samples, args.out))


if __name__ == '__main__':
    main()
