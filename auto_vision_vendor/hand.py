"""手牌切分 + 识别（正立整张牌）—— 与牌河统一的相对阈值版本。

 为什么重写
  Swift 的 TileSegmenter 用**绝对阈值**（gray>150 且 sat≤45）找牌面，
  被萬子下部的大块红「萬」字（sat 97~123）和牌顶绿沿（sat 52）打到
  colFrac ≈ 0.4，卡在 0.35 阈值边缘忽过忽不过 —— 实机 live_chi 上
  只切出 1 张暗牌、副露误判 19 张（真实 10 暗牌 + 1 摸牌 + 1 组副露）。
  牌河那套的 paper_mask 是**相对阈值**（g > 0.9×p97全图、sat<60），
  自适应每张截图的整体亮度 —— 换上之后同一张图一次切准。

## 切分思路（沿用 TileSegmenter 的「先分块、再数张」）

手牌张数**不是**固定 14 张：

| 情形 | 暗牌 | 摸牌 |
|---|---|---|
| 刚摸牌 | 13 | 1 |
| 刚打完 | 13 | 0 |
| 1 组副露后刚摸牌 | 10 | 1 |
| 1 组副露后没进张 | 10 | 0 |

手牌区永远**左对齐**：最左块 = 暗牌；紧跟的窄块（≤1.35 张宽、
空隙 0.25~0.8 张）= 摸牌；**其余一律是副露区**。

## 识别思路

手牌是**正立整张牌**（80:129），用 `Resources/tiles_clean/*.png`（80×129）
做模板：中位底色归一化的墨迹强度图 + 软 IoU（Σmin/Σmax），再用
「红色占比」裁决赤五（0X）与普通五（5X）—— 两者形状完全一样、
只有颜色不同，光比形状只差 0.01。
"""
import os
import sys

import cv2
import numpy as np


from . import split as RS
from . import reader as V

_READER_TPL = None        # reader 的图案模板（dict），懒加载缓存


def _reader_templates():
    global _READER_TPL
    if _READER_TPL is None:
        _READER_TPL, _ = V.load_templates()
    return _READER_TPL
from . import paths

TILE_W, TILE_H = 80, 129          # atlas sprite 尺寸（= tiles_clean png 尺寸）
MIN_SCORE = 0.45                  # 「认出来了」门槛（同 Swift minAcceptableScore）

# 摸牌判定：窄块（≤1.35 张宽）且与暗牌空隙 0.25~0.8 张
DRAWN_MAX_W = 1.35
DRAWN_GAP = (0.25, 0.80)


# ══════════════════════ 模板资产 ══════════════════════

class _Tpl:
    __slots__ = ('name', 'ink', 'red_ratio')

    def __init__(self, name, ink, red_ratio):
        self.name = name
        self.ink = ink
        self.red_ratio = red_ratio


_templates = None


def _ink_planes(bgr):
    """80×129 BGR → (墨迹强度图, 红色强度图)，底色按中位灰度归一。

    与 Swift TileTemplateMatcher.planes 同一口径：
      ink = clip((中位灰度 - gray)/90, 0, 1)
      red = clip((R - max(G,B))/60, 0, 1)
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gg = bgr[:, :, 1].astype(np.float32)
    bb = bgr[:, :, 0].astype(np.float32)
    median = float(np.median(g))
    ink = np.clip((median - g) / 90.0, 0.0, 1.0)
    red = np.clip((r - np.maximum(gg, bb)) / 60.0, 0.0, 1.0)
    return ink, red


def _mask_code(ink):
    """遮掉素材右上角的编号区（实拍那块是空的）。"""
    x0, y0, x1, y1 = 60, 17, 79, 36          # meta.json 的 code_box
    ink[y0:y1, x0:x1] = 0.0
    return ink


def load_templates():
    """37 张整张牌模板（tiles_clean/*.png，80×129）。进程内缓存。"""
    global _templates
    if _templates is not None:
        return _templates
    meta = json_load(os.path.join(paths.CLEAN, 'meta.json'))
    out = []
    for name in meta['tiles']:
        p = os.path.join(paths.CLEAN, name + '.png')
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        ink, red = _ink_planes(bgr)
        ink = _mask_code(ink)
        w = float(ink.sum())
        rr = float((ink * red).sum() / w) if w > 1 else 0.0
        out.append(_Tpl(name, ink, rr))
    _templates = out
    return out


def json_load(p):
    import json
    with open(p) as f:
        return json.load(f)


# ══════════════════════ 识别 ══════════════════════

def _soft_iou(a, b, dy=0, dx=0):
    """软 IoU = Σmin/Σmax，在 (dy,dx) 平移下。向量化。

    用恒等式 Σmin/Σmax = (S−D)/(S+D)，S=Σ(a+b)、D=Σ|a−b|。
    """
    H, W = a.shape
    ya0, ya1 = max(0, dy), min(H, H + dy)
    xa0, xa1 = max(0, dx), min(W, W + dx)
    if (ya1 - ya0) * (xa1 - xa0) < H * W * 0.4:
        return 0.0
    sa = a[ya0:ya1, xa0:xa1]
    sb = b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    s = float(sa.sum() + sb.sum())
    d = float(np.abs(sa - sb).sum())
    den = s + d
    return (s - d) / den if den > 1e-6 else 0.0


def recognize_tile(bgr_tile, templates=None, quality=None):
    """一张整张牌 BGR → (name, score)。赤五用红色占比裁决。"""
    tpl = templates or load_templates()
    if not tpl:
        return None, 0.0
    q = cv2.resize(bgr_tile, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    #   吃牌后本家不可出的牌会被渲染成「整张变灰降亮」（V 中位 ~120，
    #   正常牌 ~200）。灰牌的白面/墨迹对比度太低，直接提取图案会退化成
    #   白板（5z）。先做一次亮度拉伸把对比拉回来。
    if float(np.median(q)) < 150:
        q = cv2.normalize(q, None, 0, 255, cv2.NORM_MINMAX)
    ink, red = _ink_planes(q)
    ink = _mask_code(ink)
    q_red = float((ink * red).sum() / ink.sum()) if ink.sum() > 1 else 0.0

    # 图案紧裁后再匹配：模板的 pattern 是「紧裁到墨迹 bbox」，
    #   而这里裁的牌面在不同皮肤下留白比例不同（部分皮肤的白面几乎撑满
    #   裁框，模板里四周还有白边）——整牌缩放会让图案大小差 7~10%，
    #   IoU 被拉低，字形相近的牌就翻（四萬→3m）。先把图案紧裁再对模板，
    #   大小口径就一致了。牌河一直走这条路。
    pat, _pdbg = V.extract_pattern(q)
    #   extract_pattern 对白板返回 ('BLANK', None)——不是 ndarray，
    #   直接判断类型，别让整排手牌因一张白板抛异常被吞掉。
    if isinstance(pat, str):
        if pat == 'BLANK':
            return '5z', 0.98
        pat = None
    #   赤五这类整张红字的牌：「纸基准」会被红字带偏（ink 只剩 3.6%，
    #   top1 变成低分噪声）。墨迹太少时改用「红色掩膜」重取图案。
    if (pat is None or pat.mean() < 0.10) and q_red > 0.4:
        qf = q.astype(np.int16)          # 注意 q 是 BGR：红通道在 index 2
        r_ch = qf[:, :, 2]
        red_m = ((r_ch - np.maximum(qf[:, :, 1], qf[:, :, 0])) > 40
                 ).astype(np.float32)
        ys, xs = np.nonzero(red_m > 0)
        if len(ys) > 50:
            pat = red_m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if pat is not None and pat.size:
        sc = V.rank(pat, _reader_templates(), V.ALPHA)
        if sc:
            nm, sco = sc[0][1], sc[0][0]
            if len(nm) == 2 and nm[0] in ('5', '0'):
                suit = nm[1]
                _td = _reader_templates()
                r5 = next((tp.red_ratio for tp in _td.values()
                           if getattr(tp, 'name', '') == '5' + suit), None)
                r0 = next((tp.red_ratio for tp in _td.values()
                           if getattr(tp, 'name', '') == '0' + suit), None)
                if r5 is not None and r0 is not None:
                    nm = ('0' if q_red >= (r5 + r0) / 2 else '5') + suit
            if quality is not None:
                # Collapse ordinary/red-five variants before comparing rivals.
                # Color, rather than shape, chooses between those variants.
                base_name = nm.replace('0', '5')
                rivals = [item for item in sc if item[1].replace('0', '5') != base_name]
                quality['margin'] = round(float(sco - rivals[0][0]), 4) if rivals else None
                quality['alternatives'] = [item[1] for item in rivals[:3]]
                if quality['margin'] is not None and quality['margin'] < 0.02:
                    # A second, whole-tile comparison may confirm a close
                    # tight-pattern match. It never changes the chosen label.
                    # Both independent crop views must agree on the winner.
                    whole = {}
                    for tile_template in tpl:
                        key = tile_template.name.replace('0', '5')
                        value = max(_soft_iou(ink, tile_template.ink, dy, dx)
                                    for dy in range(-10, 11, 2)
                                    for dx in range(-4, 5, 2))
                        whole[key] = max(whole.get(key, 0.0), value)
                    own = whole.get(base_name, 0.0)
                    other = max((v for k, v in whole.items() if k != base_name), default=1.0)
                    quality['secondary_margin'] = round(own - other, 4) if own >= 0.50 else 0.0
            return nm, round(float(sco), 4)

    # 粗排：带系统性预偏移。
    #  paper 行段的底部常被「牌底↔白带」的过渡亮行拉偏，实拍字位比
    #   行带中心低 ~6px@129 —— 不带偏移时正确模板粗排分低、进不了
    #   决赛（8m 被平移凑分的 2m 翻掉，）。
    #   粗排取「无偏移 / -6 / -10」三档的最大值；决赛再全平移精搜。
    scored = sorted(((max(_soft_iou(ink, t.ink), _soft_iou(ink, t.ink, -6, 0),
                          _soft_iou(ink, t.ink, -10, 0)), t) for t in tpl),
                    key=lambda x: -x[0])
    # 前 6 名决赛：全平移搜索
    best, best_t = 0.0, None
    for s0, t in scored[:6]:
        for dy in range(-10, 11, 2):
            for dx in range(-4, 5, 2):
                v = _soft_iou(ink, t.ink, dy, dx)
                if v > best:
                    best, best_t = v, t
    if best_t is None:
        return None, 0.0

    name = best_t.name
    #  赤五裁决：0X 与 5X 形状一样，按「模板红占比中点」分边
    if len(name) == 2 and name[0] in ('5', '0'):
        suit = name[1]
        r5 = next((t.red_ratio for t in tpl if t.name == '5' + suit), None)
        r0 = next((t.red_ratio for t in tpl if t.name == '0' + suit), None)
        if r5 is not None and r0 is not None:
            name = ('0' if q_red >= (r5 + r0) / 2 else '5') + suit
    return name, round(best, 4)


# ══════════════════════ 切分 ══════════════════════

def _runs(on, min_len):
    out, s = [], -1
    for i, v in enumerate(on):
        if v:
            if s < 0:
                s = i
        else:
            if s >= 0:
                if i - s >= min_len:
                    out.append((s, i))
                s = -1
    if s >= 0 and len(on) - s >= min_len:
        out.append((s, len(on)))
    return out


def _merge(runs, tol):
    out = []
    for a, b in runs:
        if out and a - out[-1][1] <= tol:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def split(im, dbg=False, canvas_box=None):
    """手牌切分。im = 整图 BGR。

    返回 dict（坐标均为**整图像素**，与 overlay_cli 的牌河口径一致）：
      concealed : [(x0,y0,x1,y1), ...]   暗牌（不含摸牌）
      drawn     : (x0,y0,x1,y1) | None   摸牌；None = 这轮没有进张
      meld      : (x0,y0,x1,y1) | None   自家副露区（叠放/摆开，另计）
      band      : (x0,y0,x1,y1)          手牌行带
      pitch     : float                  牌宽
      canvas    : (x0,y0,x1,y1)
    """
    H, W = im.shape[:2]
    bx0, by0, bx1, by1 = canvas_box if canvas_box is not None else RS.canvas(im)
    cv_ = im[by0:by1, bx0:bx1]
    rgb = cv2.cvtColor(cv_, cv2.COLOR_BGR2RGB)
    #  灰牌通道：吃牌后本家不可出的牌会被
    #   渲染成**变灰降亮**（lum 仅正常的 56%，123 vs 219）——单用 rel=0.90 的
    #   paper_mask 在灰牌处整列归 0 → 手牌列剖面断块 → 第 5 张起全丢
    #   （只切出 4 张暗牌）。灰牌的 sat 极低（≈4），第二通道 rel=0.55 +
    #   sat_hi=25 能抓到，但**不能整张并入**（会把行两端的振聴标签等浅色
    #   元素也纳进来、块并成一条）——只在牌行内部的断口修补（见下方 cf 段）。
    paper = RS.paper_mask(rgb)
    gray_paper = RS.paper_mask(rgb, rel=0.55, sat_hi=25)
    h, w = paper.shape
    pitch = w * 126.0 / 2560.0          # canvas 宽度受限 → 牌宽只跟画布宽有关
    tile_h = pitch * TILE_H / TILE_W

    # ── 1. 行带：canvas 底部 25% 内 paper 占比 >0.12 的最长连续段 ──
    #    _runs 返回的是相对 search_top 的索引，必须加回偏移
    #     （坐标系混用：第一版没加，行带落到画面上部的牌河区）
    search_top = int(h * 0.75)
    rowfrac = paper[search_top:, :].mean(axis=1)
    segs = _runs(rowfrac > 0.12, min_len=20)
    if not segs:
        return _empty(bx0, by0, bx1, by1)
    ra, rb = max(segs, key=lambda t: t[1] - t[0])
    by_a, by_b = search_top + ra, search_top + rb
    #  inset 必须 ≤3%：行带缩进牌面中部后，列剖面粒度全碎
    #   （inset6 → 13 碎块，inset3/无 → 4 块恰好 = 暗牌/摸牌/副露）
    inset = max(2, int((by_b - by_a) * 0.03))
    y0, y1 = by_a + inset, by_b - inset
    if y1 - y0 < 15:
        return _empty(bx0, by0, bx1, by1)

    # ── 2. 列分块 → merge ──
    #  阈值必须低（0.10）：8p 圆圈、索子绿条纹会把牌面内部的列剖面打到
    #   0.14~0.33——阈值 0.35 时暗牌块在第 6 张就断掉（四家副露实机
    #   6m6m8m4p5p 后全丢）。牌缝（真空隙）是 0.00，与图案
    #   低值段（≥0.14）仍有分离度。merge 0.25×pitch 盖住图案低段、
    #   不吞摸牌空隙（0.3~0.5×pitch）。
    cf_n = paper[y0:y1, :].mean(axis=0)
    cf_g = gray_paper[y0:y1, :].mean(axis=0)
    # 灰牌断口修补：只填「牌行内部」的洞（正常块之间、洞宽 ≤1.2 牌宽），
    # 行两端外侧不动——否则振聴标签等浅色元素会被并进手牌块。
    colfrac = cf_n.copy()
    runs_n = _runs(cf_n > 0.10, min_len=6)
    for (a1, b1), (a2, b2) in zip(runs_n, runs_n[1:]):
        if 0 < a2 - b1 <= pitch * 1.2:
            colfrac[b1:a2] = np.maximum(cf_n[b1:a2], cf_g[b1:a2])
    # ── 3. 逐段即逐张──
    #   旧做法：_merge 把相邻牌之间 4px 的缝也合并成「一整块」，再按
    #   「块宽 ÷ 公式牌宽」等分。深色桌布 / 浅色牌底的组合下公式牌宽偏大 4%，
    #   8 张被算成 7 张，框沿行累积偏移（四萬→3m、六萬→7m）。
    #   列剖面本来就能把每张牌切成独立一段（9~12 段，
    #   每段宽误差 ±1px），直接拿段当单张，天然容忍牌的小抖动/转动。
    runs = _runs(colfrac > 0.10, min_len=6)
    if not runs:
        return _empty(bx0, by0, bx1, by1)
    widths = [b - a for a, b in runs]
    med = float(np.median(widths))
    #   图案断口兜底：明显窄于一张牌的段（8p 圆圈间、索子条纹）并入左邻
    merged = []
    for r in runs:
        if merged and (r[1] - r[0]) < 0.6 * med:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    #   宽 > 1.5 张 = 副露区（叠放/摆开的整组），其余是单张
    singles = [(a, b) for a, b in merged if (b - a) <= 1.5 * med]
    meld_runs = [(a, b) for a, b in merged if (b - a) > 1.5 * med]
    if not singles:
        return _empty(bx0, by0, bx1, by1)
    hand = (singles[0][0], (meld_runs[-1][1] if meld_runs else singles[-1][1]))
    pitch_hand = med

    #   摸牌：最后一个单张与前面的间隙 > 0.3 张
    drawn = None
    if len(singles) >= 2 and singles[-1][0] - singles[-2][1] > pitch * 0.3:
        drawn = singles[-1]
        singles = singles[:-1]

    #   副露：最右那个超宽段
    meld = meld_runs[-1] if meld_runs else None

    # ── 4. 框（每张用自己那段的白面边界，不做等分）──
    cy = (y0 + y1) / 2.0
    ty0, ty1 = cy - tile_h / 2.0, cy + tile_h / 2.0
    concealed = [(int(a + bx0), int(ty0 + by0),
                  int(b + bx0), int(ty1 + by0)) for a, b in singles]
    drawn_rect = None
    if drawn:
        a, b = drawn
        drawn_rect = (int(a + bx0), int(ty0 + by0),
                      int(b + bx0), int(ty1 + by0))
    meld_rect = None
    if meld:
        meld_rect = (int(meld[0] + bx0), int(y0 + by0),
                     int(meld[1] + bx0), int(y1 + by0))

    out = dict(concealed=concealed, drawn=drawn_rect, meld=meld_rect,
               band=(int(hand[0] + bx0), int(y0 + by0),
                     int(hand[1] + bx0), int(y1 + by0)),
               pitch=round(pitch_hand, 1), canvas=(bx0, by0, bx1, by1))
    if dbg:
        out['_dbg'] = dict(segs=segs, blocks=blocks, pitch_est=pitch)
    return out


def _empty(bx0, by0, bx1, by1):
    return dict(concealed=[], drawn=None, meld=None, band=None,
                pitch=0.0, canvas=(bx0, by0, bx1, by1))


def read_hand(im, templates=None, canvas_box=None):
    """切分 + 识别一步到位。返回 overlay_cli 需要的 hand 结构。"""
    sp = split(im, canvas_box=canvas_box)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    tiles = []

    def _crop(rect):
        x0, y0, x1, y1 = rect
        return im[max(0, y0):y1, max(0, x0):x1]

    entries = [(i, r, False) for i, r in enumerate(sp['concealed'])]
    if sp['drawn']:
        entries.append((len(sp['concealed']), sp['drawn'], True))
    for i, r, is_drawn in entries:
        quality = {}
        try:
            name, score = recognize_tile(_crop(r), templates, quality=quality)
        except Exception:
            name, score = None, 0.0      # 单张异常不连累整排
        tiles.append(dict(id=i, quad=[int(v) for v in r], name=name,
                          score=score, drawn=is_drawn, **quality))
    n_meld = 0
    if sp['meld']:
        x0, _, x1, _ = sp['meld']
        n_meld = max(0, int(round((x1 - x0) / (sp['pitch'] * 0.84))))
    if n_meld == 0:
        sp['meld'] = None          # 宽度不足 1 张的「假副露」（计分数字等）兜底
    return dict(tiles=tiles, meld_rect=sp['meld'], meld_tiles=n_meld,
                band=sp['band'], pitch=sp['pitch'], canvas=sp['canvas'])


# ══════════════════════ 宝牌指示牌 ══════════════════════

# 指示牌 → 宝牌（下一张）的循环顺序：
#   数牌 1→2→…→9→1；风牌 東1→南2→西3→北4→東1；三元 白5→發6→中7→白5
#   （ 三元是「白發中」循环，与国标「中發白」顺序不同）
_NEXT_Z = {'1': '2', '2': '3', '3': '4', '4': '1',
           '5': '6', '6': '7', '7': '5'}


def dora_of(indicator):
    """指示牌 → 宝牌（下一张）。赤五（0X）不会是指示牌，故不处理。"""
    if not indicator or len(indicator) != 2:
        return None
    if indicator[1] == 'z':
        return _NEXT_Z.get(indicator[0], '?') + 'z'
    n = int(indicator[0])
    return str(n % 9 + 1) + indicator[1]


def read_dora(im, tile_width=0.0):
    """读画面左上角的宝牌指示牌。

    面板 5 个槽位：宝牌占最左几格，其余是**空槽（亮绿）**——
    空槽的绿本身就能定位整个阵列，窗口大小/缩放怎么变都不用改常数。

    返回 dict：
      indicators : [name, ...]      读到的**宝牌指示牌**（从左到右）
      dora       : [name, ...]      换算后的**宝牌**（指示牌下一张）
      dora_rects : [{quad, name, score}]  每个指示牌的槽位框（画图层用）
      green_slots: [[x,y,w,h], ...] 空槽（调试）
      note       : str | None
    """
    H, W = im.shape[:2]
    expect = tile_width if tile_width > 4 else W * 0.049
    # 面板位置：左上角，不同窗口下大致占 x 0~35%、y 6~32%
    x0, x1 = 0, int(W * 0.35)
    y0, y1 = int(H * 0.06), int(H * 0.32)
    note = None

    # 空槽颜色随皮肤变（亮绿、橙黄等），按「高饱和 + 够亮」判定。
    #   两种都收（橙：R 高且明显高于 G/B；绿：G 高且明显高于 R/B），
    #   牌山/牌背的橙在 ROI 外，指示牌是白底也不满足，误检风险低。
    roi = im[y0:y1, x0:x1]
    b, g, r = cv2.split(roi.astype(np.int16))
    orange = (r > 150) & (r - g > 40) & (r - b > 60) & (g > 60)
    green = (g > 150) & (g - r > 35) & (g - b > 30) & (r > 60)
    mask = (orange | green).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    slots = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        ar = h / max(w, 1)
        if a < expect * expect * 0.15:
            continue
        if not (expect * 0.30 <= w <= expect * 0.85 and 1.0 <= ar <= 1.9):
            continue                      # 牌墙的绿背宽得多，滤掉
        #  cv2 的 stats 是 np.int32 —— 不转 Python int 会在 json.dumps 崩
        slots.append((int(x + x0), int(y + y0), int(w), int(h)))
    # 真面板有 4~5 个槽（1 张指示牌 + 4 个空槽）。只有 1 个「槽」
    # 说明是别处的浅色块（牌河截图左上角的白板）——不算面板。
    if len(slots) < 2:
        return dict(indicators=[], dora=[], dora_rects=[], green_slots=[],
                    note='宝牌面板：未找到（槽位不足）')
    slots.sort(key=lambda s: s[0])

    slot_w = float(np.median([s[2] for s in slots]))
    slot_h = float(np.median([s[3] for s in slots]))
    if len(slots) >= 2:
        pitch_d = float(np.median([slots[i][0] - slots[i - 1][0]
                                   for i in range(1, len(slots))]))
    else:
        pitch_d = slot_w * 1.08
    leftmost = slots[0][0]
    top_y = float(np.median([s[1] for s in slots]))
    n_dora = 5 - len(slots)

    indicators, dora_rects = [], []
    for k in range(1, n_dora + 1):
        left = leftmost - (n_dora + 1 - k) * pitch_d
        rect = (int(left), int(top_y), int(left + slot_w), int(top_y + slot_h))
        sub = im[max(0, rect[1]):rect[3], max(0, rect[0]):rect[2]]
        name, score = recognize_tile(sub)
        if name and score >= MIN_SCORE:
            indicators.append(name)
        elif note is None:
            note = f'宝牌第 {k} 张认不准（{score:.2f}）'
        dora_rects.append(dict(quad=list(rect), name=name,
                               score=round(score, 4)))
    return dict(indicators=indicators,
                dora=[dora_of(n) for n in indicators],
                dora_rects=dora_rects,
                green_slots=[list(s) for s in slots], note=note)


if __name__ == '__main__':
    # 快速自检：python hand_split.py <图>
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        paths.FIXTURES, 'live_chi_from_kamicha.jpg')
    im = cv2.imread(p)
    r = read_hand(im)
    names = [(t['name'] or 'None') for t in r['tiles']]
    print(f"{os.path.basename(p)}: 暗牌 {sum(1 for t in r['tiles'] if not t['drawn'])} 张"
          f" + 摸牌 {sum(1 for t in r['tiles'] if t['drawn'])} 张"
          f" | 副露 {r['meld_tiles']} 张 @ {r['meld_rect']}")
    print('  ' + ' '.join(f"{t['name'] or '?'}({t['score']:.2f})" for t in r['tiles']))
    dr = read_dora(im, r['pitch'])
    if dr['indicators']:
        print(f"  宝牌指示牌: {' '.join(dr['indicators'])}"
              f"  → 宝牌: {' '.join(dr['dora'])}"
              + (f"  ⚠️ {dr['note']}" if dr.get('note') else ''))
    elif dr.get('note'):
        print(f"  宝牌指示牌: — （{dr['note']}）")
