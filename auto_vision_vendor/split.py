""" 标准网格切分器 —— 通用、无标注、跨分辨率 

═══════════════════════════════════════════════════════════════════════
【为什么放弃「单牌连通域」】
  在 ground truth（4032×2058、牌 95px、PNG）上69/69 全中，
  在浏览器窗口截图（牌 45px、JPEG）上**失效** —— 实测 67 个块里
  对家只切出 4 张（实际 18 张），上/下家各 12（实际 17）。
  根因：牌与牌之间那道暗边（牌↔桌面交界）在低分辨率 + JPEG 压缩后
  **不够暗**，掩膜跨牌连通。调阈值救不了（调高会削掉牌面本身）。

═══════════════════════════════════════════════════════════════════════
【本模块：网格定「线」+ 零区间（牌缝）定「张」】
  1. **网格参数全部来自 GT 标注的规律**，不是人工顶点本身。
      踩过的坑：曾用自家那把牌宽标尺（tw=95）去算 across 的张距，
       得出 102 而真实是 85 → across 第2、3 行整体偏移 +16 / +30px。
       **每家必须用自己的 tw_ratio。**
  2. **跨分辨率**：GT 画面 3808×1834、实战 1815×874，
     画面比例都是 **2.076**（同一个 Unity 虚拟画布）→ 归一化直接换算。
  3. **张的位置靠「零区间」而不是「极大值平台」** ——
     牌河里牌紧贴，牌缝的掩膜值是 0（实测 own 7 段、across 6~7 段、
     侧家 7 段，非常稳定 = 牌数+1）；
     而**极大值平台会被牌面图案切碎**（实测 own 段宽 32~43 而牌宽 95，
     across 段宽 4~28 而 pitch 85）→ 用平台中心定相位必然失败。
  4. **零段间距比 pitch 小6~11%**（实测 own 92~98 vs 103、
     across 77~83 vs 85、侧家 52~64 vs 64）→ 用**实测间距中位**
     反推该线真实的 pitch，再据此推算每张的中心 —— 不套用名义值。
  5. **立直牌横躺不改变行高**（用户给出的物理事实）：行高恒为
     该家标准牌高 tw×tw_ar，横躺牌只是所在线的某一段变宽。
  6. **每家一个固定窗口**（GT 实测，彼此不重叠）→ 天然排除副露与手牌
     （own|meld 在 y≈0.92、right|meld 在 x≈0.75 y≈0.10~0.25）。
  7. **张数不预设**：零段数−1 就是这条线上的牌数，
     最多 n_slot 张（21 张 ÷ 6 = 3.5 → 4 条线）。

═══════════════════════════════════════════════════════════════════════
【GT 实测的每家网格（tools/river_grid.py 从 76 张标注反推）】
  座位     牌宽×牌高  tw/画面宽  张距   行距   布局
  own       95 × 89    0.024951   103.0  101.5   行沿x 堆 6 张 × 3 行
  across79 × 64    0.020746    85.0   71.5   行沿 x 堆 6 张 × 3 行
  left     118 × 55    0.031126    63.9  123.4   列沿 y 堆 6 张 × 3 列
  right    117 × 58    0.030870    64.0  123.0   列沿 y 堆 6 张 × 3 列
   侧家是「横躺的牌竖着堆」：张距沿 y，列距沿 x。
═══════════════════════════════════════════════════════════════════════
"""

import json

import cv2
import numpy as np

from . import paths


# ═════════════════ 1. 标准网格（GT 每家用自己的牌宽为单位） ═════════════════
SEATS = {
#  全量换算到「内容锚定 canvas」坐标系 
#   旧常数以旧 canvas 框（GT=(112,76,3920,1910)，含浏览器工具栏）归一化，
#   工具栏偏移被常数吸收 —— 换窗口尺寸即崩（实机 1920×1350 四家全错位）。
#   canvas() 重写为内容锚定后，GT 新框=(485,177,3547,1910)（真实 16:9 游戏区）。
#   换算（仿射，物理位置不变）：x: (v·3808+112−485)/3062  y: (v·1834+76−177)/1733
#   仅 win / anchor / anchor_xs / tw_ratio / skew_ref 与框挂钩；
#   quad_*、pitch、step、row_*_scale、tw_ar 均为「牌宽=1」基准，不变。
    'own': dict(
        win=(0.38957, 0.48324, 0.60198, 0.69744),
        tw_ratio=0.031025, tw_ar=0.9368,
        anchor=(0.56888, 0.52429),        #  沿 across 的线中心方向（y）
        pitch=1.0842, step=1.0936, step_growth=0.0025,
        n_slot=6, stack='x',
        #  逐行 y 修正
        #   旧 band cy±0.55th 系统性偏上：行0 顶高 7.5px、行1 顶高 5.5px、
        #   行2 顶高 5px + 底高 3px（@1920，GT 2× 尺度 ÷2 后完全一致
        #   —— 尺度无关的模型偏差）。(顶,底) 归一化修正量（×th）：
        #   正值 = 向下移。行3 无沿用行2。
        row_y_adj=[(0.135, 0.022), (0.099, 0.022), (0.094, 0.058), (0.094, 0.058)],
    ),
    'across': dict(
        win=(0.41120, 0.13009, 0.58817, 0.27963),
        tw_ratio=0.025800, tw_ar=0.8101,
        anchor=(0.13779, 0.15961),
        pitch=1.0759, step=0.9018, step_growth=0.0025,
        n_slot=6, stack='x',
        #  同 own：GT 实测顶偏 −11/−7/−7 (2×) → ÷2 后同样偏上
        row_y_adj=[(0.137, 0.0), (0.0875, 0.0125), (0.0875, 0.031), (0.0875, 0.031)],
    ),
    'left': dict(
        #  侧家的牌面是**平行四边形**（不是矩形）
        #   「从本家看过去，上家就是右斜的，下家是左斜的」
        #   GT 标注实测（36 张，tools/calib_grid.py）：
        #     left  上边单位向量 (+1.0000, −0.0092)
        #           右边单位向量 (−0.0433, +0.5018)← 底边往左偏
        #     right 上边单位向量 (+0.9999, −0.0046)
        #           右边单位向量 (+0.0925, +0.5177)← 底边往右偏
        #   上边基本水平、右边倾斜 ≈ 7~10°。
        #    之前输出**轴对齐矩形** → 只框住牌的左半部分（框内白纸
        #     占比掉到 0.00），识别率必然低。
        quad_top=(1.0000, -0.0092), quad_right=(-0.0433, 0.5018),
        #  每列的倾斜：列中心 x随 y 的斜率（px/px）
        #   GT 实测（每列不同，越靠外斜率越大 —— 离屏幕中轴越远透视越强）：
        #     列0 −0.1757  列1 −0.1582  列2 −0.0943
        #   318px 的 y 跨度上累计 −56~−26px —— 不修正的话整列会斜出牌。
        #  每列独立的中心 x（GT 实测该列 6 张牌中心的均值）
        #   原先用「统一 anchor + step 累加」，实测三列统一偏左 95~101px。
        #    坐标系：cut_seat 里 W/H 是**画面内**尺寸（GT 3808×1834，
        #      canvas box=(112,76)），而标注是**整图**坐标（4032×2058）
        #      → 这里必须存「(整图x − bx0) / W」。
        #      曾直接存整图归一化 → 整列右移 112px。
        anchor_xs=[0.29860, 0.33896, 0.37933],
        skews=[-0.1500, -0.1513, -0.0952],
        skew_ref=0.36872,        #  归一化（相对画面高）—— 用绝对像素会在
        #                          小窗口图上失效（实战 875 高 vs GT 1834）
        #  边长缩放（tools/sweep_quad.py 扫出来的联合最优）
        #   left: ws=0.92 hs=0.98 skew_ref=740 → IoU **0.872**
        #   right: ws=0.96 hs=0.95 skew_ref=760 → IoU **0.891**
        #原因：tw_ratio 来自**白纸掩膜的轴对齐宽度**（118.5），
        #   而标注顶边长只有 108.5 —— 掩膜把斜边多吃的部分也算进去了。
        quad_w_scale=0.92, quad_h_scale=0.98,
        #  逐行尺寸缩放（相对 quad_w_scale/quad_h_scale 的额外系数）
        #   实测真值（GT 与实战一致）：
        #     宽/tw  r0 0.919  r1 0.957  r2 0.988  r3 1.014  r4 1.010  r5 0.945
        #     高/tw  r0 0.436  r1 0.453  r2 0.475  r3 0.492  r4 0.506  r5 0.527
        #    高递增 21%、宽递增 3% —— **高度变化远大于宽度**
        #     而旧实现两者都固定 → 近处的牌被从上、下两侧切掉
        #    19:01 修正：上一版按「所有列平均」直接取值，
        #     对 c2 偏大 → 前 4 张底部压到牌下的黑色区域。
        #     tools/calib_row2.py 在 **GT 与实战两张图上**扫出的综合最优
        #     k=0.96（GT 0.98 / 实战 0.96，两边一致）。
        row_w_scale=[0.928, 0.964, 0.986, 1.036, 1.072, 1.075],
        row_h_scale=[0.948, 0.984, 1.008, 1.059, 1.076, 1.128],
        win=(0.26122, 0.24862, 0.44528, 0.54282),
        tw_ratio=0.038701, tw_ar=0.4641,
        anchor=(0.29826, 0.27740),        #  (band 中心x, 堆叠起点y) —— band 对称取值，勿改
        pitch=0.5388, step=1.0422, step_growth=0.0,
        n_slot=6, n_line=3, stack='y',
    ),
    'right': dict(
        quad_top=(0.9999, -0.0046), quad_right=(0.0925, 0.5177),
        # 符号与左家相反：列0 +0.1057  列1 +0.1321  列2 +0.1549
        anchor_xs=[0.61620, 0.65712, 0.69664],
        skews=[0.1038, 0.1370, 0.1600],
        skew_ref=0.38027,
        #  边长缩放（tools/sweep_quad.py 扫出来的联合最优）
        #   left: ws=0.92 hs=0.98 skew_ref=740 → IoU **0.872**
        #   right: ws=0.96 hs=0.95 skew_ref=760 → IoU **0.891**
        #原因：tw_ratio 来自**白纸掩膜的轴对齐宽度**（118.5），
        #   而标注顶边长只有 108.5 —— 掩膜把斜边多吃的部分也算进去了。
        quad_w_scale=0.92, quad_h_scale=0.98,
        #  逐行尺寸缩放（真值，GT 与实战一致）
        #   宽/tw  r0 0.983  r1 0.996  r2 0.996  r3 0.997  r4 0.993
        #   高/tw  r0 0.460  r1 0.479  r2 0.491  r3 0.508  r4 0.528
        #   高递增 15%，宽几乎不变 → 仍以高度缩放为主
        #    calib_row2.py 综合最优 k=1.00（GT 1.01 / 实战 1.00）→ 不缩放
        row_w_scale=[0.983, 0.996, 0.996, 0.997, 0.993, 0.993],
        row_h_scale=[0.979, 1.017, 1.023, 1.071, 1.107, 1.107],
        win=(0.58581, 0.24862, 0.77111, 0.54282),
        tw_ratio=0.038373, tw_ar=0.4894,
        anchor=(0.61681, 0.55388),
        pitch=0.5447, step=1.0465, step_growth=0.0,
        n_slot=6, n_line=3, stack='y',
    ),
}
MAX_LINE = 4          # 21 张 ÷ 每行 6 张 = 3.5 → 4 条线
ZERO_REL = 0.02# 牌缝判据：剖面值 < 2% × max
FILL_THR = 0.45       # 探针阈值：中心区白纸覆盖率
TILE_MIN = 0.25       # span 纸面强度下限：峰值 / band 横向尺寸（见 cut_seat）


# ═════════════════════════════ 2. 基础工具 ═════════════════════════════
#  canvas 必须是「物理正确的游戏画面」，不是「亮度 > 阈值的范围」：
#   浏览器工具栏、页面留白这些亮行一旦被算进 canvas，归一化坐标系整体偏移，
#   四家位置就全错。旧常量还可能恰好「吸收」某张标定图的偏移，换个窗口就崩。
#  三个锚点都取自画面内容，与窗口尺寸无关：
#    底 y1 = 最后一条亮行（游戏底缘，下方是暗壁纸）
#    顶 y0 = 牌山顶 − 边距比 × 画布高（牌山是牌背朝上的叠牌，强特征；
#            边距比 17/1080 按 16:9 画布标定）
#    左右 x = 游戏区中部行带里「最长连续亮列段」（避开工具栏/黑边与孤点）
#  交叉验证：多数界面画布为 16:9。若 (y1−y0) 明显大于 x 跨度 × 9/16
#   （底部有白带等异物），按 16:9 反推 y1。
GREEN_H_LO, GREEN_H_HI = 35, 85      # OpenCV H（绿色 ≈ 35~85）
GREEN_S_MIN, GREEN_V_MIN = 80, 60
GREEN_ROW_FRAC = 0.05                # 一行里绿像素 >5% 宽才算墙
GREEN_RUN = 3                        # 连续 3 行（防孤点）
MARGIN_R = 17.0 / 1080.0             # canvas顶→绿墙顶 距离 / 画布高（实机标定）
COL_THR = 25                         # 列亮度阈值（壁纸 ~14-16，桌面 ~35-45）
COL_GAP = 30                         # 列段断口

def _warm_bar_viewport(im, hsv):
    """用画面里的「暖色长条」定位游戏视口。

    雀魂画面左右（有时上下）有成片的暖色区域——那是**牌山**（一摞牌背
    朝上的牌），牌背配色随皮肤变（默认绿，换肤后多为橙黄），不是装饰边框。
    这些长条随画面同比例缩放，可以当视口的粗略锚点；窗口被拉宽/拉高时，
    画面内部比例不变，只是外面留出不同留白。
    三张不同窗口比例的实测：视口宽高比恒为 1.885（1782×944 / 1661×881 /
    1767×938），而整图比例是 1.56 / 1.72 / 1.41——拿整图当 canvas 会让
    归一化坐标全错。边框是暖色长条（上下横条 + 左右竖条），牌背颜色虽
    色相相近，但只占行内一小块，「行/列覆盖率 15%」就能分开。
    """
    # 暖色长条（牌背的橙/棕）——牌背配色随皮肤变，这里只覆盖暖色系；
    # 冷色牌背（绿/蓝）走不到这条路径，由手牌锚定兜底。
    warm = ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) &
            (hsv[:, :, 1] > 90) &
            (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 210))
    ry = np.nonzero(warm.mean(axis=1) > 0.08)[0]
    rx = np.nonzero(warm.mean(axis=0) > 0.08)[0]
    if len(ry) < 2 or len(rx) < 2:
        return None
    #   手牌的牌底边会横跨整行（覆盖率 45%），比两侧牌山（~10%）还显眼——
    #   不排除会把手牌整排切出视口外。
    #   判据：该行下方紧接着是白牌（V 高）→ 是牌底边，不是牌山。
    vrow = hsv[:, :, 2].mean(axis=1)
    valid = []
    for y in ry:
        below = vrow[int(y) + 8:int(y) + 40]
        if below.size and float(below.mean()) > 170:
            continue
        valid.append(int(y))
    if len(valid) < 2:
        return None
    y0, y1 = valid[0], valid[-1] + 1
    x0, x1 = int(rx[0]), int(rx[-1]) + 1
    H, W = im.shape[:2]
    if (x1 - x0) < 0.45 * W or (y1 - y0) < 0.45 * H:
        return None
    if not (1.4 < (x1 - x0) / max(1, y1 - y0) < 2.4):
        return None
    return x0, y0, x1, y1


def _hand_anchor_viewport(im):
    """用手牌行反推游戏画面区域（不依赖桌布颜色/牌山样式）。

    手牌是画面里最强的几何基准：一排白面牌，牌距（pitch）与画面尺寸的
    比例在不同皮肤/窗口比例下都稳定（实测两套界面 0.0492 / 0.0493），
    手牌左沿在 11.7% 画布宽、手牌行底在画布底上方约 1%。
    """
    h, w = im.shape[:2]
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    paper = paper_mask(rgb)
    def _segs(sig, thr, minlen):
        on = sig > thr
        out, s = [], -1
        for i, v in enumerate(on):
            if v and s < 0:
                s = i
            elif not v and s >= 0:
                if i - s >= minlen:
                    out.append((s, i))
                s = -1
        if s >= 0 and len(on) - s >= minlen:
            out.append((s, len(on)))
        return out

    top = int(h * 0.45)
    rowfrac = paper[top:, :].mean(axis=1)
    segs = _segs(rowfrac, 0.12, 20)
    if not segs:
        return None
    ra, rb = max(segs, key=lambda t: t[1] - t[0])
    y_a, y_b = top + ra, top + rb

    band = paper[y_a:y_b, :].mean(axis=0)
    runs = _segs(band, 0.10, 6)
    if len(runs) < 3:
        return None
    starts = [a for a, b in runs]
    widths = [b - a for a, b in runs]
    if len(starts) >= 2:
        pitch = float(starts[-1] - starts[0]) / (len(starts) - 1)
    else:
        return None
    if pitch < 25 or pitch > w * 0.12:
        return None
    # 首段若是「窄条」（UI 标签）跳过
    if widths[0] < 0.6 * pitch and len(starts) > 1:
        starts = starts[1:]
        y_a0 = y_a
    else:
        y_a0 = y_a

    W = pitch / 0.0492
    H = pitch / 0.0870
    x0 = starts[0] - 0.117 * W
    y1 = y_b + 0.01 * H
    y0 = y1 - H
    x0, x1 = int(round(x0)), int(round(x0 + W))
    y0, y1 = int(round(y0)), int(round(y1))
    if W < w * 0.35 or H < h * 0.35:
        return None
    # 必须裁在图内：负 x0 在 numpy 里会变成「从右侧数」，裁出完全错误的区域
    x0 = max(0, min(x0, w - 10))
    x1 = min(w, max(x1, x0 + 10))
    y0 = max(0, min(y0, h - 10))
    y1 = min(h, max(y1, y0 + 10))
    return (x0, y0, x1, y1)


def canvas(im, thr=20):
    """找游戏画面区域（内容锚定，窗口尺寸无关）。"""
    key = ('canvas', _fingerprint(im))
    hit = _FRAME_CACHE.get(key)
    if hit is not None:
        return hit
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = im.shape[:2]
    rm = g.mean(axis=1)
    br = np.nonzero(rm > thr)[0]
    if not len(br):
        _FRAME_CACHE[key] = (0, 0, w, h)
        return 0, 0, w, h
    y1 = int(br[-1]) + 1

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    bars = _warm_bar_viewport(im, hsv)
    if bars is not None:
        # 视口底取「牌山底」与「内容底（最后亮行）」的较大者：
        #   手牌那排本身不是牌山（只有牌底边，且被 above-白牌判据排除），
        #   只认牌山底会把手牌整排截在视口外。
        wx0, wy0, wx1, wy1 = bars
        # 底边优先用手牌锚定的结果（手牌行底 + 1% 画布高）：
        #   手牌下方往往是页面留白/计分大字，按「最后亮行」会多算 100+px。
        hv = _hand_anchor_viewport(im)
        if hv is not None:
            hy1 = hv[3]
            if wy1 <= hy1 <= h:
                wy1 = hy1
        out = (wx0, wy0, wx1, min(wy1, h))
        _FRAME_CACHE[key] = out
        return out
    green = ((hsv[:, :, 0] >= GREEN_H_LO) & (hsv[:, :, 0] <= GREEN_H_HI) &
             (hsv[:, :, 1] > GREEN_S_MIN) & (hsv[:, :, 2] > GREEN_V_MIN)
             ).sum(axis=1)
    gok = green > w * GREEN_ROW_FRAC
    gtop = None
    run = 0
    for i in range(h):
        run = run + 1 if gok[i] else 0
        if run >= GREEN_RUN:
            gtop = i - GREEN_RUN + 1
            break

    def y0_of(y1_):
        if gtop is None:
            # 无绿墙（异常/异主题）：退回 16:9 推断
            return max(0, y1_ - int(round(w * 9 / 16)))
        r = MARGIN_R
        return int(round((gtop - r * y1_) / (1 - r)))

    y0 = y0_of(y1)

    # x 边：游戏区中部行带（避开上沿壁纸过渡带与底部白带/黑带）
    H = y1 - y0
    yb0, yb1 = y0 + int(0.15 * H), y1 - int(0.08 * H)
    cm = g[max(0, yb0):max(1, yb1), :].mean(axis=0)
    idx = np.nonzero(cm > COL_THR)[0]
    if not len(idx):
        x0, x1 = 0, w
    else:
        segs, s = [], idx[0]
        for k in range(1, len(idx)):
            if idx[k] - idx[k - 1] > COL_GAP:
                segs.append((s, idx[k - 1]))
                s = idx[k]
        segs.append((s, idx[-1]))
        x0, x1 = max(segs, key=lambda t: t[1] - t[0])
        x0, x1 = int(x0), int(x1) + 1

    # 16:9 交叉验证：底缘下方若有异物（白带），y1 用 x跨度×9/16 推
    y169 = y0 + int(round((x1 - x0) * 9 / 16))
    if y1 - y0 > int(1.03 * (y169 - y0)):
        y1 = min(y169, h)
        y0 = y0_of(y1)
    return x0, y0, x1, y1


# ── 单帧结果缓存 ──────────────────────────────────────────────
#   canvas / paper_mask 都是全图量级的计算，read() 一次会经过
#   split（牌河）与 hand.split（手牌）两条路，各自都要求 canvas 与纸掩膜。
#   缓存按「id + shape + 稀疏采样指纹」命中，避免同一张图重复计算。
_FRAME_CACHE = {}


def _fingerprint(im):
    # 只按内容取指纹：cv2.cvtColor 每次都会产生新对象，带 id 会全部 miss
    h, w = im.shape[:2]
    step = max(1, min(h, w) // 32)
    return (h, w, int(im[::step, ::step].sum()))


def paper_mask(rgb, rel=0.80, sat_hi=60):
    """白纸掩膜：亮度够高且饱和度不高（排掉绿桌布、橙牌沿）。"""
    key = ('paper', _fingerprint(rgb), round(rel, 3), sat_hi)
    hit = _FRAME_CACHE.get(key)
    if hit is not None:
        return hit
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    thr = rel*float(np.percentile(g, 97))
    out = ((g > thr) & (sat < sat_hi)).astype(np.uint8)
    _FRAME_CACHE[key] = out
    return out


def _runs(v, thr):
    """一维信号里连续超过 thr 的区间。"""
    on = v > thr
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def probe(mask, cx, cy, hw, hh):
    """格子探针：中心矩形内的白纸覆盖率。"""
    x0 = max(0, int(round(cx-hw))); x1 = min(mask.shape[1], int(round(cx+hw)))
    y0 = max(0, int(round(cy-hh))); y1 = min(mask.shape[0], int(round(cy+hh)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


# ═════════════════ 3. 一条线：零区间 → 张位═════════════════════════
def cut_line(prof, lo, hi, n_max, expect_pitch=0.0):
    """在一条线的剖面里找牌位。

    prof          该线的白纸剖面（沿堆叠方向）
    lo, hi        有效范围（相对 prof 的下标）
    n_max         最多几张
    expect_pitch  名义张距（网格给的）—— 用来**补齐漏掉的牌缝**

    返回 (spans, pitch)：每张牌的 (起, 止)（相对 prof）与实测张距。

     为什么不能只靠「极大值平台」 
      牌面上的图案（万字的横、筒子的圆）会把剖面切碎成 24~46px 的
      碎段（GT 自家：牌宽 95 而段宽 32~43；across：pitch 85 而段宽 4~28）
      → 用平台中心定相位必然失败。

     为什么零区间也会漏 
      实战截图里对家第 2 行首张与上一行的牌**紧贴**，掩膜在该处不断开 →
      零区间数 = 牌数−1（实测 6 段 → 5 张，实际 6 张）。
      修法：**用网格的名义 pitch 检查每段宽度**，
      凡是宽于 ~1.5×pitch 的段就说明里面挤了 2 张，按 pitch 拆开。
    """
    seg = prof[lo:hi]
    if seg.size < 20 or seg.max() < 10:
        return [], 0.0
    zr = _runs(-seg, -seg.max()*ZERO_REL)
    if not zr:
        return [], 0.0
    zs = [a+lo for a, b in zr]
    ze = [b+lo for a, b in zr]
    zc = [(a+b)/2 for a, b in zr]
    gaps = [zc[i+1]-zc[i] for i in range(len(zc)-1)]
    #  用**中位**当张距 —— 牌内的图案也可能造成伪零
    pitch = float(np.median(gaps)) if gaps else 0.0
    if pitch <= 2:
        return [], 0.0
    #  名义 pitch 用于补齐：实测会偏小（GT own 92~98 vs 名义 103，
    #   实战 across 41~43 vs 名义 40.6）→ 取两者中较可信的
    nom = expect_pitch if expect_pitch > 2 else pitch
    # 牌区间 = 相邻零区间之间
    spans = [(ze[i], zs[i+1]) for i in range(len(zs)-1)]
    #  把「明显挤了多张」的宽段按 pitch 拆开
    out = []
    for a, b in spans:
        w = b-a
        k = int(round(w/nom)) if nom > 2 else 1
        if k >= 2:
            step = w/k
            for i in range(k):
                out.append((a+i*step, a+(i+1)*step))
        else:
            out.append((a, b))
    # 若最外侧还有白纸延伸，补上
    if zs[0]-lo > 0.4*pitch:
        out.insert(0, (float(lo), float(zs[0])))
    if hi-ze[-1] > 0.4*pitch:
        out.append((float(ze[-1]), float(hi)))
    #  合并被牌面图案「整列无白纸」切碎的段
    #   8p 的 4×2 深色圆圈让穿过圆心的整列剖面归零 → 一张牌被切成 3 段
    #   （19~23px）。 必须在 0.35×nom 过滤**之前**合并 —— 碎段本身就
    #   过不了那条线，先滤后合等于没合（第一版就犯了这个顺序错）。
    #   真牌缝与图案伪零缝的**宽度**可能相同（都 3~6px），不能按缝宽区分；
    #   但同一张牌的碎片**合并后宽度 ≈ 1 张牌**（≤1.15×nom），相邻两张
    #   真牌合并 ≈ 2×nom —— 按合并宽度区分是干净的。
    merged = []
    for a, b in out:
        if merged and b - merged[-1][0] <= 1.15*nom:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = [(float(a), float(b)) for a, b in merged if b-a >= 0.35*nom]
    out = out[:n_max]
    return out, pitch


# ═════════════════ 4. 切一家 ═════════════════
def _side_quad(cfg, tw, cx, cy, bw, bh, col=None, line=None, force_h=None):
    """侧家：中心 + 尺寸 → 平行四边形（TL/TR/BR/BL 四个顶点，整图坐标）。

    col  = 列号（0..2），line = 该列内的行号（0 起）——用于取逐列/逐行参数。
    force_h ≠ None 时忽略名义高，直接用 span 实测高（立直牌横放：
    牌面在行方向上比站牌长 ~35%，名义高会裁掉牌面下沿 —— ）。

     为什么不用矩形 
      侧家的牌在屏幕上就是平行四边形（从本家看过去，
      上家右斜、下家左斜）。GT 标注实测单位向量：
        left  上(+1.0000,−0.0092) 右(−0.0433,+0.5018)
        right 上(+0.9999,−0.0046) 右(+0.0925,+0.5177)
      用轴对齐矩形只框住牌的左半部分 → 识别率必然低。

     尺寸怎么定 
      掩膜量到的 bw/bh 是**轴对齐包围盒**，而真实牌是斜的，
      所以沿边方向的**投影长度**才是牌的边长：
        上边长 ≈ bw + |bh × right_dx|（斜边在 x 上的投影会多占）
      实测偏差 ±3px，够用。
    """
    ut = cfg.get('quad_top')
    ur = cfg.get('quad_right')
    if not ut or not ur:
        return None
    #  两个边长都用**名义值**，不用掩膜量出来的尺寸 
    #   掩膜白纸段的高只有牌高的 **0.53**（切 29 / 标 55）——
    #   因为侧家的牌横躺，牌的上半部被图案占满，零区间切到的是
    #   **牌与牌之间的缝**（缝隙宽度），不是整张牌的高度。
    #   → 高度必须是 tw×ar（牌宽 × 牌高比），宽度是 tw。

    #  ut / ur 都是「牌宽=1」基准下的位移向量（不是单位向量）
    #   ut = 上边向量 / 牌宽    = (+1.0000, −0.0092)   模长≈1.0（牌宽方向）
    #   ur = 右边向量 / 牌宽    = (−0.0433, +0.5018)   模长≈0.504= tw_ar
    #   → 右边向量 = ur × 牌宽（**不是** ur × 牌高！曾按牌高算，
    #     模长被缩成 0.50 → 牌高只给到 28（切出高 28/ 应 55））
    ws = cfg.get('quad_w_scale', 1.0)
    hs = cfg.get('quad_h_scale', 1.0)
    #  逐行的尺寸缩放（牌越靠下越大 —— 透视）
    #   实测（tools/edge_probe.py，白纸连通域当真值，GT 与实战曲线几乎重合）：
    #     宽/tw  r0 0.92 → r5 0.99   （+8%）
    #     高/tw  r0 0.44 → r5 0.51   （+16%）
    #   ⚠ 之前框高/框宽是**固定**的 → 远处的框比牌大（上下溢出），
    #     近处的框比牌小（切进牌内）—— 这正是用户报告的
    #     「顶（底）部缺失或扩大」，且**逐行不同**。
    #     用戶确认的物理：同列倾角相同、每张有微小抖动；
    #     真正逐行变的是**尺寸**。
    roww = cfg.get('row_w_scale')
    rowh = cfg.get('row_h_scale')
    if line is not None:
        if roww and 0 <= line < len(roww):
            ws *= roww[line]
        if rowh and 0 <= line < len(rowh):
            hs *= rowh[line]
    ux, uy = ut[0]*tw*ws, ut[1]*tw*ws
    if force_h:
        #  立直牌横放：保持右边斜率不变，高度用 span 实测值
        scale = force_h/max(abs(ur[1])*tw*ws, 1e-6)
        vx, vy = ur[0]*tw*ws*scale, ur[1]*tw*ws*scale
    else:
        vx, vy = ur[0]*tw*ws*hs, ur[1]*tw*ws*hs
    #  以「上沿中点」为原点铺开（cx,cy = 白纸段的上沿中点）
    #   注意：这样构造出的四边形，其**几何中心**在 (cx+vx/2, cy+vy/2)，
    #   不等于 (cx,cy)。曾误以为 (cx,cy) 是中心 → 整列 dy 偏 +15px
    #   （v/2 的 y 分量 13.8，与观测到的 +15 吻合）。
    hx = ux/2
    TL = (cx-hx, cy)
    TR = (cx+hx, cy)
    BR = (cx+hx+vx, cy+vy)
    BL = (cx-hx+vx, cy+vy)
    return (int(round(TL[0])), int(round(TL[1])),
            int(round(TR[0])), int(round(TR[1])),
            int(round(BR[0])), int(round(BR[1])),
            int(round(BL[0])), int(round(BL[1])))


def cut_seat(mask, W, H, seat, fill_thr=FILL_THR):
    cfg = SEATS[seat]
    tw = cfg['tw_ratio']*W
    th = tw*cfg['tw_ar']
    step = cfg['step']*tw
    w = cfg['win']
    WX0, WY0 = int(w[0]*W), int(w[1]*H)
    WX1, WY1 = int(w[2]*W), int(w[3]*H)
    ax, ay = cfg['anchor'][0]*W, cfg['anchor'][1]*H
    out = []

    n_line = cfg.get('n_line', MAX_LINE)
    for r in range(n_line):
        if cfg['stack'] == 'x':
            cy = ay + sum(step*(1+cfg['step_growth']*k) for k in range(r))
            #  逐行 y 修正（row_y_adj，见 SEATS 注释）
            adj = cfg.get('row_y_adj')
            dy0, dy1 = (adj[r] if adj and r < len(adj) else (0.0, 0.0))
            y0 = int(round(cy - th*0.55 + dy0*th))
            y1 = int(round(cy + th*0.55 + dy1*th))
            if y1 <= WY0 or y0 >= WY1:
                break
            y0 = max(y0, WY0); y1 = min(y1, WY1)
            band = mask[y0:y1, WX0:WX1]
            if band.size < 50:
                continue
            prof = band.sum(axis=0).astype(np.float32)
            spans, tp = cut_line(prof, 0, prof.size, cfg['n_slot'], cfg['pitch']*tw)
            #  span 纸面强度下限（详见下方 TILE_MIN 注释）
            spans = [(a, b) for a, b in spans
                     if prof[int(a):int(b)].max() >= TILE_MIN*band.shape[0]]
            for a, b in spans:
                #  整段必须落在窗口内 —— 边界残留（半张牌）要丢掉
                if a < 0 or b > prof.size:
                    continue
                out.append((int(WX0+a), y0, int(WX0+b), y1))
        else:
            #  用每列独立的中心 x（若配置了 anchor_xs）
            axs = cfg.get('anchor_xs')
            cx = (axs[r]*W if (axs and r < len(axs))
                  else ax + sum(step*(1+cfg['step_growth']*k) for k in range(r)))
            #  band 宽度必须限制在「本列 ± 半个牌距」，不能用整个窗口 
            #   侧家窗口右侧紧邻牌山（背面朝上的绿背牌），实测在 x=1709 处
            #   被切出 6 个框、GT 标注里根本没有 → 全是牌山。
            #   窗口是"矩形范围"，牌山与牌河在 y 上完全重叠，
            #   只有「列中心 ± 半个牌距」这条带才是纯牌河。
            band_hw = step*0.5
            x0 = int(round(cx-band_hw)); x1 = int(round(cx+band_hw))
            if x1 <= WX0 or x0 >= WX1:
                break
            x0 = max(x0, WX0); x1 = min(x1, WX1)
            band = mask[WY0:WY1, x0:x1]
            if band.size < 50:
                continue
            prof = band.sum(axis=1).astype(np.float32)
            spans, tp = cut_line(prof, 0, prof.size, cfg['n_slot'], cfg['pitch']*tw)
            #  span 纸面强度下限
            #   牌局早期某列只有 2 张时，band 下段是空桌面；相邻列的牌因
            #   skew 漂移会把左（右）边缘细条伸进本列 band —— 剖面出现
            #   峰值 2~13（band 宽 77 的 3~17%）的弱信号段，被当成牌位
            #   → 实机左家切出 3 个幻影空框。真牌的峰值是 56~64（73~83%）。
            #   → 峰值 < 25% × band 横向尺寸的一律不是牌。
            spans = [(a, b) for a, b in spans
                     if prof[int(a):int(b)].max() >= TILE_MIN*band.shape[1]]
            for si, (a, b) in enumerate(spans):
                if a < 0 or b > prof.size:
                    continue
                #  输出**平行四边形**而不是矩形 
                #   侧家的牌在屏幕上就是斜的（「上家右斜、下家左斜」）。
                #   用轴对齐矩形会只框住牌的左半部分 —— GT 实测对比图上
                #   矩形比四边形窄约 40%，识别率必然低。
                #
                #   构造方式：以该段白纸的**中心**为四边形中心，
                #   按 SEATS 里的单位向量铺开（上边沿 x、右边沿 y）。
                #  逐张按列倾斜修正中心 x 
                #   band 是竖直的，但牌列会随 y 倾斜（skews）。
                #   该张牌的实际中心 x = band 中心 + skew×(y − skew_ref)
                ay0, ay1 = int(WY0+a), int(WY0+b)
                sks = cfg.get('skews')
                sk = (sks[r] if sks and r < len(sks) else 0.0)
                cxs = (x0+x1)/2 + sk*(ay0 - cfg.get('skew_ref', 0.0)*H)
                #  立直牌横放检测：span 明显高于名义高 → 用实测高建框
                #   （48 vs 名义 35 = 1.37；站牌 34~39 = 0.97~1.11）。
                #   这样 quad 的 h/v 才会真正离群，规则 B 的换向能接住。
                nom_h = abs(cfg['quad_right'][1])*tw*cfg.get('quad_w_scale', 1.0)
                force = (ay1-ay0) if (ay1-ay0) > nom_h*1.25 else None
                qc = _side_quad(cfg, tw, cxs, ay0, x1-x0, ay1-ay0, col=r, line=si,
                                force_h=force)
                if qc is None:
                    continue
                #  过滤：四边形中心必须在「窗口内缩半牌宽」的范围内 
                #   否则会切到牌山（绿背牌，实测在 x≈1750~1772）
                #   和画面外的残留（y≈1078）
                qcx = (qc[0]+qc[2]+qc[4]+qc[6])/4
                qcy = (qc[1]+qc[3]+qc[5]+qc[7])/4
                mg = tw*0.5
                if not (WX0+mg <= qcx <= WX1-mg and WY0+mg*0.5 <= qcy <= WY1-mg*0.5):
                    continue
                out.append(qc)

    #  排序：先按「线」（列/行）聚类，再按线上位置
    #   stack='x'（自家/对家）：线沿 y 增、行内沿 x 增
    #   stack='y'（侧家）：线沿 x 增、列内沿 y 增
    #    侧家必须用「四边形的中心」而不是 t[0]（那只是 TL 的 x）——
    #     曾用 t[0] 导致三列切出结果交错（切出中心 1506/1629/1757
    #     vs 标注 1400/1525/1647，整体右移 106px）。
    #    桶取整 → 间隙聚类 
    #     旧实现 round(中心/半牌宽) 当桶键：一列的 cx 跨度（skew 抖动
    #     ~15px@1920）接近半个桶宽，桶边界落在哪张牌上由 canvas 原点
    #     决定 —— 实机换框后上家#10 因此指向另一张物理牌（regress 抓到）。
    #     改为：沿主方向排序，相邻间隙 > 半牌宽处断开成列/行，列内按次
    #     方向排。物理顺序稳定，与坐标系原点无关。
    #     间隙余量列内相邻 cx 差 ~3px@1920（~6px@GT）、
    #     列间最近间隙 ~36px@1920（~77px@GT）、阈值 = 半牌宽 28/59。
    def cxy(t):
        if len(t) == 8:
            return ((t[0]+t[2]+t[4]+t[6])/4, (t[1]+t[3]+t[5]+t[7])/4)
        return ((t[0]+t[2])/2, (t[1]+t[3])/2)
    #  视角顺序：从该家视角「第一行第一张→最后一行最后一张」。
    #   屏幕坐标系（图像 y 向下）实测（GT 校准）：
    #   own（下方）  ：行 y 升序（上排→下排）、行内 x 升序（左→右）
    #   across（上方）：行 y 降序（下排→上排）、行内 x 降序（右→左）
    #   left（左方） ：列 x 降序（右列→左列）、列内 y 升序（上→下）
    #   right（右方）：列 x 升序（左列→右列）、列内 y 降序（下→上）
    #   原来是全升序 → 只有 own 符合，其余三家顺序反了。
    VIEW_DIR = {
        'own': (1, 1),
        'across': (-1, -1),
        'left': (-1, 1),
        'right': (1, -1),
    }

    def cluster_sort(items, ip, isec, gap, dm=1, ds=1):
        items = sorted(items, key=lambda t: cxy(t)[ip], reverse=(dm < 0))
        groups, prev = [], None
        for t in items:
            if prev is None or abs(cxy(t)[ip] - cxy(prev)[ip]) > gap:
                groups.append([])
            groups[-1].append(t)
            prev = t
        res = []
        for grp in groups:
            res.extend(sorted(grp, key=lambda t: cxy(t)[isec], reverse=(ds < 0)))
        return res

    dm, ds = VIEW_DIR.get(seat, (1, 1))
    if cfg['stack'] == 'x':
        out = cluster_sort(out, 1, 0, max(th*0.5, 4), dm, ds)
    else:
        out = cluster_sort(out, 0, 1, max(tw*0.5, 4), dm, ds)
    return out


def split(im, dbg=False):
    bx0, by0, bx1, by1 = canvas(im)
    cv = im[by0:by1, bx0:bx1]
    m = paper_mask(cv)
    H, W = m.shape
    seats = {}
    for seat in SEATS:
        raw = cut_seat(m, W, H, seat)
        # 统一加上 canvas 的 box 偏移 → 整图坐标
        #   矩形：(x0,y0,x1,y1) → 每项都按 x/y 分别加
        #   四边形：(TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy) → 逐顶点加
        out2 = []
        for t in raw:
            if len(t) == 8:
                v = np.asarray(t, np.float32).reshape(4, 2)
                v = v + [bx0, by0]
                out2.append(tuple(int(round(x)) for xy in v for x in [xy[0]] for _ in [0])
                            if False else tuple(int(round(q)) for q in v.flatten()))
            else:
                a, b, c, d = t
                out2.append((a+bx0, b+by0, c+bx0, d+by0))
        seats[seat] = out2
    return (seats, (bx0, by0, bx1, by1), cv, m) if dbg else (seats, (bx0, by0, bx1, by1))


# ═════════════════ 5. 自检 ═════════════════
def _as_quad(t):
    """把切分输出统一成 4×2 四边形（TL/TR/BR/BL）。

     split() 的输出**已经是整图坐标**（box 偏移在里面加过了），
      所以这里**不要再加 box** —— 曾多加一次，算出 IoU 0.25 的假结果。
    """
    if len(t) == 8:
        return np.asarray(t[:8], np.float32).reshape(4, 2)
    x0, y0, x1, y1 = t
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def _poly_iou(a, b, shape):
    """两个四边形的 IoU（用 cv2 填充后求交并）—— 形状对不对的终极指标。"""
    H, W = shape
    ma = np.zeros((H, W), np.uint8)
    cv2.fillPoly(ma, [a.astype(np.int32)], 1)
    mb = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mb, [b.astype(np.int32)], 1)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter/max(union, 1)


def selftest():
    """GT 自检：**顶点级 IoU** —— 形状对不对的终极指标。

     为什么必须用 IoU 而不是「中心命中」
      中心命中只验位置，不验形状。曾用它在IoU 0.60 的情况下报69/69
      （轴对齐矩形 vs 平行四边形，形状差 40%），害我误判"框已套准"。
    """
    im, _ = paths.load_gt()
    d = json.load(open(paths.GT_JSON))['groups']
    seats, box, cv, m = split(im, dbg=True)
    H, W = m.shape
    shape = (H, W)
    print(f'GT 画面 {W}x{H}   canvas box={box}')
    print()
    print(f'{"座位":<8}{"切/标":>10}{"中心命中":>12}{"IoU 中位":>12}'
          f'{"IoU 最小":>12}{"顶点误差中位":>14}')
    tot_hit = tot_ann = tot_cut = 0
    ious = []
    for seat in ['own', 'across', 'left', 'right']:
        cut = seats[seat]
        quads = [_as_quad(t) for t in cut]
        ann = []
        for a in d[f'{seat}|river']:
            q = np.asarray(a['quad'], np.float32)      # 整图坐标
            ann.append(q)
        hit = 0
        v_err = []
        row_iou = []
        for qa in ann:
            ca = qa.mean(axis=0)
            best, bd = None, 1e18
            for qc in quads:
                cc = qc.mean(axis=0)
                dd = float(((cc-ca)**2).sum())
                if dd < bd:
                    bd, best = dd, qc
            if best is None:
                continue
            if bd < (50**2):
                hit += 1
            v_err.append(float(np.abs(best-qa).mean()))
            row_iou.append(_poly_iou(best, qa, shape))
        ious += row_iou
        f = lambda v: (f'{np.median(v):.3f}' if v else '  n/a')
        print(f'{seat:<8}{len(cut):>4}/{len(ann):<5}{hit:>6}/{len(ann):<5}'
              f'{f(row_iou):>12}{f(row_iou) if not row_iou else f"{min(row_iou):.3f}":>12}'
              f'{f(v_err):>14}')
        tot_hit += hit
        tot_ann += len(ann)
        tot_cut += len(cut)
    print()
    print(f'  合计：切出 {tot_cut} 张，标注 {tot_ann} 张，'
          f'中心命中 {tot_hit}/{tot_ann}，IoU 中位 {np.median(ious):.3f}')
    return tot_hit, tot_ann, tot_cut


if __name__ == '__main__':
    selftest()