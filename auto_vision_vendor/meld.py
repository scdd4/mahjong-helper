"""各家副露切分 + 识别（移植 Swift MeldReader，）。

## 区域（虚拟坐标 2560×1440，按 2560×1440 虚拟坐标标定）
  own    右下（手牌右侧），平放透视牌
  left   左侧中部，**竖摞**（上家副露从上往下摞）
  right  右侧（下家）， 区域待标定（7806 被角色立绘遮挡未验证）
  across 顶部右侧，牌面**倒置**（对家视角）

## 切分（Swift 验证过的口径）
白纸掩膜（lum>150 且 sat<45）→ 补洞（边界灌水，图案是洞）→ 8 邻域连通域。
 不做闭运算/膨胀：牌缝本来就是断开的。

## 识别
每块 = 一张牌。valid = 本块腐蚀 2px（去掉与邻牌相接的一圈，否则图案框
被撑大、NCC 掉到 0.2）。图案墨迹 = max(绝对暗度, 红墨) 限制在 valid 内，
紧裁 bbox → fit_to_box 到模板尺寸 → rank（IoU+NCC）。
每块独立试 4 个旋向（竖牌 0°、横躺 90°/270°、对家倒置 180°）。
无墨 = 白板（5z），不是识别失败。
"""
import os
import sys

import cv2
import numpy as np


from . import split as RS
from . import reader as V
from . import orientation as ORI
from . import paths

# 副露区（虚拟坐标 2560×1440）+ 每家分组轴 + 延伸方向
#
#  延伸方向 grow
#   第一个副露标定起点，后续副露沿该家方向延伸，直到遇桌布/手牌：
#   - own  手牌在左，副露在手牌右，新组往**右**（+x）
#   - left 第一个在左下角，新组往**上方**（-y）——实测确认
#   - right 画面**右上角**，牌侧躺（rot±90），第一个在上、新组往**下**（+y）
#     
#   - across 顶部**左侧**（对家牌河左上），牌倒置（rot2），新组往**右**？（+x）
#     
SEATS = {
    'own':    dict(region=(1760, 1245, 690, 150), axis='x', grow=+1),
    'left':   dict(region=(70, 780, 270, 510), axis='y', grow=-1),
    #    collect=False 的家（own/left）走 er 原路径——通用贴边收编会把
    #     bbox 内邻牌的墨/立体暗边吸进来，毁掉正立牌的图案

    'right':  dict(region=(1965, 33, 193, 236), axis='y', grow=+1, min_side=0.35,
                   collect=True, blob=True),
    #    right blob=True：竖摞相邻，块内混「下张顶边」杂墨，取主墨团去杂
    #     （字牌单团无害；筒子多圆不可开——main_blob 会砍到只剩一团）
    'across': dict(region=(480, 40, 600, 190), axis='x', grow=-1, min_side=0.38,
                   collect=True),
    #    across dedupe=True：倒置平放的字可能贴块边（补洞不算洞），
    #     须主墨团；own/left 是单张多笔画团（六+萬），取主团丢笔画——不可开
}

MIN_AREA = 300

_tpl_red_cache = {}


def _tpl_red_ratio(name):
    """模板图案 bbox 内的红墨占比（与 meld 块的紧裁口径一致）。"""
    if name in _tpl_red_cache:
        return _tpl_red_cache[name]
    import json as _json
    meta = _json.load(open(os.path.join(paths.CLEAN, 'meta.json')))
    bgr = cv2.imread(os.path.join(paths.CLEAN, name + '.png'))
    if bgr is None:
        _tpl_red_cache[name] = None
        return None
    pat = cv2.imread(os.path.join(paths.CLEAN, name + '.pattern.png'),
                     cv2.IMREAD_GRAYSCALE)
    if pat is None:
        _tpl_red_cache[name] = None
        return None
    ys, xs = np.nonzero((255 - pat) / 255.0 > 0.5)
    if len(xs) < 10:
        _tpl_red_cache[name] = None
        return None
    sub = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # 与块的 _ink_of / red_ch 同口径：绝对暗度 + 红墨
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rr = sub[:, :, 2].astype(np.float32)
    gg = sub[:, :, 1].astype(np.float32)
    bb = sub[:, :, 0].astype(np.float32)
    ink = np.clip((150.0 - g) / 75.0, 0.0, 1.0)
    red = np.clip((rr - np.maximum(gg, bb) - 18.0) / 60.0, 0.0, 1.0)
    m = ink > 0.5
    v = float((ink[m] * red[m]).sum() / ink[m].sum()) if m.any() else 0.0
    _tpl_red_cache[name] = v
    return v
MIN_SCORE = 0.30
ROT_ALL = (0, 1, 2, 3)            # 每块全试 4 旋向（混排时无法预判）


def _ink_of(bgr):
    """图案墨迹：绝对暗度 + 红墨（与牌河/Swift 一致）。"""
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    b = bgr[:, :, 0].astype(np.float32)
    k = np.clip((150.0 - lum) / 75.0, 0.0, 1.0)
    red = np.clip((r - np.maximum(g, b) - 18.0) / 60.0, 0.0, 1.0)
    return np.maximum(k, red)


def _is_paper(bgr):
    """白纸判据：亮 + 低饱和（Swift 原口径，副露场景已验证）。"""
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mx = bgr.max(axis=2).astype(np.int16)
    mn = bgr.min(axis=2).astype(np.int16)
    return (lum > 150) & (mx - mn < 45)


def _fill_holes(paper):
    """补洞：从边界灌水，灌不进的非白像素 = 洞（图案）。

    用 OpenCV 的 floodFill（C 实现），与逐像素洪水等价但快两个量级——
    这函数在每帧里要跑十几次，是识别耗时的大头。
    """
    H, W = paper.shape
    inv = (~paper).astype(np.uint8)          # 1 = 非纸
    if not inv.any():
        return paper.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    seeds = []
    for x in range(W):
        if inv[0, x]:
            seeds.append((x, 0))
        if inv[H - 1, x]:
            seeds.append((x, H - 1))
    for y in range(H):
        if inv[y, 0]:
            seeds.append((0, y))
        if inv[y, W - 1]:
            seeds.append((W - 1, y))
    for sx, sy in seeds:
        if inv[sy, sx] == 1:
            cv2.floodFill(inv, mask, (sx, sy), 2)
    outside = (inv == 2)
    solid = paper.copy()
    solid[~paper & ~outside] = True
    return solid


def _erode_mask(own, it=2):
    """布尔腐蚀（4 邻域 × it 次）。"""
    m = own.copy()
    for _ in range(it):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


def _rotate(plane, times):
    return np.rot90(plane, times % 4)


def _main_blob(ink, thr=0.5):
    """取墨迹的**最大连通域**，其余清零。

     叠放副露（对家/上家）的块里常混入「下一张牌的顶边」墨迹——
     不清掉的话 fit_to_box 的拉伸基准被污染，倒字匹配全废（7806 实测）。
    """
    m = (ink > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return ink
    best = 1 + int(np.argmax([stats[i][4] for i in range(1, n)]))
    x, y, w, h, _ = stats[best]
    out = np.zeros_like(ink)
    keep = lab[y:y + h, x:x + w] == best
    out[y:y + h, x:x + w] = ink[y:y + h, x:x + w] * keep
    return out


def _order_quad(pts):
    """4 顶点 → TL/TR/BR/BL 顺序（sum=x+y 最小 TL、diff=y-x 最小 TR）。"""
    p = np.asarray(pts, np.float32).reshape(4, 2)
    tl = p[np.argmin(p.sum(axis=1))]
    br = p[np.argmax(p.sum(axis=1))]
    diff = p[:, 1] - p[:, 0]
    tr = p[np.argmin(diff)]
    bl = p[np.argmax(diff)]
    return np.array([tl, tr, br, bl], np.float32)


def _paper_quad(rgb, block_mask=None):
    """白纸本体 → 凸包四边形（4 顶点）。副露牌面是透视平行四边形，
    轴对齐 bbox 会兜进侧阴影/牌底；用牌河同款 paper_body（排除
    sat<48 侧阴影 + lum 阈值的牌底）再凸包逼近真实牌面四角。"""
    #  不用 V.paper_body——它的「离中心最近 3 域」在副露牌上只抓到
    #   牌面上部（图案把白纸切碎，下部小域不够 big 门槛）→ 四边形缺半张。
    #   改：牌河白纸判据（lum>0.90×p98 & sat<48，排除侧阴影/牌底）
    #   → 补洞（图案是洞）→ 最大域 = 完整牌面 → 凸包。
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = (rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16))
    m = ((g > 0.90 * float(np.percentile(g, 98))) & (sat < 48)).astype(np.uint8)
    if block_mask is not None:
        #  用切分时的本块 solid 掩膜约束——副露牌相邻，白纸域在透视下
        #   会跨到邻牌（across[1]/right[0] 的 warp 偏到邻牌）
        m = m & (block_mask > 0)
    if m.sum() < 50:
        return None
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    solid_m = _fill_holes(m2.astype(bool)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid_m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax([stats[k][4] for k in range(1, n)]))
    bx, by, bw, bh, ba = (int(v) for v in stats[i])
    if ba < 400 or min(bw, bh) < 20:
        return None
    ys, xs = np.nonzero(lab[by:by+bh, bx:bx+bw] == i)
    xs = xs + bx
    ys = ys + by
    if len(xs) < 60:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    #  极值点取角：凸包 approxPolyDP 会把透视角抹到边上
    #   （左上角顶点跑到上边中间，warp 后牌面斜置）。透视凸四边形的
    #   4 个极值点 = 4 个真角，且 TL→TR→BR→BL 天然顺时针与 dst 同手性。
    d_sum = pts.sum(axis=1)
    d_diff = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(d_sum)]
    br = pts[np.argmax(d_sum)]
    tr = pts[np.argmin(d_diff)]
    bl = pts[np.argmax(d_diff)]
    approx = np.array([tl, tr, br, bl], np.float32)
    # 退化保护：四点重合（细长块）→ 弃
    if len({tuple(p) for p in approx}) < 4:
        return None
    return approx


def _recognize_tile(rgb, quad, T, seat='across'):
    """四边形 warp 透视校正 → 裁正立牌面 → 4 旋向试 → 识别。
    返回 (score, name, rot, face)。 顺序：warp → take_face → rot
    （先转再裁会在 rot1/3 时横切一半，同牌河 face_of 的教训）。"""
    q = np.asarray(quad, np.float32)
    ar = V.quad_ar(q)
    dst = np.array([[0, 0], [V.TPL_W, 0], [V.TPL_W, V.TPL_H], [0, V.TPL_H]],
                   np.float32)
    w = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(q, dst),
                            (V.TPL_W, V.TPL_H), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    x1, y1, x2, y2 = V.RIVER_FACE_BOX
    face = np.ascontiguousarray(w[y1:y2, x1:x2])
    best = (0.0, None, 0)
    for rot in range(4):
        f = np.rot90(face, rot)
        res = V.extract_pattern(f)
        if res is None:
            continue
        pat, _box = res
        if pat is None or isinstance(pat, str):   # (None,None) / 'BLANK'
            continue
        sc = V.rank(pat, T)
        if not sc:
            continue
        score, name = sc[0][0], sc[0][1]
        #  规则 D 万子拓扑仲裁：
        #   万子数字近邻混淆（横躺 3m↔5m、2m↔8m，分差仅 0.01~0.04）时，
        #   用「数字区横向块数 / 纵向峰数」消歧。只在 top5 有该万子且
        #   分差 < 0.06 时生效（同牌河 arbitrate_kan）。
        top5 = [x[1] for x in sc[:5]]
        gap = sc[0][0] - sc[1][0] if len(sc) > 1 else 9.9
        # 规则 D 的可用性统一由 orientation 决定（名义摆向 + 非横 warp）；
        # 低分时传 None → 只允许模板唯一拓扑救援（同牌河）
        fkan = (V.arbitrate_kan(pat, top5,
                                gap if sc[0][0] >= V.KAN_ARB_LOW else None)
                if ORI.topology_allowed(seat, rot, ar) else None)
        if fkan is not None:
            name, score = fkan, 9.0
        if score > best[0]:
            best = (score, name, rot)
    return best + (face,)


def _back_ratio(im, r):
    """矩形区域里「牌背」的像素占比。

    牌背颜色随皮肤变（默认皮肤是绿、换肤后常见橙黄），统一按
    「高饱和 + 中等亮度」判定；牌面（白底）饱和低，不会被算进来。
    """
    x0 = max(0, int(r[0])); y0 = max(0, int(r[1]))
    x1 = min(im.shape[1], int(r[2])); y1 = min(im.shape[0], int(r[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    green = (h > 35) & (h < 95)
    orange = (h > 8) & (h < 32)
    m = ((green | orange) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60))
    return float(m.mean())


def _is_back_block(im, tile):
    """该块是不是牌背（暗杠两端盖住的牌，不是牌面）。

    左家暗杠下端的牌背斜边被当成白纸切出「三角」块，
    还识别成 5z(0.436)——必须当背牌剔除，否则凑成 3 张绕过暗杠判定。
    """
    q = tile['quad']
    return _back_ratio(im, q) > 0.5


def _ankou_sides(im, group, axis='x'):
    """2 张明牌、两端外侧各有一张**牌背** → 暗杠。

    雀魂暗杠渲染：背 / 明 / 明 / 背 —— 两端盖住，中间两张亮面。
    * 自家/对家副露沿 x 排列（axis='x'）→ 查左右；
    * 上家/下家副露沿 y 排列（axis='y'）→ 查上下（左家暗杠）。
    识别只能看到中间 2 张明牌，组大小校验（≥3）会把它当误报丢掉。
    """
    if len(group) != 2:
        return False
    key = 0 if axis == 'x' else 1
    g = sorted(group, key=lambda t: t['quad'][key])
    n0, n1 = g[0]['quad'], g[-1]['quad']
    if axis == 'x':
        w = n0[2] - n0[0]
        before = _back_ratio(im, (n0[0] - w * 1.1, n0[1], n0[0] - w * 0.1, n0[3]))
        after = _back_ratio(im, (n1[2] + w * 0.1, n1[1], n1[2] + w * 1.1, n1[3]))
    else:
        h = n0[3] - n0[1]
        before = _back_ratio(im, (n0[0], n0[1] - h * 1.1, n0[2], n0[1] - h * 0.1))
        after = _back_ratio(im, (n1[0], n1[3] + h * 0.1, n1[2], n1[3] + h * 1.1))
    return before > 0.25 and after > 0.25


def _recognize_roi(im, x0, y0, x1, y1, T, axis, blob, min_side, expect,
                   collect=False):
    """识别一个 roi 内的副露牌，返回整图坐标的 tiles 列表。"""
    height, width = im.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return []
    roi = im[y0:y1, x0:x1]
    paper = _is_paper(roi)
    if not paper.any():
        return []
    solid = _fill_holes(paper)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= MIN_AREA and min(w, h) >= min_side:
            comps.append((int(x), int(y), int(w), int(h), int(i)))
    comps.sort(key=lambda c: c[0] if axis == 'x' else c[1])

    rgb_roi = roi.astype(np.float32)
    red_ch = np.clip((rgb_roi[:, :, 2] - np.maximum(rgb_roi[:, :, 1],
                     rgb_roi[:, :, 0]) - 18.0) / 60.0, 0.0, 1.0)

    tiles = []
    for cx, cy, cw, ch, lab_id in comps:
        qx0, qy0 = x0 + cx, y0 + cy
        qx1, qy1 = qx0 + cw, qy0 + ch
        quad = [qx0, qy0, qx1, qy1]

        # 角色立绘过滤：块内暖色占比 >0.5（头像/立绘是亮色低饱和暖色调）
        blk = roi[cy:cy + ch, cx:cx + cw]
        bb, gg, rr = cv2.split(blk.astype(np.int16))
        warm = float(((rr > gg + 15) & (rr > 60)).mean())
        if warm > 0.5:
            continue

        #  四边形识别：白纸本体 → 凸包四边形 → warp 透视校正。
        #   副露牌透视比牌河重（平行四边形），轴对齐 bbox 会兜进侧阴影/牌底。
        pad = 6
        py0 = max(0, cy - pad); py1 = min(roi.shape[0], cy + ch + pad)
        px0 = max(0, cx - pad); px1 = min(roi.shape[1], cx + cw + pad)
        sub = roi[py0:py1, px0:px1]
        bmask = np.zeros(roi.shape[:2], np.uint8)
        bmask[cy:cy + ch, cx:cx + cw] = (lab[cy:cy + ch, cx:cx + cw] == lab_id)
        pq = _paper_quad(sub, bmask[py0:py1, px0:px1])
        if pq is None:
            continue
        quad_full = (pq + np.array([px0, py0], np.float32)).tolist()
        # quad_full 是 roi 坐标 → 整图坐标 = + (x0, y0)
        # （ 别用 qx0+x0：qx0 已含 x0+cx，会双重偏移，画出来的框整体跑偏）
        quad_img = [[int(round(x0 + p[0])), int(round(y0 + p[1]))]
                    for p in quad_full]

        best = _recognize_tile(roi, np.asarray(quad_full, np.float32), T)
        if best[1] is None:
            tiles.append(dict(name='5z', score=0.0, rotated=False,
                              quad=[qx0, qy0, qx1, qy1], quad4=quad_img))
            continue
        name = best[1] if best[0] >= MIN_SCORE else None
        # 赤牌红占比裁决（红色旋转不变）： 用 warp 后正立 face 的图案
        # 红占比——块内口径会被邻牌的紅萬字污染（left 横躺 5m
        # pr=0.560 vs 中点 0.559 恰好翻错）
        if name and len(name) == 2 and name[0] in ('5', '0'):
            suit = name[1]
            r5 = _tpl_red_ratio('5' + suit)
            r0 = _tpl_red_ratio('0' + suit)
            if r5 is not None and r0 is not None:
                f = np.rot90(best[3], best[2])
                ink_f = V.ink_of(f)
                rr_, gg_, bb_ = (f[:, :, i].astype(np.float32) for i in (2, 1, 0))
                red_f = np.clip((rr_ - np.maximum(gg_, bb_) - 18.0) / 60.0,
                                0.0, 1.0)
                mm = ink_f > 0.5
                if mm.any():
                    pr = float((ink_f * red_f)[mm].mean())
                    name = ('0' if pr >= (r5 + r0) / 2 else '5') + suit
        tiles.append(dict(name=name, score=round(best[0], 4),
                          rotated=best[2] != 0, quad=[qx0, qy0, qx1, qy1],
                          quad4=quad_img))
    return tiles


def _overlap(a, b):
    """两个矩形（[x0,y0,x1,y1]）中心距 < 半张牌 → 视为同一张。"""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return abs(ca[0] - cb[0]) < 20 and abs(ca[1] - cb[1]) < 20


def read_all(im, canvas_box=None, verbose=False):
    """读四家副露。返回 {seat: dict(groups=[[tile...]], note=str)}。

    tile = dict(name, score, rotated, quad=[x0,y0,x1,y1] 整图坐标)

     延伸识别：起点 region 识别第一组，然后从第一组的
      远端边沿 grow 方向逐「一组跨度」窗口延伸，遇无牌窗口即停（桌布/手牌）。
    """
    H, W = im.shape[:2]
    bx0, by0, bx1, by1 = canvas_box or RS.canvas(im)
    CW, CH = bx1 - bx0, by1 - by0
    T, _ = V.load_templates()
    expect = W * 0.049

    out = {}
    for seat, cfg in SEATS.items():
        rx, ry, rw, rh = cfg['region']
        axis = cfg['axis']
        grow = cfg.get('grow', 0)
        blob = cfg.get('blob', False)
        collect = cfg.get('collect', False)
        min_side = expect * cfg.get('min_side', 0.45)

        x0 = max(0, bx0 + int(rx / 2560 * CW))
        y0 = max(0, by0 + int(ry / 1440 * CH))
        x1 = min(W, x0 + int(rw / 2560 * CW))
        y1 = min(H, y0 + int(rh / 1440 * CH))
        if x1 - x0 < 24 or y1 - y0 < 24:
            out[seat] = dict(groups=[], note='区域太小')
            continue

        # 起点窗口（第一组）
        all_tiles = _recognize_roi(im, x0, y0, x1, y1, T, axis, blob,
                                   min_side, expect, collect)

        # 沿 grow 方向延伸：从第一组的远端边逐窗口扫描
        if grow != 0 and all_tiles:
            step = int(expect * 2.5)          # 一组副露沿主轴跨度
            gap = int(expect * 0.4)           # 组间空隙
            if axis == 'x':
                lo = min(t['quad'][0] for t in all_tiles)
                hi = max(t['quad'][2] for t in all_tiles)
                wy0 = min(t['quad'][1] for t in all_tiles) - int(expect * 0.5)
                wy1 = max(t['quad'][3] for t in all_tiles) + int(expect * 0.5)
                cur = hi if grow > 0 else lo
                for _ in range(4):
                    if grow > 0:
                        wx0, wx1 = cur + gap, cur + gap + step
                    else:
                        wx1, wx0 = cur - gap, cur - gap - step
                    if wx0 < 0 or wx1 > W:
                        break
                    extra = _recognize_roi(im, wx0, wy0, wx1, wy1, T, axis,
                                           blob, min_side, expect, collect)
                    if not extra:
                        break                    # 遇桌布/手牌
                    for t in extra:
                        if not any(_overlap(t['quad'], p['quad'])
                                   for p in all_tiles):
                            all_tiles.append(t)
                    cur = wx1 if grow > 0 else wx0
            else:
                lo = min(t['quad'][1] for t in all_tiles)
                hi = max(t['quad'][3] for t in all_tiles)
                wx0 = min(t['quad'][0] for t in all_tiles) - int(expect * 0.5)
                wx1 = max(t['quad'][2] for t in all_tiles) + int(expect * 0.5)
                cur = hi if grow > 0 else lo
                for _ in range(4):
                    if grow > 0:
                        wy0, wy1 = cur + gap, cur + gap + step
                    else:
                        wy1, wy0 = cur - gap, cur - gap - step
                    if wy0 < 0 or wy1 > H:
                        break
                    extra = _recognize_roi(im, wx0, wy0, wx1, wy1, T, axis,
                                           blob, min_side, expect, collect)
                    if not extra:
                        break
                    for t in extra:
                        if not any(_overlap(t['quad'], p['quad'])
                                   for p in all_tiles):
                            all_tiles.append(t)
                    cur = wy1 if grow > 0 else wy0

        # 分组：相邻块中心距 > 1.45×牌距 断开（牌距 = 中位块宽）
        tiles = sorted(all_tiles, key=lambda t: t['quad'][0] if axis == 'x'
                       else t['quad'][1])
        groups, cur, ref = [], [], 0.0
        for t in tiles:
            c = (t['quad'][0] + t['quad'][2]) / 2 if axis == 'x' \
                else (t['quad'][1] + t['quad'][3]) / 2
            if cur:
                gap_now = abs(c - ref)
                ref_w = np.median([x['quad'][2] - x['quad'][0] if axis == 'x'
                                   else x['quad'][3] - x['quad'][1]
                                   for x in cur])
                if gap_now > max(ref_w * 1.45, 60):
                    groups.append(cur)
                    cur = []
            cur.append(t)
            ref = c
        if cur:
            groups.append(cur)

        #  组大小过滤：一组副露最少 3 张。
        #   之前 {3,4} 会误伤「2 组副露紧挨」的情况——6 张 x 间距 < 1.45×牌宽
        #   分不开 → 并成 6 张一组 → 被 {3,4} 丢弃 → 0 张（自家
        #   副露 6 张一张没认出的根因）。误报的组通常是 2 张（摸牌/牌河碎块），
        #   仍被 ≥3 过滤。
        #  绿背块剔除：暗杠两端的绿背斜边会被
        #   当成白纸切出斜三角块、识别成低分假牌（5z 0.436）——先剔除，
        #   否则「2 明 + 1 假」凑成 3 张绕过暗杠判定。
        groups = [[t for t in g if not _is_back_block(im, t)] for g in groups]
        groups = [g for g in groups if g]
        #  组大小过滤：正常副露最少 3 张。
        #   补：暗杠只露出中间 2 张明牌（两端背牌不识别）——
        #   若两端外侧各有一张牌背（_ankou_sides，按列方向）则保留并标 kan。
        kept = []
        for g in groups:
            if len(g) >= 3:
                kept.append(g)
            elif _ankou_sides(im, g, axis):
                for t in g:
                    t['kan'] = True
                kept.append(g)
        groups = kept
        tiles = [t for g in groups for t in g]

        named = [t['score'] for t in tiles if t['name'] and t['score'] > 0]
        conf = float(np.mean(named)) if named else 0.0
        note = f'认出 {len(named)}/{len(tiles)} 张'
        if any(t.get('kan') for t in tiles):
            note += '（含暗杠）'
        out[seat] = dict(groups=groups, note=note, confidence=round(conf, 3))
    return out


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(_HERE), 'fixtures',
        'live_chi_from_kamicha.jpg')
    im = cv2.imread(p)
    r = read_all(im)
    print(os.path.basename(p))
    for seat, v in r.items():
        names = [' '.join(t['name'] or '?' for t in g) for g in v['groups']]
        print(f"  {seat:7s} {v['note']}  |  " +
              ' || '.join(names))
