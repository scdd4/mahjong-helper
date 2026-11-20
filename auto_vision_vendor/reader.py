"""  正式管线：牌河 / 副露的完全本地识别

三条新增通用规则（都不依赖"这是立直/吃/碰"的语义标注，只靠可测的几何与分数）：

┌─ 规则A  边界碎块剔除 ────────────────────────────────────────────
  现象：牌河里某张牌的图案 bbox 被**裁块边角上的桌布阴影**撑大。
  实例：across#32 的 6z（發），右下角混进一块 22px 的阴影（占墨迹 1.4%），
        bbox 从模板的 62×60 被撑成 94×68，IoU 从 0.71 掉到 0.34。
  判据：连通块 **面积占比 < max_frac** 且 **贴着裁块边界** → 删。
   两个条件缺一不可（都实测过）：
      只看面积 → 万子的横（占 9.3%）被删 → 2m 变 1m，全量掉到 65/76
      只看贴边 → 8p 的图案本来就到底边，整张被删
   试过但**不能用**的两条路（都留了记录）：
      · 按墨迹浓度过滤 —— 阴影块核心浓度 0.933，真笔画 1.000，几乎不可分
      · 用 `g > 0.80×纸面亮度` 补 body —— 万子笔画只有纸面的 0.62，被当阴影切掉

┌─ 规则B  几何离群 → 试全部 4 个旋向 ────────────────────────────
  现象：立直牌横放、吃/碰的搭牌侧放 —— 相对该家方向躺倒了 90°。
  实例：across#35（对家立直）四边形 h/v=2.37，同组中位 1.25
        right#61（kāechī 的搭牌）h/v=1.53，同组中位 2.44
  判据：同组内 h/v 偏离中位数 > geo_tol → 该牌试 4 个旋向，取最高分。
   只对离群牌试旋向，其余 73 张一律用该家基准旋向 → 不会连累已对的。

┌─ 规则C  IoU + NCC 组合评分 ────────────────────────────────────
  现象：IoU 只看"重合面积"，对**笔画粗细差异**不敏感，
        会出现 5s vs 0s 差 0.002 的平局（但 NCC 能分：0.726 vs 0.707）。
  做法：score = 0.5×IoU + 0.5×NCC。
   alpha 在 0.0~0.8 全区间都是 75/76（1.0 才掉到 74），
    取 0.5 是取区间中点，不是取最优 —— 稳健优先。

 踩过的坑：face 旋转的顺序 —— `take_face` 的裁剪框 (3,2,77,127) 是按
  125×74 定的，必须**先裁后转**。先 rot90 再裁，rot=1/3 时尺寸变成 74×125，
  框就把牌横切一半。rot=2（180°）尺寸不变，所以这个 bug 藏了很久，
  只在规则B 开始试 rot1/rot3 时才暴露。

【当前成绩】76 张全量 0.30 秒：**top1 = 76/76，top3 = 76/76**

规则D —— 万子拓扑仲裁（用户质疑后加的）
  用户："2m 两个横、8m 一撇一捺，怎么看都不会错啊" —— 他对。
   把own#6 判成 8m，查清了三个叠加原因：
  ① 「萬」稀释判别信号：数字只占墨迹 27.5%，「萬」占 72.5%，
     而 2m/3m/8m 的「萬」完全相同。
       只看数字区（上 32 行）：vs 2m IoU 0.639 / vs 8m IoU 0.370  ← 分得开
       看全图：              vs 2m IoU 0.617 / vs 8m IoU 0.650  ← 8m 反而赢
  ② 各向异性拉伸（y×0.774）把牌河的两条横压到 8m 那一行附近 → 像素层面两者靠近
  ③ 牌河与手牌是**两套渲染**，连通域拓扑系统性不同：
       "块数与模板一致"全量 22 张万子命中 **0/22** → **跨渲染比拓扑必然失效**

  解法：用「有几条横」这种拓扑量绕开压缩比，且**只用牌河侧自洽**的统计。
     (横向块数, 纵向峰数) = (1,2) → 2m　　(2,1) → 3m
     全量 22 张万子里这两档 **6/6 全对**；其余 (1,1) 16 张挤在一起，
     不硬编造分类，交给像素比对。
   必须限定"像素 top5 里有该万子"才允许仲裁 ——
    否则 `kan_top` 会把筒子的圆环、索子的分节也当成"萬"，
    3p/4p/9p/3s 被误判成 2m/3m，全量从 76掉到 72。
    万子必须**先判花色**，拓扑只在万子内部消歧。
"""
import json, os
import cv2
import numpy as np

from . import orientation as ORI

from . import paths as _P
CLEAN = _P.CLEAN

TPL_W, TPL_H = 80, 129

#  顶点顺序只用**循环平移**（保持手性），绝不用 (0,3,2,1) 那种镜像交换。
#   经 tools/sweep_rot.py 穷举 4×4 组合逐组验证，这四组是全局最优：
#     own=(shift0,rot0)  across=(shift0,rot2)  left=(shift1,rot0)  right=(shift1,rot2)
SEAT_SHIFT = {'own': 0, 'across': 0, 'left': 1, 'right': 1}
SEAT_ROT   = {'own': 0, 'across': 2, 'left': 0, 'right': 2}

# 规则A 参数
SPECK_MAX_FRAC = 0.08      # 面积占比阈值。0.03~0.12 全区间结果一致
SPECK_BORDER   = 1         # 距裁块边界几个像素算"贴边"

#  白板判据：ink 占牌面本体的比例低于此 → 判白板（5z）
#   起因：
#     上家#10  白板 → None   ink/body = 0.0112
#     自家#2   白板 → 3p     ink/body = 0.0013
#     左#11/右#5 白板 → 5z ✓   ink/body = 0.0000（墨迹为零，已能判对）
#   上两张之所以判错，是因为它们**混进了杂质**（邻牌边缘 / 桌布带），
#   墨迹数超过 6 下限→ 不走'BLANK' 分支 → 撑大 bbox → 提取失败或误匹配。
#
#    阈值定法（tools/stress_blank，GT 上做退化压力测试）
#   实测 GT 全部 67 张非白板，在**实战可比条件**（降采样 0.40~0.55、
#   JPEG q60~95，对应牌宽 47~70px，与实战 56.5px 一致）下的 ink 占比：
#     最小 0.0806   p1 0.0858   中位 0.27
#   而两张真白板是 0.0112 / 0.0013 → **裕量 7.2 倍**。
#   取 0.045（≈0.0806/1.8，两侧都留够）。
#
#   ⚠ 定阈值时必须用「实战可比条件」，不能用更苛刻的：
#     若按 0.30 倍降采样（牌宽 35px）测，非白板会掉到 0.0135，
#     与白板的 0.0112 只差 0.0005 → 阈值无从下手。
#     实战牌宽是 56.5px，35px 那种画质在雀魂里根本不会出现。
BLANK_INK_MAX = 0.045
# 规则B 参数
GEO_TOL        = ORI.GEO_TOL   # 单一来源见 orientation.py（GT 校准 0.15~0.30）
# 规则C 参数
ALPHA          = 0.5       # score = α·IoU + (1−α)·NCC
# 规则D 参数（万子拓扑仲裁）
KAN_TOP_SHARE  = 0.28      # 最大连通域占墨迹多少以上才认作"萬"
KAN_TOP_MAXFRAC = 0.75     # 「萬」顶边不能超过牌面高度的这么多
KAN_MIN_AREA   = 60        # 数字区连通域的最小面积（滤抗锯齿噪点）
KAN_PEAK_THR   = 0.15      # 纵向投影的峰阈值（相对最大值）
#  数字区拓扑 → 牌名。**只用牌河侧自洽**的量，不跨渲染比拓扑（见下）
#   统计映射（牌河实测自洽，像素分不开时用）：(1,2)/(2,1)
KAN_TOPO_STAT = {(1, 2): '2m', (2, 1): '3m'}
#  模板唯一映射：1-9m 模板的 digit_topology 全表里，
#   只有这 4 个拓扑只对应 1 个数字：
#     2m=(2,3) 3m=(3,2) 6m=(3,1) 8m=(2,2)
#   （1/4/5/7/9m 全是 (1,2)——共享拓扑不能反向仲裁）。
#   用途：**低分救援**——切分 quad 偏差让图案缩水时（own#0 二萬被读成
#   0m 0.39，gap 0.084 > KAN_ARB_GAP 挡住仲裁），唯一拓扑可以突破 gap。
KAN_TOPO_UNIQUE = {(2, 3): '2m', (3, 2): '3m', (3, 1): '6m', (2, 2): '8m'}
KAN_TOPO_MAP = {**KAN_TOPO_STAT, **KAN_TOPO_UNIQUE}
# 低分阈值：像素 top1 低于此值时，像素本身不可信 → 唯一拓扑可介入
KAN_ARB_LOW = 0.45
#  规则 D 的仲裁门限
#   起因：实战下家#8 与对家#13 都是**八萬**，像素 top1 就是 8m（0.753/ 0.727），
#   排第 2 才是 3m（0.635 / 0.601）。但 KAN_TOPO_MAP 把 (2,1)硬映射成 3m
#   ——「八」也是一撇一捺＝2 块 1 峰，与「三」的拓扑撞了。
#   结果规则 D 用 score=9.0 把**本来正确**的答案改成了错的。
#
#   门限依据（GT 上 23 张万子的实测分差）：
#     规则 D 唯一「强制改对」的那张（像素 top1=8m，强制 2m，对）分差 = **0.028**
#     规则 D 生效但无害（强制项本就等于像素 top1）的分差   = 0.076 ~ 0.159
#     实战两个错例的分差                             = **0.118 / 0.126**
#   → 取 0.06：保住 0.028 那张的修复，放行 0.118/0.126 给像素。
#   语义：**像素分得开就信像素，分不开才让拓扑仲裁。**
KAN_ARB_GAP = 0.06

CODE_BOX = (60, 17, 79, 36)
CODE_TAIL = (54, 17, 60, 36)
TPL_FACE_BOX = (3, 16, 77, 122)
#  牌河侧另设一套：牌河顶部 y<13 已经是干净纸面（自家#1：y=13 才开始有墨），
#   照模板切掉 16 行会把数字顶部横切掉 → "二萬"变"一萬"
RIVER_FACE_BOX = (3, 2, 77, 127)


def gray(p):
    return cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) if p.ndim == 3 else p


# ══════════════════════ 模板资产 ══════════════════════
_TEMPLATE_CACHE = {}


def load_templates(use_norm=False):
    """{牌名: pattern 掩膜}（紧裁到 bbox、无留白，墨=1 底=0）。

    首次调用需读 34 组 .pattern.png 并转 float，实测约 2.5s——这是单次
    识别 3s 里最大的固定开销。模板运行期不变，缓存后降到毫秒级。
    """
    cached = _TEMPLATE_CACHE.get(use_norm)
    if cached is not None:
        return dict(cached[0]), cached[1]
    with open(os.path.join(CLEAN, 'meta.json'), encoding='utf-8') as source:
        meta = json.load(source)
    T = {}
    for k in meta['tiles']:
        if use_norm:
            f = os.path.join(CLEAN, f'{k}.norm.npy')
            if not os.path.exists(f):
                continue
            if meta['tiles'][k].get('pattern_size'):
                T[k] = np.load(f)
        else:
            f = os.path.join(CLEAN, f'{k}.pattern.png')
            # meta 里列了 0z（赤字牌），但上游没有对应模板文件。
            # 先判存在再 imread，避免每局刷一条 cv2 解码失败警告。
            if not os.path.exists(f):
                continue
            pv = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if pv is None:
                continue
            T[k] = (255 - pv).astype(np.float32)/255.0
    _TEMPLATE_CACHE[use_norm] = (dict(T), meta)
    return T, meta


# ══════════════════════ 几何 ══════════════════════
def quad_ar(quad):
    """屏幕上「水平边 / 垂直边」的平均长度比。

     只看两组边的长度，不看顶点顺序 —— 循环平移不改变它。
      用于检测"这张牌相对该家方向躺倒了"（立直牌横放、吃/碰搭牌侧放）。
    """
    q = np.asarray(quad, np.float32)
    hs = vs = 0.0
    for i in range(4):
        d = q[(i+1) % 4] - q[i]
        hs += abs(float(d[0])); vs += abs(float(d[1]))
    return (hs/4)/max(vs/4, 1e-6)


def group_ar_median(items):
    ars = sorted(quad_ar(a['quad']) for a in items)
    return ars[len(ars)//2] if ars else 1.0


def face_of(im, quad, seat, extra_rot=0):
    """标注四边形 → 透视矫正到 80×129 → 按家旋向摆正 → 裁牌面 → 额外旋转。

     顺序是 **warp → 按家旋向 → take_face → extra_rot**。
       `take_face` 的裁剪框 (3,2,77,127) 按 125×74 定，
       先转再裁会在 rot=1/3 时横切一半（rot=2 尺寸不变，所以 bug 藏了很久）。
    """
    q = np.asarray(quad, np.float32)
    k = SEAT_SHIFT.get(seat, 0) % 4
    if k:
        q = q[[(i+k) % 4 for i in range(4)]]
    # 摆向判定统一在 orientation.py（横躺牌 warp 到横矩形，避免竖向拉伸）
    if ORI.is_landscape(seat, quad_ar(q)) and extra_rot in (0, 1):
        #  横放牌（立直）专用分支：
        #   原来一律 warp 到 80×129 竖矩形——横放 quad（≈87×71）被竖向拉伸
        #   1.8 倍、图案畸变，四个旋向全不中（自家立直 6p 错成 2s）。
        #   现在 warp 到 129×80 横矩形（方向一致、无形变）再 rot90 回正；
        #   extra_rot=0/1 分别对应两种躺向，由 read_quad 旋向循环选优。
        #  横放牌前进方向（屏右）紧贴下一张牌，切分 quad 的右边界常多含
        #   10~20% 邻牌内容——warp 后在横矩形上把**左侧** 20px 切掉
        #   （rot90(-1) 顺时针：左→上，正好对应 rot90 后的顶部污染带：
        #   不裁 ink bbox 被邻牌碎片拉横，6p 被压到 0.295 < 2s 0.384；
        #   裁后 6p 0.368 登顶——实测。上一张牌在屏左，故污染在左）。
        #   注意：改源 quad 内缩会改变透视映射（分数漂移），裁 warp 结果才等价。
        dst = np.array([[0, 0], [TPL_H, 0], [TPL_H, TPL_W], [0, TPL_W]], np.float32)
        w = cv2.warpPerspective(im, cv2.getPerspectiveTransform(q, dst),
                                (TPL_H, TPL_W), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        w = w[:, 20:]                            # 横矩形左侧去 20px（去邻牌）
        w = np.rot90(w, -1 if extra_rot == 0 else 1)
        r = SEAT_ROT.get(seat, 0)
        if r:
            w = np.rot90(w, r)
        return take_face(w)
    dst = np.array([[0, 0], [TPL_W, 0], [TPL_W, TPL_H], [0, TPL_H]], np.float32)
    w = cv2.warpPerspective(im, cv2.getPerspectiveTransform(q, dst),
                            (TPL_W, TPL_H), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    r = SEAT_ROT.get(seat, 0)
    if r:
        w = np.rot90(w, r)
    f = take_face(w)
    return np.ascontiguousarray(np.rot90(f, extra_rot)) if extra_rot else f


def take_face(rgb):
    """按经验框裁出牌面。 这里试过「按白纸 bbox 自适应裁剪」，已回滚 

    实测（GT 全自动端到端 67/69 → 66/69，实战 across 低分 12 → 13）：
      warp 后的 80×129 图里，白纸主域 bbox 已经**把牌面填满**
      （across#9 是 77×110@3,0，own#5 是 80×112@0,17），
      自适应裁剪等于按域边界再切一刀 → 引入 ±2px 的额外抖动。
    → 牌面在 warp 图里**本来就是对齐的**（warp 用的就是切分四边形），
      不需要再从像素里重新找边界。写死的经验框反而更稳。

    保留这个函数名与签名，是为了让 face_of 的调用点不必改。
    """
    x1, y1, x2, y2 = RIVER_FACE_BOX
    return np.ascontiguousarray(rgb[y1:y2, x1:x2])


# ══════════════════════ 牌面清理 ══════════════════════
def wipe_code(rgb):
    """牌河侧**没有**右上编号（用户纠正过），原样返回，函数只为接口统一。"""
    return rgb


def ink_of(rgb):
    """墨迹 = 绝对暗度 + 红墨 + 蓝墨 + 绿墨。归一化到 [0,1]，墨=1。

     绝不用"比周围暗多少"的相对阈值 —— 牌沿/阴影会全被算成墨。
     颜色必须参与：牌河里筒子的蓝圈、索子的绿棒、字牌的绿字是有彩色的，
      只判"暗"抓不到它们。
     但**颜色只用来判"是不是墨"，不参与后续比对** ——
      比对时用的是这张二值掩膜，颜色信息在这一步之后就丢掉了。
    """
    g = gray(rgb).astype(np.float32)
    r, g2, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    dark = np.clip((160.0 - g)/75.0, 0, 1)
    red   = np.clip((r - np.maximum(g2, b) - 18.0)/60.0, 0, 1)
    blue  = np.clip((b - np.maximum(r, g2) - 20.0)/70.0, 0, 1)
    green = np.clip((g2 - np.maximum(r, b) - 20.0)/70.0, 0, 1)
    return np.maximum(np.maximum(dark, red), np.maximum(blue, green))


def paper_body(rgb, close_k=5, seed_frac=0.30, min_seed=30):
    """白纸掩膜 → 灌水（标签图法）→ 逐域取外接框 = 牌面本体掩膜。

     三个踩过的坑 
    1. **不能用轮廓填充**。图案是白纸域里的洞，任何"把洞填实"都会让 body 在图案处为 0，
       `ink*body` 把图案全部清零（实测 across#38：ink 1998px → 剩 0）。正确：逐域取外接框。
    2. **种子不能只取中心窗口**。一筒的筒圈在牌面正中，中心 44×44 里只剩 11~22 个白纸像素
       → paper_body 返回 None。修法：中心白纸不足 min_seed 时改取"离中心最近的大域"。
    3. **不能改用"掩膜 ∪ 补洞"来替代外接框**（试过，无效）：
       across#32 的白纸域是 74×125 的大域，域外那一条桌布阴影（lum≈186 < 阈值 198）
       本来被掩膜正确排除，但外接框一填就把它算成牌面 → ink 绝对暗度判据把它变成"墨"
       → 图案 bbox 右下角凭空多出一块 22px 墨迹。
       改用逐域 floodFill 补洞后结果与外接框**完全相同**（阴影那一侧本来就与纸不连通，
       补洞不会填它）→ 等于没改。真正该做的是规则A 的边界碎块剔除。
    """
    g = gray(rgb).astype(np.float32)
    sat = (rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16))
    m = ((g > 0.90*float(np.percentile(g, 98))) & (sat < 48)).astype(np.uint8)
    H, W = m.shape
    if m.sum() < 50:
        return None
    m2 = (cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
          if close_k >= 3 else m)
    n, lab = cv2.connectedComponents(m2, 8)
    if n <= 1:
        return None
    sy, sx = H//2, W//2
    #  先把「够大的域」筛出来（占域面积 ≥8% 或绝对面积 ≥200），再选种子 
    #   坑：原来只看「中心窗口里有几个白纸像素」。当牌面图案把白纸切成很多小块时
    #   （自家#14「一筒」：圆环+内环把白纸打成 17 个域），
    #   中心 45×45 里有 42 个白纸像素 ≥min_seed=30 → 走"中心种子"分支，
    #   但那 42 个像素**散落在 17 个小域里**，每个域的 bbox 面积都 < 200
    #   → 全被 continue 掉 → 返回 None → 整张牌识别失败。
    #   对照：那个大域本身面积 4000、bbox 覆盖 x[0,73] y[9,124]，几乎是整张牌面。
    areas = sorted(((int((lab == i).sum()), i) for i in range(1, n)), reverse=True)
    tot_area = float(sum(a for a, _ in areas)) or 1.0
    big = [(a, i) for a, i in areas
           if a >= 200 or a/tot_area >= 0.08]
    if not big:
        return None
    # 种子 = 与中心「最近」的3 个够大的域（按质心距离）。
    #  用质心距离而不是"中心窗口像素数"：图案把白纸切碎时后者会失效，
    #   但牌面主体那个大域的质心必然离中心最近。
    cands = []
    for a, i in big[:6]:
        ys_i, xs_i = np.nonzero(lab == i)
        cands.append((float((ys_i.mean()-sy)**2 + (xs_i.mean()-sx)**2), i))
    cands.sort()
    keep = sorted({i for _, i in cands[:3]})
    out = np.zeros_like(m)
    for i in keep:
        ys_i, xs_i = np.nonzero(lab == i)
        if len(ys_i) == 0:
            continue
        bx1, bx2 = int(xs_i.min()), int(xs_i.max())+1
        by1, by2 = int(ys_i.min()), int(ys_i.max())+1
        if (bx2-bx1)*(by2-by1) < 200:
            continue
        out[by1:by2, bx1:bx2] = 1
    return out if out.sum() >= 60 else None


# ══════════════════════ 规则A：边界碎块剔除 ══════════════════════
def drop_border_specks(ink, max_frac=SPECK_MAX_FRAC, border=SPECK_BORDER):
    """删掉「面积占比 < max_frac」且「贴着裁块边界」的连通块。

    治 across#32 的 6z：bbox 右下角混进一块桌布阴影。
    两个条件缺一不可 —— 详见模块 docstring 的实测记录。
    """
    H, W = ink.shape[:2]
    b = (ink > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if n <= 1:
        return ink
    tot = int(b.sum())
    out = ink.copy()
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if area/max(tot, 1) >= max_frac:
            continue
        if x <= border or y <= border or x+w >= W-border or y+h >= H-border:
            out[lab == i] = 0.0
    return out


def extract_pattern(face, cov_thr=0.06, specks=True):
    """牌面 → 实质性图案（紧裁到 bbox、无留白）。返回 (pattern, box) 或 ('BLANK',None)。"""
    body = paper_body(face)
    if body is None:
        return None, None
    ink = ink_of(face) * body
    if specks:
        ink = drop_border_specks(ink)
    bs = float((body > 0.5).sum())
    ink_ratio = float((ink > 0.5).sum())/max(bs, 1.0)
    ys, xs = np.nonzero(ink > 0.5)
    #  白板判定：三个条件都指向"牌面本来空白"，一律走BLANK 
    #   ① len(ys) < 6      —— 几乎没有墨
    #   ② ink_ratio 过低   —— 墨迹占牌面本体不到 BLANK_INK_MAX（4.5%）
    #   ③ bbox 退化        —— 有墨但撑不起一个图案（宽或高 < 4px）
    #   实战两个反例：上家#10 墨迹 104px/9250 = 0.0112、bbox 48×4（贴牌底）；
    #                   自家#2   墨迹  10px/7992 = 0.0013、bbox 4×1（贴牌顶）
    #   → 都不该参与模板竞争，直接判白板。
    #   绝不能返回全零图去参与竞争 —— 白板 vs 白板 IoU=1，会把 5z 顶到第一。
    if len(ys) < 6 or ink_ratio < BLANK_INK_MAX:
        return 'BLANK', None
    y1, y2, x1, x2 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
    if y2-y1 < 4 or x2-x1 < 4:
        return 'BLANK', None
    sub = ink[y1:y2, x1:x2]
    for ax in (0, 1):        # 剥边框（整行/整列几乎全墨）
        if sub.shape[ax] < 6:
            continue
        bad = (sub > 0.5).mean(axis=1-ax) >= 0.88
        a = 0
        while a < sub.shape[ax] and bad[a]:
            a += 1
        b = sub.shape[ax]-1
        while b > a and bad[b]:
            b -= 1
        if b-a >= 4:
            sub = sub[a:b+1] if ax == 0 else sub[:, a:b+1]
    # 行/列覆盖率收紧。 必须 ceil —— int() 截断少 1 像素，
    #   框会延伸到最底边的噪点，归一化后整个模板就废了（0.79→0.19）。
    th = max(1, int(np.ceil(max(sub.shape)*cov_thr)))
    rr = np.nonzero((sub > 0.5).sum(axis=1) >= th)[0]
    cc = np.nonzero((sub > 0.5).sum(axis=0) >= th)[0]
    if len(rr) >= 2 and len(cc) >= 2:
        sub = sub[rr[0]:rr[-1]+1, cc[0]:cc[-1]+1]
    if min(sub.shape) < 4:
        return None, None
    return sub.astype(np.float32), (x1, y1, x2, y2)


# ══════════════════════ 规则C：组合评分 ══════════════════════
def fit_to_box(src, box_w, box_h):
    """把 src **各向异性拉伸**到 (box_w × box_h)，不补白、不居中。

    用户原话："把待比对图案拉伸到跟 pattern 一样大小再去比较"、
              "对模板标准化后图案上下左右都没有留白，那么待比对的图案也应该
               做这种处理，再拉伸到同一横纵维度去对比"。

     不居中补白：补白会让"图案在画布里的位置"随长宽比变化（实测 own#4 位置对不上）。
      两边都是"紧裁到 bbox、无留白"的图案，直接拉成同一 (w,h) 逐像素比。
     也不用"等比缩放 + 补白"：牌河与手牌是两套渲染，实测 across#38 的"東"
      牌河 75×102 vs 模板 55×94 → 宽差 1.36×、高差 1.09×，等比修不了。
    """
    h, w = src.shape[:2]
    if h < 1 or w < 1 or box_w < 1 or box_h < 1:
        return None
    sx, sy = box_w/float(w), box_h/float(h)
    # 最近的采样：图案是二值墨迹掩膜，插值方式对匹配几乎无影响，
    # 但每帧要做几千次，INTER_NEAREST 明显更快
    interp = cv2.INTER_NEAREST
    return cv2.resize(src, (int(box_w), int(box_h)),
                      interpolation=interp).astype(np.float32)


def iou(a, b, thr=0.5):
    ab, bb = a > thr, b > thr
    u = (ab | bb).sum()
    return float((ab & bb).sum()/u) if u else 0.0


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt(float((a*a).sum())*float((b*b).sum()))
    return float((a*b).sum()/d) if d > 0 else 0.0


def match_one(pattern, tpl_pattern):
    q = fit_to_box(pattern, tpl_pattern.shape[1], tpl_pattern.shape[0])
    if q is None or q.shape != tpl_pattern.shape:
        return 0.0, 0.0
    return iou(q, tpl_pattern), ncc(q, tpl_pattern)


def rank(pattern, T, alpha=ALPHA):
    """返回按 α·IoU+(1−α)·NCC 降序的 (score, name, IoU, NCC) 列表。"""
    sc = []
    for k, tp in T.items():
        if tp is None:
            continue
        i_, n_ = match_one(pattern, tp)
        sc.append((alpha*i_ + (1-alpha)*n_, k, i_, n_))
    sc.sort(reverse=True)
    return sc


# ══════════════════════ 规则D：万子拓扑仲裁 ══════════════════════
def kan_top(pat, min_share=KAN_TOP_SHARE, max_frac=KAN_TOP_MAXFRAC):
    """「萬」的顶边行号 —— 万子牌面的「萬」必然是最大连通域且在下方。
    返回 None 表示"这个牌面没有明确的大块"（筒子/索子/字牌）。"""
    b = (pat > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if n <= 1:
        return None
    tot = int(b.sum())
    bi = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[bi, cv2.CC_STAT_AREA]/max(tot, 1) < min_share:
        return None
    y = int(st[bi, cv2.CC_STAT_TOP])
    if y <= 0 or y > pat.shape[0]*max_frac:
        return None
    return y


def digit_topology(pat, min_area=KAN_MIN_AREA, peak_thr=KAN_PEAK_THR):
    """万子数字区的拓扑特征 → (横向块数, 纵向峰数)。不是万子返回 None。

    nblk：数字区的横向连通块数（先剔掉最靠下那块 —— 那是「萬」顶部的粘连）
    npk ：数字区纵向投影的峰数（"二"= 两条横 → 2 峰）

     为什么用这个而不是像素形状 
    万子的判别信息全在数字区，而"萬"在 2m/3m/8m 里**完全相同**，
    且只占墨迹 72.5%。查 own#6（2m 被判成 8m）时量到：

      只看数字区（上 32 行）：vs 2m IoU 0.639 / vs 8m IoU 0.370  ← 分得开
      看全图：              vs 2m IoU 0.617 / vs 8m IoU 0.650  ← 8m 赢

    → 判别信号被"萬"稀释。各向异性拉伸又把牌河的两条横压到 8m 那一行附近，
      像素层面两者靠近。改用"有几条横"这种**拓扑量**就绕开了压缩比。

     但**不能跨渲染比拓扑** 
    牌河与手牌是两套渲染，连通域数系统性地不同 —— 全量 22 张万子
    "块数与模板一致"的命中是 0/22（牌河 1m/4m/5m/6m/7m/9m 全是 1 块，
     模板是 2 块）。所以 KAN_TOPO_MAP 的取值只来自**牌河侧自洽**的统计。
    """
    y = kan_top(pat)
    if y is None:
        return None
    num = pat[:y]
    b = (num > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    blocks = [(int(st[i, cv2.CC_STAT_TOP]), int(st[i, cv2.CC_STAT_HEIGHT]),
               int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n)
              if int(st[i, cv2.CC_STAT_AREA]) >= min_area]
    if not blocks:
        return None
    ymax = max(t[0]+t[1] for t in blocks)
    nblk = len([t for t in blocks if t[0]+t[1] < ymax-2]) or len(blocks)
    prof = b.sum(axis=1).astype(float)[:y]
    if len(prof) == 0 or prof.max() <= 0:
        return None
    thr = max(1.0, float(prof.max())*peak_thr)
    npk, i, on = 0, 0, prof > thr
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            npk += 1
            i = j
        else:
            i += 1
    return nblk, npk


def arbitrate_kan(pat, sc_top5, gap=None):
    """万子拓扑仲裁：像素比对的 top5 里若有该万子，且拓扑指向它，就采纳拓扑。

     必须限定"像素 top5 里有该万子"才允许仲裁 ——
      不加这条约束时，`kan_top` 会把筒子的圆环、索子的分节也当成"萬"，
      3p/4p/9p/3s 全被误判成 2m/3m，全量从 75掉到 72。
      万子必须**先判花色**，拓扑只在万子内部用来消歧。

     必须限定「像素 top1−top2 分差 < KAN_ARB_GAP」才允许**改变**结果 
      见 KAN_ARB_GAP 处的实测数据。简言之：
        像素分得开 → 信像素；像素分不开 → 让拓扑仲裁。
      没有这条，实战两个「八萬」会被拓扑强改成「三萬」（分差 0.118/0.126）。
    """
    if not any(k.endswith('m') for k in sc_top5):
        return None
    tp = digit_topology(pat)
    if tp is None:
        return None
    forced = KAN_TOPO_MAP.get(tp)
    if forced is None or forced not in sc_top5:
        # 拓扑指向的牌连像素 top5 都没有 → 不可信，让位给像素
        return None
    # 像素已经明显分开了（gap 大）→ 拓扑没有用武之地，别插手
    if gap is None:
        # 低分场景（调用方判定 top1 < KAN_ARB_LOW）：只有模板唯一拓扑能改结果。
        # (1,2)/(2,1) 是 5 个数字共享的统计映射，低分时强行仲裁会把
        # 5m/1m 之类改成 2m/3m——这是 那批误伤的同类风险。
        return forced if tp in KAN_TOPO_UNIQUE else None
    if gap > KAN_ARB_GAP:
        return None
    return forced


# ══════════════════════ 规则B：几何离群 → 换向 ══════════════════════
RETRY_THR = 0.50         # 首旋向得分低于此 → 补试其余 3 个旋向
RETRY_MARGIN = 0.15      # 重试结果须高出首旋向这么多才采纳（防低分蒙对的牌被翻错）

def _eval_rotation(im, quad, seat, r, T, alpha, specks, use_kan, ar,
                   skip_blank):
    """单个旋向的评估。

    返回 (score, name, top3, r, iou, ncc, kan_hit) / None / 'BLANK'。
    skip_blank=True（重试分支）：白板返回 'BLANK' 标记，调用方跳过——
      重试不该用白板翻掉首试的结果；
    skip_blank=False（首试）：白板直接返回 5z 结果（任何旋向都成立）。
    """
    pat, _ = extract_pattern(face_of(im, quad, seat, extra_rot=r), specks=specks)
    if pat is None:
        return None
    if isinstance(pat, str):
        return 'BLANK' if skip_blank else (1.0, '5z', ['5z'], r, 0.0, 0.0, None)
    sc = rank(pat, T, alpha)
    if not sc:
        return None
    i_, n_ = sc[0][2], sc[0][3]
    name, score = sc[0][1], sc[0][0]
    kan_hit = None
    # 规则 D 只在名义摆向且非横 warp 时可信（orientation 模块文档）
    if use_kan and ORI.topology_allowed(seat, r, ar):
        top5 = [x[1] for x in sc[:5]]
        gap = sc[0][0]-sc[1][0] if len(sc) > 1 else 9.9
        # 像素低分（切分偏差/图案缩水）→ 传 None：只允许模板唯一拓扑救援
        f = arbitrate_kan(pat, top5,
                          gap if sc[0][0] >= KAN_ARB_LOW else None)
        if f is not None:
            name, score, kan_hit = f, 9.0, f
    return (score, name, [name]+[x[1] for x in sc[:3] if x[1] != name], r,
            i_, n_, kan_hit)


def read_quad(im, quad, seat, group_median_ar, T, geo_tol=GEO_TOL,
              alpha=ALPHA, specks=True, use_kan=True):
    """识别单张牌。返回 (name, score, top3, extra_rot, IoU, NCC, kan_hit)。

     先按该家基准旋向识别；只有当这张牌的四边形"h/v"明显偏离同组中位数时
      （立直牌横放、吃/碰搭牌侧放 —— 相对该家躺倒 90°），才试遍 4 个旋向取最高分。
     非离群牌只跑 1 个旋向 → 不会因为换向连累已经认对的牌。
     补充：首旋向得分 < RETRY_THR 时也补试 4 个旋向。
      原因：侧家立直牌的横放发生在「牌面朝向」上，切分给出的四边形
      仍是名义形状，h/v 不离群 → 几何判据失效（实机下家#13 一索读成 4p）。
      低分本身就是姿态异常的信号，换向重试是确定性机制。
     重试结果必须比首旋向高 RETRY_MARGIN 才采纳：
      GT 右家#0 真值 1s 首旋向只蒙对 0.25，重试 2z 0.261 险胜就会翻错。
    """
    ar = quad_ar(quad)
    rel = abs(ar - group_median_ar)/max(group_median_ar, 1e-6)
    # 旋向集合统一由 orientation.rotations_for 决定（own 专属 ar 判据依据
    # 见该模块：own 基线 1.05 / across 1.24 / 侧家 2.05，绝对阈值只对 own 成立）
    rots = ORI.rotations_for(seat, ar, rel)
    best = None
    for r in rots:
        res = _eval_rotation(im, quad, seat, r, T, alpha, specks, use_kan, ar,
                             skip_blank=False)
        if res is None or res == 'BLANK':
            continue
        if best is None or res[0] > best[0]:
            best = res
    #  低分补试其余旋向；结果须高出首旋向 RETRY_MARGIN 才采纳
    if best is not None and best[0] < RETRY_THR and len(rots) < 4:
        first = best
        retry_best = None
        for r in (0, 1, 2, 3):
            if r in rots:
                continue
            res = _eval_rotation(im, quad, seat, r, T, alpha, specks, use_kan, ar,
                                 skip_blank=True)
            if res is None or res == 'BLANK':
                continue
            if retry_best is None or res[0] > retry_best[0]:
                retry_best = res
        if retry_best is not None and retry_best[0] > first[0] + RETRY_MARGIN:
            best = retry_best
    if best is None:
        return None, 0.0, [], 0, 0.0, 0.0, None
    score, name, top3, r, i_, n_, kan_hit = best
    return name, score, top3, r, i_, n_, kan_hit


def read_group(im, items, seat, T, alpha=ALPHA, specks=True, geo_tol=GEO_TOL,
               use_kan=True):
    """识别一整组（一家的一手牌 / 一组副露）。返回逐张结果列表。"""
    med = group_ar_median(items)
    out = []
    for a in items:
        r = read_quad(im, a['quad'], seat, med, T, geo_tol, alpha, specks, use_kan)
        out.append(dict(id=a['id'], truth=a['name'], name=r[0], score=r[1],
                        top3=r[2], rot=r[3], iou=r[4], ncc=r[5], kan_hit=r[6]))
    return out
