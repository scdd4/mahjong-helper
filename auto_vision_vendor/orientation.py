"""摆向规则中心。

## 为什么有这个文件

识别的核心难点是「牌在屏幕上的摆向」与「该家视角的名义摆向」不一致：
立直牌横放、侧家的牌整体横躺、副露区的搭牌斜放。历史上这类判断散落在
recognize.face_of / recognize.read_quad / meld._recognize_tile
三处，各写各的 ar 阈值（1.18 / 1.15 / 1.5），同类守卫重复 4 份——
改一处容易漏另外两处。本模块把它们集中成**单一事实源**：

* 阈值常量（全部带 GT 实测依据）
* `is_landscape(seat, ar)`      —— 该 quad 是否横躺（warp 取横矩形）
* `topology_allowed(seat, rot, ar)` —— 规则 D（万子拓扑仲裁）是否可信
* `rotations_for(seat, ar, rel)`   —— 该牌要尝试的旋向集合

## 两条通用原则（，不是单点补丁）

1. **非名义摆向的图案拓扑不可比**
   拓扑量（数字区横向块数/纵向峰数）只在「正立、无拉伸」的渲染上成立。
   横躺/侧躺 warp 出来的图案是另一套渲染 → 规则 D 会误伤：
   - 上家立直 3p 被强改成 2m（score=9.0 直接采信）
   - 对家副露 5s 被强改成 2m
   统一守卫：`topology_allowed = rot == 0 and not is_landscape(...)`。

2. **横躺牌必须 warp 到横矩形，否则竖向拉伸**
   侧家/副露的横躺牌（ar 2.0+）若 warp 到 80×129 竖矩形，图案被竖向
   拉伸 ~3 倍：字牌图案简单还能认，5p 的五个筒直接认不出
   （实测：下家 5p → 5z）。自家立直牌已验证（横矩形 + rot90 后
   5p→6p 修复），走 landscape；**侧家/副露的横路径暂锁**——横路径要叠加
   该家的 SEAT_ROT，两个躺向 × SEAT_ROT 的组合矩阵尚未在 GT 上校准
   （试开侧家横 warp → 整家 16 张全变 5z）。校准后只需把
   `SIDE_LANDSCAPE_WARP` 打开，规则调用点不用动。
"""

# ── 阈值（GT 实测依据）────────────────────────────────────────────
# own 正常牌 ar 上限：GT 实测 max 1.126（竖牌露出上部，ar≈1.05）
AR_OWN_ERECT_MAX = 1.15
# own 立直横放起始：GT 实测 1.225（与正常牌之间有空档 1.126~1.2）
AR_OWN_LANDSCAPE = 1.18
# 侧家横躺起始：left/right 的牌在屏幕上整体横躺，GT 实测 2.0~2.2
AR_SIDE_LANDSCAPE = 1.5
# 相对同组中位的离群容差（立直牌/搭牌）。GT 校准过 0.15~0.30 一致
GEO_TOL = 0.25

#  侧家 / 副露的横躺牌横 warp 开关（见模块文档第 2 条）。
#   关闭时这类牌走竖路径（有拉伸，复杂图案会错）；打开前必须先在 GT 上
#   枚举 (warp 方向 × rot90 方向 × SEAT_ROT) 定出组合矩阵。
SIDE_LANDSCAPE_WARP = False


def is_landscape(seat, ar: float) -> bool:
    """该 quad 是否横躺——warp 目标应取横矩形（129×80）再 rot90 回正。"""
    if seat == "own":
        return ar > AR_OWN_LANDSCAPE
    if seat in ("left", "right"):
        return SIDE_LANDSCAPE_WARP and ar > AR_SIDE_LANDSCAPE
    # across：正常牌 ar≈1.24 是 45° 侧视（非横躺），保持竖路径
    return False


def is_side_landscape(seat, ar: float) -> bool:
    """侧家横躺（不论开关）——用于旋向集合判断。"""
    return seat in ("left", "right") and ar > AR_SIDE_LANDSCAPE


def topology_allowed(seat, rot: int, ar: float) -> bool:
    """规则 D（万子拓扑仲裁）是否可用。见模块文档第 1 条。

    = 名义摆向（rot==0）且不走横 warp（is_landscape）。
    横 warp 的开关在 SIDE_LANDSCAPE_WARP——打开侧家横路径后，
    那些牌的拓扑自动失效，调用点无需改。
    """
    return rot == 0 and not is_landscape(seat, ar)


def rotations_for(seat, ar: float, rel: float):
    """该牌要尝试的旋向集合。

    * 侧家横躺：横路径开时试两个躺向；关时只走名义旋向（现状）
    * 离群（立直/搭牌）：全 4 旋向
    * 其余：名义旋向
    """
    if is_side_landscape(seat, ar) and SIDE_LANDSCAPE_WARP:
        return (0, 1)
    if rel > GEO_TOL or (seat == "own" and ar > AR_OWN_ERECT_MAX):
        return (0, 1, 2, 3)
    return (0,)
