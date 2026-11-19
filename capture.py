# -*- coding: utf-8 -*-
"""
雀魂手牌截图识别引擎
====================
流程：截屏 -> 按标定框把手牌行切成 N 格 -> 归一化特征 -> 模板匹配 / 聚类

用法（由 server.mjs 调用，也可自己跑）：
  python capture.py '{"mode":"probe"}'
  python capture.py '{"mode":"scan","box":[x,y,w,h],"n":14}'
    python capture.py '{"mode":"learn","scanId":"扫描返回的ID","assign":{"0":12,"1":5}}'
  python capture.py '{"mode":"scan","box":[...],"n":14,"image":"test.png"}'   # 用图片代替截屏（调试）

输出：单行 JSON 到 stdout。
"""
import sys, os, json, base64, io, math, uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
TPL_DIR = os.environ.get("MJ_TPL_DIR") or os.path.join(ROOT, "templates")
CELL_DIR = os.environ.get("MJ_CELL_DIR") or os.path.join(ROOT, "cells")
SHOT = os.path.join(ROOT, "latest.png")
CALIB = os.environ.get("MJ_CALIB") or os.path.join(ROOT, "calib.json")

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- 截屏
def dpi_aware():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def grab():
    dpi_aware()
    from PIL import ImageGrab
    return ImageGrab.grab(all_screens=True)


def load_image(arg):
    p = arg.get("image")
    if p is not None:
        if not isinstance(p, str) or not os.path.isfile(p):
            raise ValueError("指定的图片文件不存在；未调用屏幕截图")
        with Image.open(p) as source:
            return source.convert("RGB")
    img = grab().convert("RGB")
    if os.environ.get("MJ_SAVE_SHOT") == "1":
        img.save(SHOT)
    return img


def tile_count(value):
    if type(value) is not int or not 1 <= value <= 14:
        raise ValueError("n 必须是 1 到 14 的整数")
    return value


def valid_box(value, name="box", image_size=None, cells=1):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(name + " 必须是 [x,y,w,h]")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError(name + " 坐标必须是有限数字")
    x, y, w, h = value
    if x < 0 or y < 0 or w < cells or h < 2:
        raise ValueError(name + " 超出屏幕或标定框太小")
    if image_size and (x + w > image_size[0] or y + h > image_size[1]):
        raise ValueError(name + " 超出当前截图；请重新标定位置")
    return list(value)


# ---------------------------------------------------------------- 特征
PAD = 0.14          # 每格四周多留 14% 余量，供位移搜索用
WIN_W, WIN_H = 28, 40
SMALL_W, SMALL_H = WIN_W + 8, WIN_H + 8   # 36 x 48
R = 3                                     # 位移范围 ±3（小图坐标，约等于原图 ±12%）


def feat_from_image(im):
    """单张紧裁图 -> 28x40 零均值单位向量（仅用于加载模板）"""
    g = im.convert("L").resize((WIN_W, WIN_H), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32).ravel()
    a = a - a.mean()
    n = float(np.sqrt((a * a).sum()))
    return a / (n if n > 1e-6 else 1.0)


def padded(im, rect):
    x, y, w, h = rect
    px, py = w * PAD, h * PAD
    return im.crop((int(round(x - px)), int(round(y - py)),
                    int(round(x + w + px)), int(round(y + h + py))))


def normalize_aspect(im, aspect=None):
    """裁成统一长宽比：手牌和宝牌指示牌的牌面比例不一样，先统一再比"""
    aspect = aspect or (WIN_W / float(WIN_H))
    w, h = im.size
    if w <= 0 or h <= 0:
        return im
    a = w / float(h)
    if a > aspect:
        nw = max(1, int(round(h * aspect)))
        x = (w - nw) // 2
        return im.crop((x, 0, x + nw, h))
    nh = max(1, int(round(w / aspect)))
    y = (h - nh) // 2
    return im.crop((0, y, w, y + nh))


def windows(big):
    """把带余量的格子缩到 36x48，再取 49 个位移窗口；返回 (49, 1120)"""
    small = big.convert("L").resize((SMALL_W, SMALL_H), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    out = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            win = a[4 + dy:4 + dy + WIN_H, 4 + dx:4 + dx + WIN_W]
            v = win.ravel() - win.mean()
            n = float(np.sqrt((v * v).sum()))
            out.append(v / (n if n > 1e-6 else 1.0))
    return np.stack(out)


def center_vec(W):
    return W[(R * (2 * R + 1)) + R]      # dx=0,dy=0 那一行


def b64(im, fmt="PNG"):
    buf = io.BytesIO()
    im.save(buf, fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_templates():
    os.makedirs(TPL_DIR, exist_ok=True)
    out = {}
    for f in os.listdir(TPL_DIR):
        if not f.endswith((".npy", ".png")):
            continue
        try:
            label = int(os.path.splitext(f)[0])
            if not 0 <= label < 34:
                continue
            if f.endswith(".npy"):
                v = np.load(os.path.join(TPL_DIR, f), allow_pickle=False).astype(np.float32)
            else:
                with Image.open(os.path.join(TPL_DIR, f)) as im:
                    v = feat_from_image(im.convert("RGB"))
            norm = float(np.linalg.norm(v))
            if v.shape != (WIN_W * WIN_H,) or not np.isfinite(v).all() or not math.isfinite(norm) or norm < 1e-6:
                continue
            out[label] = v / norm
        except (ValueError, TypeError, OSError, EOFError):
            continue
    return out


# ---------------------------------------------------------------- 切格
def cell_rects(box, n):
    x, y, w, h = [float(v) for v in box]
    cw = w / n
    rects = []
    for i in range(n):
        l = x + i * cw + cw * 0.10
        r = x + (i + 1) * cw - cw * 0.10
        t = y + h * 0.08
        b = y + h * 0.92
        rects.append([int(round(l)), int(round(t)), int(round(r - l)), int(round(b - t))])
    return rects


def scan(img, box, n, save_cells=False):
    rects = cell_rects(box, n)
    cells = []
    for i, (x, y, w, h) in enumerate(rects):
        crop = img.crop((x, y, x + w, y + h))
        big = padded(img, rects[i])
        if save_cells:
            os.makedirs(CELL_DIR, exist_ok=True)
            big.save(os.path.join(CELL_DIR, "%d.png" % i))   # 存带余量的图，学习时用同一套管线
        cells.append({"i": i, "rect": rects[i], "crop": crop, "W": windows(big)})
    return cells


def cluster_cells(cells, thr=0.96):
    """把长得一样的格子归为一簇（第一次用没有模板时，只需人工标几种牌）"""
    clusters = []
    for c in cells:
        v = center_vec(c["W"])
        placed = False
        for cl in clusters:
            if float(np.dot(v, cl["vec"])) >= thr:
                cl["cells"].append(c["i"])
                placed = True
                break
        if not placed:
            clusters.append({"id": len(clusters), "vec": v, "crop": c["crop"], "cells": [c["i"]]})
    return clusters


def classify(cells, tpl, thr=0.85, margin_min=0.02):
    labels = sorted(tpl.keys())
    T = np.stack([tpl[k] for k in labels]) if labels else None
    out = []
    for c in cells:
        if T is None:
            out.append({"i": c["i"], "label": None, "score": 0, "margin": 0,
                        "rect": c["rect"], "crop": b64(c["crop"])})
            continue
        S = c["W"] @ T.T                       # (49, K)  位移窗口 × 模板
        per_tpl = S.max(axis=0)                # 每个模板在所有位移里的最好得分
        best_k = int(per_tpl.argmax())
        best_score = float(per_tpl[best_k])
        srt = np.sort(per_tpl)
        second = float(srt[-2]) if len(srt) > 1 else -1.0
        lab = labels[best_k]
        ok = best_score >= thr and (best_score - second) >= margin_min
        out.append({
            "i": c["i"],
            "label": lab if ok else None,
            "score": round(best_score, 4),
            "margin": round(best_score - second, 4),
            "rect": c["rect"],
            "crop": b64(c["crop"]),
        })
    return out


# ---------------------------------------------------------------- 舍牌区
DISC_COLS, DISC_ROWS = 6, 4


def _tile_cells(crop, cols, rows, pad=0.10):
    w, h = crop.size
    cw, ch = w / float(cols), h / float(rows)
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * cw + cw * pad))
            x1 = int(round((c + 1) * cw - cw * pad))
            y0 = int(round(r * ch + ch * pad))
            y1 = int(round((r + 1) * ch - ch * pad))
            if x1 - x0 < 8 or y1 - y0 < 8:
                out.append(None)
                continue
            out.append(crop.crop((x0, y0, x1, y1)))
    return out


def _batch_scores(crops, tpl):
    """把一堆小格一次性做矩阵乘，避免逐格循环"""
    labels = sorted(tpl.keys())
    if not labels:
        return None, None
    T = np.stack([tpl[k] for k in labels])
    Ws = []
    for im in crops:
        Ws.append(windows_small(im)[4])
    W = np.stack(Ws)                 # (N, 1120)
    S = W @ T.T                      # (N, K)
    return labels, S


def windows_small(im):
    """舍牌格子小，用 ±1 的 9 个窗口就够了"""
    big = im
    small = big.convert("L").resize((WIN_W + 2, WIN_H + 2), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            win = a[1 + dy:1 + dy + WIN_H, 1 + dx:1 + dx + WIN_W]
            v = win.ravel() - win.mean()
            n = float(np.sqrt((v * v).sum()))
            out.append(v / (n if n > 1e-6 else 1.0))
    return np.stack(out)


def scan_discards(img, boxes, tpl, thr=0.82):
    """boxes: {side: [x,y,w,h]}；左右两家的牌是横放的，自动试 0/90/270 度"""
    result = {}
    if not tpl:
        return result
    for side, box in boxes.items():
        if not box or len(box) != 4:
            continue
        bx, by, bw, bh = [float(v) for v in box]
        best = None
        for rot in (0, 90, 270):
            crop = img.crop((int(round(bx)), int(round(by)),
                             int(round(bx + bw)), int(round(by + bh))))
            if rot:
                crop = crop.rotate(rot, expand=True)
            crops = [c for c in _tile_cells(crop, DISC_COLS, DISC_ROWS) if c is not None]
            if not crops:
                continue
            labels, S = _batch_scores(crops, tpl)
            per_tpl = S.max(axis=1)                 # 每格的最好分
            idx = S.argmax(axis=1)
            second = np.partition(S, -2, axis=1)[:, -2] if len(labels) > 1 else np.full(len(crops), -1.0)
            hit = [(labels[i], float(s)) for i, s, alt in zip(idx, per_tpl, second)
                   if s >= thr and s - alt >= 0.02]
            score = len(hit) if len(hit) >= 3 else 0   # 少于 3 张就当作没认出来
            if best is None or score > best["score"]:
                best = {"score": score, "rot": rot, "tiles": [h[0] for h in hit],
                        "scores": sorted([h[1] for h in hit], reverse=True)[:8]}
        if best and best["score"] >= 3:
            result[side] = {"rot": best["rot"], "tiles": best["tiles"],
                            "count": len(best["tiles"])}
    return result


# ---------------------------------------------------------------- 手牌文本
NAMES = []
for s in ["万", "筒", "索"]:
    for i in range(1, 10):
        NAMES.append("%d%s" % (i, s))
for h in ["东", "南", "西", "北", "白", "发", "中"]:
    NAMES.append(h)


def hand_text(labels):
    """labels: 按位置排好的 34 进制下标，返回 m/p/s/z 记法"""
    cnt = [0] * 34
    for v in labels:
        if v is None:
            return None
        cnt[v] += 1
    out = []
    for si, k in enumerate(["m", "p", "s"]):
        buf = "".join(str(i + 1) * cnt[si * 9 + i] for i in range(9))
        if buf:
            out.append(buf + k)
    buf = "".join(str(i + 1) * cnt[27 + i] for i in range(7))
    if buf:
        out.append(buf + "z")
    return "".join(out)


# ---------------------------------------------------------------- 主流程
def dispatch(arg):
    if not isinstance(arg, dict):
        raise ValueError("请求必须是 JSON 对象")
    mode = arg.get("mode", "scan")

    if mode == "probe":
        img = load_image(arg)
        disp = img.copy()
        disp.thumbnail((1280, 1280), Image.BILINEAR)
        calib = None
        if os.path.exists(CALIB):
            try:
                calib = json.load(open(CALIB, "r", encoding="utf-8"))
            except Exception:
                calib = None
        return {
            "ok": True, "mode": "probe",
            "w": img.width, "h": img.height,
            "dispW": disp.width, "dispH": disp.height,
            "scale": disp.width / float(img.width),
            "image": b64(disp),
            "templates": len(load_templates()),
            "calib": calib,
        }

    if mode == "scan":
        n = tile_count(arg.get("n", 14))
        saved = {}
        if os.path.exists(CALIB):
            try:
                with open(CALIB, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if not isinstance(saved, dict):
                    saved = {}
            except Exception:
                saved = {}
        box = arg.get("box") if arg.get("box") is not None else saved.get("box")
        dora_box = arg.get("doraBox") if arg.get("doraBox") is not None else saved.get("doraBox")
        disc_boxes = arg.get("discardBoxes") if arg.get("discardBoxes") is not None else saved.get("discardBoxes", {})
        if box is None:
            return {"ok": False, "error": "还没有标定手牌位置", "needCalib": True}
        box = valid_box(box, cells=n)
        if dora_box is not None:
            dora_box = valid_box(dora_box, "doraBox", cells=5)
        if not isinstance(disc_boxes, dict) or any(k not in ("right", "top", "left") for k in disc_boxes):
            raise ValueError("discardBoxes 必须只包含 right/top/left")
        disc_boxes = {k: valid_box(v, "discardBoxes." + k) for k, v in disc_boxes.items()}
        img = load_image(arg)
        valid_box(box, image_size=img.size, cells=n)
        if dora_box is not None:
            valid_box(dora_box, "doraBox", img.size, cells=5)
        for side, value in disc_boxes.items():
            valid_box(value, "discardBoxes." + side, img.size)
        os.makedirs(CELL_DIR, exist_ok=True)
        session_path = os.path.join(CELL_DIR, "scan.json")
        if os.path.exists(session_path):
            os.remove(session_path)
        cells = scan(img, box, n, save_cells=True)
        scan_id = uuid.uuid4().hex
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump({"scanId": scan_id, "n": n}, f)
        tpl = load_templates()
        res = classify(cells, tpl) if tpl else [
            {"i": c["i"], "label": None, "score": 0, "margin": 0, "rect": c["rect"], "crop": b64(c["crop"])}
            for c in cells
        ]
        cls = cluster_cells(cells)
        # 用已识别结果给簇打标
        by_i = {r["i"]: r for r in res}
        clusters = []
        for cl in cls:
            labs = [by_i[i]["label"] for i in cl["cells"]]
            lab = labs[0] if labs and all(l == labs[0] for l in labs) else None
            clusters.append({"id": cl["id"], "crop": b64(cl["crop"]), "cells": cl["cells"], "label": lab})

        # 宝牌指示牌：5 格，翻开的能认出，牌背会被判成未知而自动跳过
        dora = []
        if dora_box and len(dora_box) == 4 and tpl:
            d_cells = scan(img, dora_box, 5, save_cells=False)
            for r in classify(d_cells, tpl, thr=0.78, margin_min=0.01):
                if r["label"] is not None:
                    dora.append({"slot": r["i"], "label": r["label"], "score": r["score"], "crop": r["crop"]})

        # 三家舍牌（横向自动判断）
        # 模板不齐时不准，宁可不认：没见过的牌会被硬套成最像的模板
        discards = {}
        disc_note = "舍牌识别为实验功能，可能漏牌；请对照截图手动填写已见牌" if disc_boxes else None
        if disc_boxes:
            if len(tpl) == 34:
                discards = scan_discards(img, disc_boxes, tpl)
            else:
                disc_note = "模板只有 %d/34，舍牌先不启用（认错牌比认不出更糟）" % len(tpl)

        labels = [r["label"] for r in res]
        disp = img.copy()
        disp.thumbnail((1280, 1280), Image.BILINEAR)
        return {
            "ok": True, "mode": "scan",
            "w": img.width, "h": img.height,
            "dispW": disp.width, "dispH": disp.height,
            "scale": disp.width / float(img.width),
            "image": b64(disp),
            "box": list(box), "n": n, "scanId": scan_id,
            "doraBox": list(dora_box) if dora_box else None,
            "cells": res, "clusters": clusters,
            "hand": hand_text(labels),
            "unknown": sum(1 for v in labels if v is None),
            "dora": dora,
            "doraNote": "模板识别无法保证宝牌完整；请对照截图手动填写" if dora_box else None,
            "discards": discards,
            "discardNote": disc_note,
            "discardBoxes": disc_boxes or None,
            "templates": len(tpl),
        }

    if mode == "learn":
        # 用上一次 scan 存下的格子图，按人工标注写入模板
        assign = arg.get("assign", {})
        if not isinstance(assign, dict) or len(assign) > 14:
            raise ValueError("assign 必须是最多 14 个格子的标注对象")
        try:
            with open(os.path.join(CELL_DIR, "scan.json"), "r", encoding="utf-8") as f:
                session = json.load(f)
        except (OSError, ValueError):
            raise ValueError("没有可学习的扫描，请先重新识别")
        if not arg.get("scanId") or arg.get("scanId") != session.get("scanId"):
            raise ValueError("这次扫描已过期，请重新识别后标注")
        n = tile_count(arg.get("n", session["n"]))
        if n != session["n"]:
            raise ValueError("标注张数和扫描不一致")
        for cell_i, label in assign.items():
            if not isinstance(cell_i, str) or cell_i not in {str(i) for i in range(n)}:
                raise ValueError("标注格子下标越界")
            if type(label) is not int or not 0 <= label < 34:
                raise ValueError("牌面标签必须是 0 到 33 的整数")
            if not os.path.isfile(os.path.join(CELL_DIR, "%d.png" % int(cell_i))):
                raise ValueError("扫描格子已失效，请重新识别")
        os.makedirs(TPL_DIR, exist_ok=True)
        saved = 0
        for cell_i, label in assign.items():
            p = os.path.join(CELL_DIR, "%d.png" % int(cell_i))
            if not os.path.exists(p):
                continue
            big = Image.open(p).convert("RGB")
            v = center_vec(windows(big))
            np.save(os.path.join(TPL_DIR, "%d.npy" % int(label)), v.astype(np.float32))
            saved += 1
        return {"ok": True, "mode": "learn", "saved": saved}

    if mode == "calib":
        box = arg.get("box")
        n = tile_count(arg.get("n", 14))
        if not box:
            return {"ok": False, "error": "缺 box"}
        box = valid_box(box, cells=n)
        json.dump({"box": box, "n": n}, open(CALIB, "w", encoding="utf-8"), ensure_ascii=False)
        return {"ok": True, "mode": "calib", "box": box, "n": n}

    return {"ok": False, "error": "unknown mode"}


def _utf8_stdio():
    """Windows 下管道默认是 GBK，必须强制 UTF-8，否则中文会变乱码"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass


def serve():
    """常驻模式：从 stdin 逐行读请求，往 stdout 逐行写结果（省掉每次 150ms 的 import）"""
    _utf8_stdio()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = dispatch(json.loads(line))
        except Exception as e:
            import traceback
            out = {"ok": False, "error": str(e), "trace": traceback.format_exc()[-600:]}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main():
    _utf8_stdio()
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    if raw.startswith("@"):
        with open(raw[1:], "r", encoding="utf-8") as f:
            raw = f.read()
    arg = json.loads(raw) if raw.strip() else {}
    print(json.dumps(dispatch(arg), ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve()
    else:
        try:
            main()
        except Exception as e:
            import traceback
            print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()[-800:]}, ensure_ascii=False))










