"""Read a complete concealed Mahjong Soul hand from a PIL screenshot.

No calibration, user-provided templates, game hooks or network requests are used.
The bundled majsoul-vision templates and algorithms are covered by their MIT
license in auto_vision_vendor/LICENSE. Matching scores are similarity scores,
not calibrated probabilities. Coordinates are local to the supplied image.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".python-deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

TILE_NAME = re.compile(r"^(?:[1-9][mps]|0[mps]|[1-7]z)$")
DEFAULT_MIN_SCORE = 0.60


class RecognitionError(ValueError):
    """A screenshot cannot safely be used to produce hand advice."""
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _failure(code, message, started, **details):
    return {
        "ok": False,
        "code": code,
        "message": message,
        "hand": "",
        "tiles": [],
        "boxes": [],
        "confidence": 0.0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        **details,
    }


def _tile_label(name):
    digit, suit = name
    value = 5 if digit == "0" else int(digit)
    return {"m": 0, "p": 9, "s": 18, "z": 27}[suit] + value - 1


def _viewport(bgr, cv2, np):
    """Locate a 16:9 game client, including centered black letterboxing.

    Window capture normally supplies the complete client. Prefer its visible
    rectangle when it already has the expected aspect, so orange tile edges
    cannot be mistaken for upstream's warm-color wall anchors.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = gray > 20
    rows = np.flatnonzero(bright.mean(axis=1) > 0.10)
    cols = np.flatnonzero(bright.mean(axis=0) > 0.10)
    if len(rows) and len(cols):
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        if x1 - x0 >= 960 and y1 - y0 >= 500 and abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.055:
            return (x0, y0, x1, y1)
    return None


def _assess_hand(raw, started, min_score=DEFAULT_MIN_SCORE):
    """Validate the actual recognition result before allowing any advice."""
    tiles, boxes = [], []
    for index, item in enumerate(raw.get("tiles", [])):
        name = item.get("name")
        score = float(item.get("score") or 0)
        if not math.isfinite(score):
            score = 0.0
        rect = item.get("quad") or []
        if len(rect) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in rect):
            return _failure("invalid_geometry", "手牌位置识别不完整，请保持游戏画面完整可见后重试。", started)
        x0, y0, x1, y1 = (int(x) for x in rect)
        box = [x0, y0, x1 - x0, y1 - y0]
        if box[2] <= 0 or box[3] <= 0:
            return _failure("invalid_geometry", "手牌位置识别不完整，请保持游戏画面完整可见后重试。", started)
        valid_name = isinstance(name, str) and TILE_NAME.fullmatch(name) is not None
        tiles.append({
            "index": index,
            "name": name if valid_name else None,
            "label": _tile_label(name) if valid_name else None,
            "score": round(score, 4),
            "confidence": round(max(0.0, min(1.0, score)), 4),
            "box": box,
            "drawn": bool(item.get("drawn")),
            "red": bool(valid_name and name[0] == "0"),
            "margin": item.get("margin"),
            "alternatives": item.get("alternatives", []),
            "secondary_margin": item.get("secondary_margin", 0.0),
        })
        boxes.append(box)

    common = {
        "tiles": tiles,
        "boxes": boxes,
        "tile_count": len(tiles),
        "canvas": list(raw.get("canvas") or []),  # x0,y0,x1,y1
        "meld_tiles": int(raw.get("meld_tiles") or 0),
        "confidence": min((t["confidence"] for t in tiles), default=0.0),
        "mean_confidence": round(sum(t["confidence"] for t in tiles) / len(tiles), 4) if tiles else 0.0,
    }
    if common["meld_tiles"] > 0 or raw.get("meld_rect"):
        return _failure("open_hand", "检测到吃、碰或杠；目前只支持没有副露的完整手牌，本次不生成建议。", started, **common)
    if len(tiles) not in (13, 14):
        return _failure("invalid_count", f"识别到 {len(tiles)} 张手牌，需要完整的 13 或 14 张。请等摸牌动画结束，并保持整排手牌可见。", started, **common)
    uncertain = [t["index"] + 1 for t in tiles if t["name"] is None or not math.isfinite(t["score"]) or t["score"] < min_score
                 or (t["margin"] is not None and t["margin"] < 0.02 and t["secondary_margin"] < 0.03)]
    if uncertain:
        return _failure("low_confidence", "有牌面识别不够确定，本次不生成建议。请使用默认牌面，并保持游戏画面清晰完整后重试。", started, uncertain_tiles=uncertain, **common)
    counts = Counter(t["label"] for t in tiles)
    if any(n > 4 for n in counts.values()):
        return _failure("invalid_tiles", "识别结果出现同种牌超过 4 张，本次不生成建议。请保持手牌清晰可见后重试。", started, **common)
    red = Counter(t["name"][1] for t in tiles if t["red"])
    if any(n > 1 for n in red.values()):
        return _failure("invalid_tiles", "赤宝牌识别结果异常，本次不生成建议。请使用默认牌面后重试。", started, **common)

    return {
        "ok": True,
        "code": "recognized",
        "message": "已识别完整手牌。" if len(tiles) == 14 else "已识别 13 张手牌，显示听牌与进张。",
        "hand": "".join(t["name"] for t in tiles),
        "red": {suit: red.get(suit, 0) for suit in ("m", "p", "s")},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        **common,
    }


def recognize_image(image, min_score=DEFAULT_MIN_SCORE):
    """Return {ok, hand, tiles, confidence, boxes, ...} for a PIL Image.

    Each tile box is [x,y,width,height]; canvas is [x0,y0,x1,y1]. A failed
    result always has an empty hand, so a caller cannot accidentally analyze
    an unsupported or uncertain hand. The caller owns screenshot acquisition.
    """
    started = time.perf_counter()
    if not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not 0.5 <= min_score <= 1:
        return _failure("invalid_threshold", "识别阈值必须在 0.5 到 1 之间。", started)
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from auto_vision_vendor import hand as hand_reader
        from auto_vision_vendor import split as image_splitter
    except ImportError:
        return _failure("missing_dependency", "截图识别组件尚未安装，请运行启动程序完成依赖检查。", started)
    if not isinstance(image, Image.Image):
        return _failure("invalid_image", "识别需要有效的截图图片。", started)
    width, height = image.size
    if width < 960 or height < 540:
        return _failure("image_too_small", "游戏截图分辨率太低，请将游戏窗口放大到至少 960×540 后重试。", started, image=[width, height])
    if width * height > 40_000_000:
        return _failure("image_too_large", "截图尺寸过大，请只截取雀魂窗口。", started, image=[width, height])
    try:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        # Upstream caches full-size image masks using a sparse fingerprint.
        # Keep that cache strictly within one request: otherwise memory grows
        # with every screenshot and colliding fingerprints can reuse old masks.
        image_splitter._FRAME_CACHE.clear()
        try:
            viewport = _viewport(bgr, cv2, np) or image_splitter.canvas(bgr)
            vx0, vy0, vx1, vy1 = viewport
            vw, vh = vx1 - vx0, vy1 - vy0
            if vw < 960 or vh < 500 or abs(vw / vh - 16 / 9) > 0.08:
                return _failure("unsupported_viewport", "未找到完整的 16:9 游戏画面，请保持雀魂窗口完整可见后重试。", started, image=[width, height])
            cropped = bgr[vy0:vy1, vx0:vx1]
            normalized = cv2.resize(cropped, (1280, 720), interpolation=cv2.INTER_AREA if vw > 1280 else cv2.INTER_LINEAR)
            raw = hand_reader.read_hand(normalized, canvas_box=(0, 0, 1280, 720))
            sx, sy = vw / 1280, vh / 720
            def original_rect(rect):
                return [int(round(vx0 + rect[0] * sx)), int(round(vy0 + rect[1] * sy)),
                        int(round(vx0 + rect[2] * sx)), int(round(vy0 + rect[3] * sy))]
            for tile in raw.get("tiles", []):
                tile["quad"] = original_rect(tile["quad"])
            if raw.get("meld_rect"):
                raw["meld_rect"] = original_rect(raw["meld_rect"])
            raw["canvas"] = viewport
        finally:
            image_splitter._FRAME_CACHE.clear()
        canvas = raw.get("canvas") or (0, 0, width, height)
        x0, y0, x1, y1 = canvas
        if x1 - x0 < 960 or y1 - y0 < 500:
            return _failure("canvas_too_small", "未找到完整清晰的游戏画面，请将雀魂窗口放大并保持手牌完整可见。", started, image=[width, height])
        result = _assess_hand(raw, started, float(min_score))
        result["image"] = [width, height]
        result["engine"] = "majsoul-vision"
        result["engine_version"] = "70b7dc8407aedbb46166e587da397d4564c7f255"
        return result
    except Exception:
        return _failure("recognition_failed", "没有识别到清晰完整的手牌。请进入对局、使用默认牌面，等动画结束后重试。", started, image=[width, height])


def recognize_hand(image, min_score=DEFAULT_MIN_SCORE):
    """Strict worker API: raise RecognitionError rather than analyze a bad hand.

    tiles contains 0..33 integer tile labels in screen order; recognized_tiles
    retains per-tile names, red flags, boxes and similarity scores for review.
    """
    result = recognize_image(image, min_score)
    if not result["ok"]:
        raise RecognitionError(result["code"], result["message"], result)
    details = result.pop("tiles")
    red_counts = result.pop("red")
    result["recognized_tiles"] = details
    result["tiles"] = [tile["label"] for tile in details]
    result["red_counts"] = red_counts
    result["red"] = [red_counts[suit] > 0 for suit in ("m", "p", "s")]
    return result


def main():
    parser = argparse.ArgumentParser(description="离线识别雀魂截图中的完整门清手牌")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    try:
        from PIL import Image
        with Image.open(args.image) as source:
            result = recognize_image(source.convert("RGB"))
    except (OSError, ValueError):
        result = _failure("invalid_image", "无法读取指定图片。", time.perf_counter())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
