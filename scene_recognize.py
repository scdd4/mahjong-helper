"""Local full-scene evidence. Uncertain rivers/melds remain explicitly unverified."""
from collections import Counter
import math
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from desktop_runtime import ROOT, tile_index, effective_hand_total  # injects .python-deps into sys.path
import cv2
import numpy as np
from spark_strategy import Cancelled
from table_viewport import locate_table
from tile_recognizer import recognize as recognize_cached


def find_viewport(image):
    import cv2
    import numpy as np
    array = np.asarray(image.convert('RGB'))
    height, width = array.shape[:2]
    white = (array.min(axis=2) > 230).mean(axis=1)
    top, bottom = 0, height
    while top < height * .10 and white[top] > .75:
        top += 1
    while bottom > height * .90 and white[bottom - 1] > .75:
        bottom -= 1
    # A normal title bar can have a one-pixel gray border preceding it.
    header = [i for i in range(min(int(height * .08), 100)) if white[i] > .90]
    if header and min(header) < 5:
        top = max(top, max(header) + 1)
    return [0, top, width, bottom]


def _face_ink_ratio(rgb):
    import numpy as np
    h, w = rgb.shape[:2]
    interior = rgb[max(1,int(h*.09)):max(2,int(h*.94)), max(1,int(w*.08)):max(2,int(w*.92))]
    if interior.size == 0:
        return 1.0
    return float((interior.min(axis=2) < 105).mean())


def read_concealed(image, cancel=None):
    """Detect tile bodies in the bottom row, without forcing a 16:9 resize.

    Small windows are upscaled to a reference width before template matching;
    template scores otherwise drop with downscaling blur. Returned boxes stay
    in the caller's original pixel coordinates.
    """
    import cv2
    import numpy as np
    from auto_vision_vendor import hand

    reference_width = 1920
    scale = 1.0
    if image.width < reference_width:
        scale = reference_width / image.width
        work = image.resize((reference_width, max(2, round(image.height * scale))), Image.Resampling.LANCZOS)
    else:
        work = image
    try:
        return _read_concealed_impl(work, cancel, scale)
    finally:
        if work is not image:
            work.close()


def _read_concealed_impl(image, cancel, scale=1.0):
    import cv2
    import numpy as np
    from auto_vision_vendor import hand, split
    split._FRAME_CACHE.clear()  # thread-local: clear this worker's stale frame

    def aborted():
        return cancel is not None and cancel.is_set()

    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    start = int(h * .69)
    roi = rgb[start:]
    high = roi.max(axis=2).astype('int16')
    low = roi.min(axis=2).astype('int16')
    paper = ((high >= 150) & (low >= 110) & ((high - low) < 105)).astype('uint8')
    count, labels, stats, centers = cv2.connectedComponentsWithStats(paper, 8)
    candidates = []
    for index in range(1, count):
        x, y, tw, th, area = (int(v) for v in stats[index])
        if tw < w * .018 or tw > w * .085 or th < h * .06 or th > h * .24:
            continue
        if not .35 <= tw / th <= .95 or area < tw * th * .33:
            continue
        candidates.append((x, y + start, tw, th))
    if not candidates:
        return {'tiles': [], 'count': 0, 'reliable': False, 'warnings': ['没有找到完整手牌行']}
    groups = []
    for box in sorted(candidates, key=lambda b: b[1] + b[3]):
        base = box[1] + box[3]
        group = next((g for g in groups if abs(base - (g[0][1] + g[0][3])) < h * .025), None)
        if group is None:
            groups.append([box])
        else:
            group.append(box)
    boxes = max(groups, key=lambda g: (len(g), sum(b[2] * b[3] for b in g)))
    boxes.sort(key=lambda b: b[0])
    median_w = float(np.median([b[2] for b in boxes]))
    median_h = float(np.median([b[3] for b in boxes]))
    boxes = [b for b in boxes if .65 * median_w <= b[2] <= 1.35 * median_w and .70 * median_h <= b[3] <= 1.30 * median_h]
    tiles = []
    for x, y, tw, th in boxes:
        if aborted():
            return {'tiles': [], 'count': 0, 'reliable': False, 'warnings': ['手牌识别已取消']}
        top = max(0, y - int(th * .065))
        bottom = min(h, y + th + 1)
        full = rgb[top:bottom, x:x+tw]
        bright = full[(full.min(axis=2) > 130) & (full.max(axis=2) - full.min(axis=2) < 110)]
        if len(bright) > 50:
            background = np.median(bright, axis=0)
            highlighted = bool(background[0]-background[1] > 15 and background[0]-background[2] > 15)
            gain = max(background) / np.maximum(background, 1)
            balanced = np.clip(full * gain, 0, 255).astype('uint8')
        else:
            balanced = full
            highlighted = False
        options = []
        variants = []
        for candidate in (full, balanced):
            variants.append(candidate)
            for fraction in (.05, .08):
                variants.append(candidate[int(candidate.shape[0] * fraction):candidate.shape[0]-2])
        for crop in variants:
            for inset in (0, 2, 4):
                if crop.shape[0] <= inset * 2 + 10 or crop.shape[1] <= inset * 2 + 10:
                    continue
                part = crop[inset:crop.shape[0]-inset or None, inset:crop.shape[1]-inset or None]
                quality = {}
                name, score = recognize_cached(hand,cv2.cvtColor(part, cv2.COLOR_RGB2BGR), quality=quality)
                if name == '5z' and _face_ink_ratio(part) > .035:
                    continue
                options.append((float(score), name, quality))
        if options:
            score, name, quality = max(options, key=lambda item: item[0])
        else:
            score, name, quality = 0.0, None, {}
        tiles.append({'name': name, 'score': round(score, 4), 'box': [x, top, tw, bottom - top],
                      'margin': quality.get('margin'), 'secondary_margin': quality.get('secondary_margin', 0),
                      'source': 'local_template','highlighted':highlighted})
    unclear = [t for t in tiles if t['name'] and (t['score'] < .60 or (t['margin'] is not None and t['margin'] < .02 and t['secondary_margin'] < .03))]
    if unclear:
        # Independent global normalization can confirm a close pattern match.
        # It cannot replace a tile with a different label.
        try:
            from auto_vision_vendor import split as splitter
            splitter._FRAME_CACHE.clear()
            try:
                normalized=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(1280,720))
                reference=hand.read_hand(normalized,canvas_box=(0,0,1280,720))
                for candidate in reference['tiles']:
                    a,b,c,d=candidate['quad']
                    candidate['box']=[a*w/1280,b*h/720,(c-a)*w/1280,(d-b)*h/720]
            finally:
                splitter._FRAME_CACHE.clear()
            for tile in unclear:
                x,y,tw,th = tile['box']
                for candidate in reference.get('tiles', []):
                    a,b,c,d = candidate['box']
                    overlap = max(0,min(x+tw,a+c)-max(x,a))*max(0,min(y+th,b+d)-max(y,b))
                    if overlap / max(1,min(tw*th,c*d)) > .75 and candidate.get('name') == tile['name'] and candidate.get('score',0) >= .70:
                        if candidate.get('margin') is None or candidate['margin'] >= .02 or candidate.get('secondary_margin',0) >= .03:
                            tile['cross_checked'] = True
        except Exception:
            pass
    if scale != 1.0:
        for tile in tiles:
            tile['box'] = [round(v / scale) for v in tile['box']]
    warnings = []
    if len(tiles) not in (13, 14):
        warnings.append('手牌为%d张；可能有副露、遮挡或漏检' % len(tiles))
    if any(not t['name'] or (not t.get('cross_checked') and (t['score'] < .60 or (t['margin'] is not None and t['margin'] < .02 and t['secondary_margin'] < .03))) for t in tiles):
        warnings.append('部分手牌的本地识别不确定')
    counts = Counter(tile_index(t['name']) for t in tiles if t['name'])
    if any(n > 4 for n in counts.values()):
        warnings.append('本地结果含同种超过4张的牌')
    return {'tiles': tiles, 'count': len(tiles), 'reliable': not warnings, 'warnings': warnings}


def _recognize_tile_variants(rgb, hand_reader, reject_blank_white=False):
    """White-balance + multi-crop recognition, shared by hand and meld paths.

    rgb is an HxWx3 uint8 crop that may include some table background; the
    bright-pixel median drives per-channel gain before several crop variants
    are scored. Returns the best (name, score) or (None, 0.0).
    """
    full = rgb
    bright = full[(full.min(axis=2) > 130) & (full.max(axis=2) - full.min(axis=2) < 110)]
    if len(bright) > 50:
        background = np.median(bright, axis=0)
        gain = np.maximum(background, 1)
        balanced = np.clip(full * (max(background) / np.maximum(background, 1)), 0, 255).astype('uint8')
    else:
        balanced = full
    options = []
    for candidate in (full, balanced):
        variants = [candidate]
        for fraction in (.05, .08):
            if candidate.shape[0] > 20:
                variants.append(candidate[int(candidate.shape[0] * fraction):candidate.shape[0] - 2])
        for crop in variants:
            for inset in (0, 2, 4):
                if crop.shape[0] <= inset * 2 + 10 or crop.shape[1] <= inset * 2 + 10:
                    continue
                part = crop[inset:crop.shape[0] - inset or None, inset:crop.shape[1] - inset or None]
                try:
                    name, score = recognize_cached(hand_reader,cv2.cvtColor(part, cv2.COLOR_RGB2BGR))
                except Exception:
                    continue
                if reject_blank_white and name == '5z' and _face_ink_ratio(part) > .035:
                    continue
                options.append((float(score), name))
    if not options:
        return None, 0.0
    best_score, best_name = max(options, key=lambda item: item[0])
    return best_name, best_score


def _meld_quad_to_original(tile, viewport):
    """Map a meld tile quad on the 1280x720 normalized map to viewport pixels."""
    vx0, vy0, vx1, vy1 = viewport
    vw, vh = vx1 - vx0, vy1 - vy0
    quad = tile.get('quad4') or tile.get('quad')
    if quad is None:
        return None
    points = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    sx = vx0 + points[:, 0] * vw / 1280.0
    sy = vy0 + points[:, 1] * vh / 720.0
    x0, y0 = np.floor([sx.min(), sy.min()]).astype(int)
    x1, y1 = np.ceil([sx.max(), sy.max()]).astype(int)
    return x0, y0, x1, y1


def _warp_tile(orig_bgr, viewport, quad_norm):
    """Perspective-rectify a meld tile from its four normalized-map corners."""
    vx0, vy0, vx1, vy1 = viewport
    vw, vh = vx1 - vx0, vy1 - vy0
    points = np.asarray(quad_norm, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] = vx0 + points[:, 0] * vw / 1280.0
    points[:, 1] = vy0 + points[:, 1] * vh / 720.0
    w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[3] - points[2])))
    h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    if w < 10 or h < 10:
        return None
    destination = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(orig_bgr, matrix, (w, h))


def _refine_meld_tile(tile, orig_bgr, viewport, hand_reader):
    """Gate by score, then re-check low-confidence meld tiles.

    Called tiles sit at slight perspective angles, so the multi-crop hand
    matcher runs on a perspective-rectified crop over four rotations and keeps
    the best confident result compared against the meld template score.
    """
    merged = {**tile}
    score = float(tile.get('score') or 0)
    merged['meld_raw'] = tile.get('name')
    merged['name'] = tile.get('name') if score >= .55 else None
    if merged['name'] is not None:
        return merged
    quad = tile.get('quad4') or tile.get('quad')
    if quad is None:
        return merged
    try:
        tile_img = _warp_tile(orig_bgr, viewport, quad)
        if tile_img is None:
            return merged
        rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
        by_name = {}
        for rotation in range(4):
            image = np.rot90(rgb, k=rotation)
            name, cross_score = _recognize_tile_variants(image, hand_reader)
            # A blank white-dragon face is rotation-symmetric template candy;
            # below .85 it is noise on inked number-tile faces (same guard the
            # region fallback uses).
            if name == '5z' and cross_score < .85:
                continue
            if name and cross_score >= .50:
                by_name.setdefault(name, []).append(cross_score)
            # The correct orientation consistently scores >= .55 on clean
            # crops (measured on truth-labeled melds), so stop there instead
            # of paying for all four rotations.
            if cross_score >= .55:
                break
        if by_name:
            # Multi-rotation agreement on one name beats a single lucky read.
            agreed = [(n, v) for n, v in by_name.items() if len(v) >= 2]
            cross_score, name = max((max(v), n) for n, v in by_name.items())
            if agreed:
                name, scores = max(agreed, key=lambda item: max(item[1]))
                cross_score = max(scores)
            face_ink = _face_ink_on_paper(rgb)
            if cross_score >= .60 and cross_score >= score:
                merged['name'] = name
                merged['score'] = round(cross_score, 4)
                merged['cross_checked'] = True
            elif cross_score >= .50 and cross_score >= score and agreed:
                merged['name'] = name
                merged['score'] = round(cross_score, 4)
                merged['cross_checked'] = True
            elif cross_score >= .50 and cross_score >= score and face_ink >= .04:
                # An inked face rules the white-dragon out, and measured meld
                # reads show correct number-tile names start near .50 while
                # wrong ones (now guarded) stay below .40 — accept with the
                # cross-checked flag so downstream keeps it verifiable.
                merged['name'] = name
                merged['score'] = round(cross_score, 4)
                merged['cross_checked'] = True
    except Exception:
        pass
    return merged


def _reinforce_meld_group(group):
    """A pon/kan shows repeated copies of one tile kind; use that as evidence.

    Votes are cast per tile kind (a red five counts as the same kind as a
    normal five), so a mixed 5p/0p pon is recognised as one kind. Chi mixes
    different kinds and is left untouched. Unnamed tiles inherit the majority
    kind only when at least two named copies agree, flagged low-confidence.
    """
    if not group:
        return
    kind_votes = Counter()
    for tile in group:
        if tile.get('name'):
            try:
                kind_votes[tile_index(tile['name'])] += 1
            except (ValueError, TypeError):
                continue
    if not kind_votes:
        return
    kind, votes = kind_votes.most_common(1)[0]
    named_all = [t for t in group if t.get('name') and tile_index(t['name']) == kind]
    if votes < 2:
        # A pon/kan holds three copies of one kind. Two independent pieces of
        # evidence can name the whole set: a single high-confidence cross-read,
        # or a unanimous raw meld kind (e.g. 5p/0p/5p is one kind of tile).
        anchor = named_all[0] if len(named_all) == 1 else None
        if anchor and anchor.get('cross_checked') and (anchor.get('score') or 0) >= .70:
            kind = tile_index(anchor['name'])
            source = anchor['name']
        else:
            raw_kinds = set()
            for tile in group:
                if tile.get('meld_raw'):
                    try:
                        raw_kinds.add(tile_index(tile['meld_raw']))
                    except (ValueError, TypeError):
                        continue
            if len(raw_kinds) == 1 and kind in raw_kinds:
                named_all = [t for t in group if t.get('name')]
                source = max(named_all, key=lambda t: t.get('score') or 0)['name']
            else:
                return
        for tile in group:
            if not tile.get('name') or tile_index(tile['name']) != kind:
                tile['name'] = source
                tile['inferred'] = True
        group[0]['group_confidence'] = 'low'
        return
    named = named_all
    best = max(named, key=lambda t: t.get('score') or 0)
    for tile in group:
        if not tile.get('name') or tile_index(tile['name']) != kind:
            tile['name'] = best['name']
            tile['inferred'] = True
    group[0]['group_confidence'] = 'low'


def _infer_chi_middle(group):
    """A chi shows three consecutive kinds of one suit in display order.

    When exactly two tiles are named with the same suit and a gap of two, the
    missing middle kind is uniquely determined no matter which side of the
    table the meld sits on, so it is safe to infer. Edge tiles (gap of one)
    stay unnamed: their identity depends on display direction, which varies
    by seat. Inferred tiles are flagged and the group stays low-confidence.
    """
    if len(group) != 3:
        return
    named = [t for t in group if t.get('name')]
    unnamed = [t for t in group if not t.get('name')]
    if len(named) != 2 or len(unnamed) != 1:
        return
    try:
        first, second = sorted((tile_index(t['name']) for t in named))
    except (ValueError, TypeError):
        return
    if first >= 27 or second >= 27 or first//9 != second//9 or second - first != 2:
        return
    # Index 0-26 maps to 1m-9m/1p-9p/1s-9s; the middle of a gap-2 pair is
    # uniquely determined regardless of meld display direction.
    middle_code = str(first % 9 + 2) + 'mps'[first // 9]
    unnamed[0]['name'] = middle_code
    unnamed[0]['inferred'] = True
    unnamed[0]['score'] = 0.0
    group[0]['group_confidence'] = 'low'


# Across-meld zone as ratios of the game viewport, matching the widened
# upstream region on the normalized map. Others' called tiles sit here.
_ACROSS_MELD_RATIOS = (0.1875, 0.0278, 0.2344, 0.1319)
_RIGHT_MELD_RATIOS = (0.7461, 0.0222, 0.0754, 0.1639)
# Own melds sit right of the hand row (measured from quad bounding boxes);
# the hand never reaches this far right, so a closed hand yields no blocks.
_OWN_MELD_RATIOS = (0.74, 0.845, 0.255, 0.135)


def _read_dora_panel(orig_bgr):
    """Skin-tolerant dora panel reading at original resolution.

    The upstream reader locates empty slots by a saturated green/orange mask,
    which misses skins whose empty slots are a pale leaf motif. This fallback
    detects the panel's white tile faces instead and names them with the hand
    matcher; leaf slots are told apart from a true white dragon by their
    motif ink, and reading stops at the first leaf because indicators always
    sit left of the empty slots. Returns indicator names left to right, or []
    when the region does not pass the panel consistency checks.
    """
    H, W = orig_bgr.shape[:2]
    roi = orig_bgr[0:int(H * .30), 0:int(W * .30)]
    if roi.size == 0:
        return []
    paper = ((roi.max(axis=2) >= 150) & (roi.min(axis=2) >= 110)
             & ((roi.max(axis=2).astype(int) - roi.min(axis=2)) < 105)).astype('uint8')
    count, _, stats, _ = cv2.connectedComponentsWithStats(paper, 8)
    blocks = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if h < 40 or area < w * h * .35 or not .55 <= w / max(h, 1) <= .90:
            continue
        blocks.append([x, y, w, h])
    if not blocks:
        return []
    # Panel slots share one row; keep the row with the most candidates.
    rows = {}
    for block in blocks:
        rows.setdefault(block[1] // 20, []).append(block)
    blocks = max(rows.values(), key=len)
    if not 1 <= len(blocks) <= 6:
        return []
    blocks.sort(key=lambda b: b[0])
    tiles = []
    for x, y, w, h in blocks:
        crop_bgr = roi[max(0, y - 3):y + h + 5, max(0, x - 2):x + w + 3]
        if crop_bgr.size == 0:
            continue
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        ink = _face_ink_on_paper(rgb)
        name, score = _recognize_tile_variants(rgb, _hand_reader_for_panel())
        if not name:
            return []          # unreadable slot: the panel hypothesis fails
        if name == '5z' and ink > .15:
            break              # first leaf-motif empty slot: indicators done
        if name == '5z' and score < .85:
            return []
        tiles.append(name)
    return tiles if 1 <= len(tiles) <= 5 else []


def _hand_reader_for_panel():
    """The hand matcher module is the panel's recognizer (lazy import)."""
    from auto_vision_vendor import hand as hand_module
    return hand_module


def _face_ink_on_paper(rgb):
    """Ink fraction measured only on the paper face, ignoring table background."""
    bright = ((rgb.max(axis=2) >= 150) & (rgb.min(axis=2) >= 110)
              & ((rgb.max(axis=2).astype(int) - rgb.min(axis=2)) < 105))
    dark = rgb.min(axis=2) < 105
    zone = cv2.dilate(bright.astype('uint8'), np.ones((5, 5), 'uint8')).astype(bool)
    return float((dark & zone).sum()) / max(1, int(bright.sum()))


def _recognize_tile_either_way(crop, hand_reader):
    """Recognize one tile crop; inverted (opponent-view) tiles are tried rotated.

    A blank dragon scores near 0.98 on its clean white face, so 5z is only
    trusted above 0.85; lower 5z matches are template noise on paper-heavy
    number-tile faces. The returned ink fraction feeds group-level checks.
    """
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 and crop.shape[2] == 3 else crop
    ink = _face_ink_on_paper(rgb)
    best = {'name': None, 'score': 0.0, 'ink': round(ink, 4)}
    for rotated, image in ((False, rgb), (True, cv2.rotate(rgb, cv2.ROTATE_180))):
        try:
            name, score = _recognize_tile_variants(image, hand_reader)
        except Exception:
            continue
        if name == '5z' and score < .85:
            continue
        if name and score > best['score']:
            best = {'name': name, 'score': round(float(score), 4), 'rotated': rotated,
                    'ink': round(ink, 4)}
    return best


def _fallback_seat_melds(orig_bgr, ratios, hand_reader, axis='x', four_way=False):
    """Screen-region fallback for a seat whose melds the upstream reader missed.

    Runs at original resolution: paper blobs become one slot each (pitch is the
    tile pitch), recognized over rotations, grouped by spacing, then reinforced
    by the pon/kan same-tile constraint. Only well-evidenced groups survive.
    axis='x' lays slots out horizontally (across), 'y' vertically (right).
    """
    H, W = orig_bgr.shape[:2]
    rx, ry, rw, rh = ratios
    roi = orig_bgr[int(ry * H):int((ry + rh) * H), int(rx * W):int((rx + rw) * W)]
    if roi.size == 0:
        return []
    paper = ((roi.max(axis=2) >= 150) & (roi.min(axis=2) >= 110)
             & ((roi.max(axis=2).astype(int) - roi.min(axis=2)) < 105)).astype('uint8')
    count, _, stats, _ = cv2.connectedComponentsWithStats(paper, 8)
    blocks = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if w < 18 or h < 24 or area < w * h * .3:
            continue
        blocks.append([x, y, w, h])
    if not blocks:
        return []
    tile_h = float(np.median([b[3] for b in blocks]))
    if axis == 'y':
        centers = sorted(b[1] + b[3] / 2 for b in blocks)
    else:
        centers = sorted(b[0] + b[2] / 2 for b in blocks)
    gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 8]
    # Meld tiles sit on a constant pitch; that pitch IS the tile width/height.
    pitch = float(np.median(gaps)) if gaps else max(tile_h * .72, 30.0)
    slots = []
    for x, y, w, h in blocks:
        cx, cy = x + w / 2, y + h / 2
        if axis == 'y':
            sy, ey = int(cy - pitch / 2), int(cy + pitch / 2)
            slots.append((x, x + w, sy, ey))
        else:
            sx, ex = int(cx - pitch / 2), int(cx + pitch / 2)
            slots.append((sx, ex, y, y + h))
    key = (lambda s: (s[2] + s[3]) / 2) if axis == 'y' else (lambda s: (s[0] + s[1]) / 2)
    slots.sort(key=key)
    recognized = []
    for sx, ex, sy, ey in slots:
        crop = roi[max(0, sy - 3):ey + 4, max(0, sx - 2):ex + 2]
        if crop.size == 0:
            continue
        if four_way:
            best = {'name': None, 'score': 0.0, 'ink': 0.0}
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            best['ink'] = round(_face_ink_on_paper(rgb), 4)
            for k in range(4):
                image = np.rot90(rgb, k=k)
                try:
                    name, score = _recognize_tile_variants(image, hand_reader)
                except Exception:
                    continue
                if name == '5z' and score < .85:
                    continue
                if name and score > best['score']:
                    best = {'name': name, 'score': round(float(score), 4),
                            'rotated': k != 0, 'ink': best['ink']}
        else:
            best = _recognize_tile_either_way(crop, hand_reader)
        recognized.append({'slot': (sx, ex, sy, ey), **best})
    if not recognized:
        return []
    groups, current = [], [recognized[0]]
    for item in recognized[1:]:
        key2 = (lambda s: (s[0] + s[1]) / 2) if axis == 'x' else (lambda s: (s[2] + s[3]) / 2)
        gap = key2(item['slot']) - key2(current[-1]['slot'])
        if gap > pitch * 1.45:
            groups.append(current)
            current = [item]
        else:
            current.append(item)
    groups.append(current)
    out = []
    for run in groups:
        # Called sets come in threes (pon/chi) or fours (kan). False positives
        # pollute the seen-tile counts, so adopt only well-evidenced sets:
        # three unanimous strong slots, or two agreeing inverted slots with a
        # confident anchor (an opponent-view number set). A four-slot run is
        # kept whole as a kan candidate instead of being cut into 3+leftover.
        chunks = [run] if len(run) == 4 else [run[i:i + 3] for i in range(0, len(run) - 2, 3)]
        for chunk in chunks:
            if len(chunk) < 3:
                continue
            counts = Counter(t['name'] for t in chunk)
            name, votes = counts.most_common(1)[0] if counts else (None, 0)
            strong = votes >= 3 and any(t['score'] >= .90 for t in chunk)
            inverted_pair = (votes >= 2 and max(
                (t['score'] for t in chunk if t.get('name') == name and t.get('rotated')),
                default=0.0) >= .35
                             and all(t.get('ink', 0) > .05 for t in chunk))
            if strong or inverted_pair:
                confidence = 'high' if strong else 'low'
                tiles = []
                for t in chunk:
                    inferred = (not strong) and (t['name'] != name or t['score'] < .60)
                    tiles.append({'name': name, 'score': t['score'],
                                  'rotated': t.get('rotated', False),
                                  'inferred': inferred,
                                  'group_confidence': confidence})
                out.append(tiles)
            else:
                # A chi has three different tiles: no kind vote applies. Keep
                # the group as nameless so downstream knows a meld exists.
                tiles = [{'name': None, 'score': t['score'],
                          'rotated': t.get('rotated', False), 'inferred': True}
                         for t in chunk]
                _infer_chi_middle(tiles)
                out.append(tiles)
    return out


def read_scene(image, cancel=None):
    import cv2
    import numpy as np
    from auto_vision_vendor import split, reader, meld, hand

    def aborted():
        return cancel is not None and cancel.is_set()

    started = time.perf_counter()
    located, viewport_source = locate_table(image)
    viewport = located or find_viewport(image)
    hand_input = image.crop(viewport)
    crop = image.crop(viewport).convert('RGB')
    orig_bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(orig_bgr, (1280, 720), interpolation=cv2.INTER_AREA)
    result = {'viewport': viewport, 'image_size': list(image.size), 'hand': None,
              'rivers': {}, 'melds': {}, 'dora_indicators': [], 'riichi': {}, 'warnings': [],
              'metadata': {'round': None, 'scores': None, 'remaining': None,
                           'viewport_source': viewport_source,
                           'note': '场风、点棒与余牌由视觉模型读取'}}

    def _detect_riichi_by_geometry(items, values, seat):
        """Detect a sideways-laid tile in a river = riichi (heuristic, not fact).

        Riichi is physically shown as the last discarded tile turned 90°.
        The upstream orientation module already calibrates the aspect ratios
        from ground truth (own>1.18, side>1.5, across uses relative outlier).
        We reuse quad_ar + group median; only flag the LAST river tile to avoid
        mistaking sideways meld/dora tiles for riichi. Result is a candidate
        with reason, never silently treated as confirmed truth.
        """
        if not items or not values or len(items) < 4:
            return False, ''
        ars = [reader.quad_ar(it['quad']) for it in items]
        med = sorted(ars)[len(ars)//2]
        # absolute thresholds from orientation.py GT calibration
        abs_min = {'own': 1.18, 'left': 1.5, 'right': 1.5, 'across': 1.8}.get(seat, 1.8)
        # examine the last two river positions (riichi tile is last, may be followed
        # by a small gap) — require clear outlier vs the group median
        for idx in range(len(items)-1, max(-1, len(items)-3), -1):
            ar = ars[idx]
            if ar >= abs_min and ar >= med * 1.35:
                return True, '第%d张牌横放(ar=%.2f,组中位=%.2f)' % (idx+1, ar, med)
        return False, ''

    def _environment():
        """Rivers, melds, dora panel and riichi: independent of the hand read."""
        split._FRAME_CACHE.clear()  # thread-local: clear this worker's stale frame
        env = {'rivers': {}, 'melds': {}, 'dora_indicators': [], 'riichi': {}, 'warnings': []}
        try:
            full_box = (0, 0, 1280, 720)
            mask = split.paper_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            templates, _ = reader.load_templates()
            for seat in ('own', 'across', 'left', 'right'):
                if aborted():
                    raise Cancelled('识别已取消')
                raw = split.cut_seat(mask, 1280, 720, seat)
                items = []
                for i, rect in enumerate(raw):
                    if len(rect) == 4:
                        a,b,c,d = rect
                        quad = [[a,b],[c,b],[c,d],[a,d]]
                    else:
                        quad = np.asarray(rect).reshape(4,2).tolist()
                    items.append({'id': i, 'name': None, 'quad': np.asarray(quad, np.float32)})
                values = reader.read_group(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), items, seat, templates) if items else []
                env['rivers'][seat] = [{'name': item.get('name'), 'score': round(float(item.get('score', 0)), 4), 'unverified': True} for item in values]
                if seat != 'own':
                    detected, why = _detect_riichi_by_geometry(items, values, seat)
                    if detected:
                        env['riichi'][seat] = 'candidate'   # 图推测，非确认事实
                        env['warnings'].append('%s 疑似立直（%s），请结合立直棒与牌河复核' % (seat, why))
            groups = meld.read_all(bgr, full_box)
            for seat, value in groups.items():
                safe_groups = []
                for group in value.get('groups', []):
                    # orig_bgr is already cropped: its origin is (0, 0), even
                    # when the game's window sits elsewhere on the desktop.
                    refined = [_refine_meld_tile(tile, orig_bgr, [0, 0, orig_bgr.shape[1], orig_bgr.shape[0]], hand) for tile in group]
                    _reinforce_meld_group(refined)
                    _infer_chi_middle(refined)
                    if len(refined)==2 and all(t.get('kan') for t in refined):
                        names=[t.get('name') for t in refined]
                        if all(names) and tile_index(names[0])==tile_index(names[1]):
                            refined.extend([{'name':names[0],'score':min(t.get('score',0) for t in refined),
                                             'kan':True,'inferred':True} for _ in range(2)])
                    safe_groups.append(refined)
                env['melds'][seat] = {'groups': safe_groups, 'unverified': True,
                                      'note': '只作候选；请模型用原图复核吃/碰/杠与牌名'}
            for seat, ratios in (('across', _ACROSS_MELD_RATIOS), ('right', _RIGHT_MELD_RATIOS)):
                if not env['melds'][seat]['groups']:
                    fallback = _fallback_seat_melds(orig_bgr, ratios, hand,
                                                    axis='y' if seat == 'right' else 'x',
                                                    four_way=seat == 'right')
                    if fallback:
                        env['melds'][seat]['groups'] = fallback
                        env['melds'][seat]['note'] = '上游副露识别为空，已用区域兜底识别；低置信，请模型用原图复核'
            dora = hand.read_dora(bgr)
            indicators = dora.get('indicators', [])
            if not indicators:
                # The upstream colored-slot mask misses several skins; the
                # white-face panel reader is the tolerant fallback.
                indicators = _read_dora_panel(orig_bgr)
            env['dora_indicators'] = indicators
        except Cancelled:
            raise
        except Exception as error:
            env['warnings'].append('部分场面本地识别未完成：' + type(error).__name__)
        return env

    try:
        split._FRAME_CACHE.clear()
        with ThreadPoolExecutor(max_workers=2) as pool:
            hand_future = pool.submit(read_concealed, hand_input, cancel)
            env_future = pool.submit(_environment)
            own = hand_future.result()
            env = env_future.result()
        if aborted():
            return {'viewport': viewport, 'image_size': list(image.size), 'hand': own,
                    'rivers': {}, 'melds': {}, 'dora_indicators': [], 'warnings': ['识别已取消']}
        for tile in own['tiles']:
            tile['box'][0]+=viewport[0];tile['box'][1]+=viewport[1]
        result['hand'] = own
        result['rivers'] = env['rivers']
        result['melds'] = env['melds']
        result['dora_indicators'] = env['dora_indicators']
        result['riichi'] = env['riichi']
        result['warnings'].extend(env['warnings'])
        # A closed 13/14-tile hand cannot have melds; clear the upstream own
        # candidates (the hand result was not available inside the env thread),
        # then let the region fallback recover them for open hands.
        if own['count'] in (13, 14):
            result['melds']['own'] = {'groups': []}
        if own['count'] not in (13,14) and not result['melds']['own']['groups']:
            fallback = _fallback_seat_melds(orig_bgr, _OWN_MELD_RATIOS, hand)
            if fallback:
                result['melds']['own'] = {'groups': fallback, 'unverified': True,
                                          'note': '上游副露识别为空，已用本家区域兜底识别；低置信，请模型用原图复核'}
        own_meld_groups = result['melds'].get('own', {}).get('groups', [])
        own_meld_tiles = sum(len(group) for group in own_meld_groups)
        total = effective_hand_total(own['tiles'],own_meld_groups)
        concealed_clear=all(t.get('name') and (t.get('cross_checked') or (t.get('score',0)>=.60 and
                            (t.get('margin') is None or t.get('margin')>=.02 or t.get('secondary_margin',0)>=.03))) for t in own['tiles'])
        # 低置信/推断补全的副露不能把整手标成可靠，否则污染本地核算与事实注入
        melds_clear=all(len(g) in (3,4) and all(t.get('name') for t in g)
                        and not any(t.get('inferred') for t in g)
                        and (g[0].get('group_confidence') if isinstance(g[0],dict) else None)!='low'
                        for g in own_meld_groups)
        if own['count'] not in (13, 14) and total in (13, 14) and own_meld_groups and concealed_clear and melds_clear:
            # Open hand: concealed tiles plus called melds make a full hand.
            # Local template scores already gated the meld names above.
            own['reliable'] = True
            own['warnings'] = []
            own['open_hand'] = {'concealed': own['count'], 'meld_tiles': own_meld_tiles, 'total': total}
    except Cancelled:
        result['warnings'].append('识别已取消')
        raise
    except Exception as error:
        result['warnings'].append('部分场面本地识别未完成：' + type(error).__name__)
    finally:
        split._FRAME_CACHE.clear()
        crop.close()
        if hand_input is not image:
            hand_input.close()
    result['warnings'].extend(own['warnings'] if result['hand'] else [])
    result['warnings'].append('模板结果不是完整场况真值；对手暗牌不可见，振听与鸣牌顺序不能仅凭单帧完全确定')
    result['elapsed_ms'] = round((time.perf_counter() - started) * 1000)
    return result
