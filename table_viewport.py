"""Locate the table in an existing screenshot; never captures the desktop itself."""
from PIL import Image
from desktop_runtime import ROOT  # make project-scoped native dependencies available


def valid_box(box, size):
    try:
        left, top, right, bottom = map(int, box)
        return (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]
                and right - left >= 320 and bottom - top >= 180)
    except (TypeError, ValueError):
        return False


def _hand_anchor_count(rgb):
    """Count aligned, similarly sized tile bodies, without interpreting tile names."""
    import cv2
    import numpy as np
    height, width = rgb.shape[:2]
    start = round(height * .64)
    region = rgb[start:]
    high = region.max(axis=2).astype('int16')
    low = region.min(axis=2).astype('int16')
    mask = ((high >= 150) & (low >= 110) & (high - low < 105)).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if (width * .018 <= w <= width * .085 and height * .06 <= h <= height * .24
                and .35 <= w / max(1, h) <= .95 and area >= w * h * .33):
            boxes.append((int(x), int(y + start), int(w), int(h)))
    best = 0
    for seed in boxes:
        row = [b for b in boxes if abs(b[1] + b[3] - seed[1] - seed[3]) < height * .025
               and .65 * seed[2] <= b[2] <= 1.35 * seed[2]
               and .70 * seed[3] <= b[3] <= 1.30 * seed[3]]
        row.sort()
        if not row or row[-1][1] + row[-1][3] < height * .83:
            continue
        run = longest = 1
        for left, right in zip(row, row[1:]):
            gap = right[0] - left[0]
            run = run + 1 if .65 * seed[2] <= gap <= 1.9 * seed[2] else 1
            longest = max(longest, run)
        best = max(best, longest)
    return best


def detect_table_box(image):
    """Conservative visual fallback for a game window embedded in a full desktop.

    Dark connected surfaces propose rectangles; an aligned hand row must support
    a rectangle. This prevents the assistant, taskbar or ordinary desktop icons
    from becoming a table merely because they contain bright rectangles.
    """
    import cv2
    import numpy as np
    width, height = image.size
    scale = min(1.0, 1600 / width, 1200 / height)
    small = image.convert('RGB')
    try:
        if scale < 1:
            resized = small.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
            small.close()
            small = resized
        rgb = np.asarray(small)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = (gray < 175).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), 'uint8'))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for x, y, w, h, area in stats[1:]:
            if (w < 320 * scale or h < 180 * scale or area < rgb.shape[0] * rgb.shape[1] * .035
                    or not .9 <= w / max(1, h) <= 3.8 or area < w * h * .40):
                continue
            # Window shadows or black wallpaper can join the same component.
            # Keep its substantial rectangular surface, not those thin bridges.
            density = (gray[y:y+h, x:x+w] < 175).mean(axis=1) >= .28
            edges = np.diff(np.r_[False, density, False].astype('int8'))
            runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
            if not runs:
                continue
            start, end = max(runs, key=lambda span: span[1] - span[0])
            y, h = y + int(start), int(end - start)
            if h < 180 * scale:
                continue
            anchors = _hand_anchor_count(rgb[y:y+h, x:x+w])
            if not 5 <= anchors <= 14:
                continue
            box = [round(x / scale), round(y / scale),
                   min(width, round((x + w) / scale)), min(height, round((y + h) / scale))]
            if valid_box(box, image.size):
                candidates.append((anchors in (13, 14), anchors, area, box))
        if not candidates:
            return None
        return max(candidates, key=lambda value: value[:3])[3]
    finally:
        small.close()


def locate_table(image):
    explicit = image.info.get('game_bbox')
    if valid_box(explicit, image.size):
        return list(map(int, explicit)), image.info.get('viewport_source', 'window')
    detected = detect_table_box(image)
    if detected:
        # Preserve screenshots already filled by the game. Component edges can
        # stop at the white bottoms of tiles; trimming a few border pixels then
        # changes their crops and can invalidate a previously reliable match.
        area = (detected[2]-detected[0])*(detected[3]-detected[1])
        if area >= image.width*image.height*.90:
            return [0,0,image.width,image.height], 'image'
        return detected, 'visual'
    return None, 'unlocated'


def prepare_table_image(image):
    """Return a new image for preview and analysis, retaining the source offset."""
    box, source = locate_table(image)
    result = image.crop(tuple(box)) if box else image.copy()
    result.info = dict(image.info)
    result.info['viewport_source'] = source
    result.info['source_image_size'] = list(image.size)
    if box:
        result.info['source_game_bbox'] = box
        result.info['game_bbox'] = [0, 0, result.width, result.height]
    else:
        result.info.pop('game_bbox', None)
    return result
