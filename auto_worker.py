"""One-click window capture and hand recognition. No game input, no model API."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.python-deps'))


def dispatch(payload):
    if not isinstance(payload, dict) or payload.get('mode') != 'analyze-window':
        raise ValueError('自动分析请求无效')
    from PIL import Image
    from auto_recognize import recognize_hand
    started = time.perf_counter()
    if 'image' in payload:
        # Only the server's explicit test environment can supply this field.
        image_path = payload['image']
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            raise ValueError('离线测试图片不存在；未截取其他画面')
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        window = {'title': '离线识别验证', 'width': image.width, 'height': image.height, 'method': 'fixture'}
    else:
        from window_capture import capture_mahjong_window
        image, window = capture_mahjong_window()
        window = dict(window)
        window.setdefault('width', image.width)
        window.setdefault('height', image.height)
    try:
        result = recognize_hand(image)
        return {**result, 'ok': True, 'window': window, 'ms': round((time.perf_counter() - started) * 1000)}
    finally:
        image.close()


def handle(payload):
    try:
        return dispatch(payload)
    except ModuleNotFoundError as error:
        return {'ok': False, 'code': 'dependency_missing', 'error': '自动识别组件未就绪，请重新安装项目识别依赖（缺少 %s）' % error.name}
    except Exception as error:
        return {'ok': False, 'code': getattr(error, 'code', 'recognition_failed'), 'error': str(error) or '截图分析失败，请重试'}


def main():
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    if '--serve' not in sys.argv:
        payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {'mode': 'analyze-window'}
        print(json.dumps(handle(payload), ensure_ascii=False, allow_nan=False))
        return
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            result = handle(json.loads(line))
        except (ValueError, TypeError):
            result = {'ok': False, 'code': 'invalid_request', 'error': '自动分析请求不是有效的 JSON 对象'}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
