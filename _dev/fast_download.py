"""多线程分块下载器（ModelScope/HF resolve），带断点续传和进度显示。

用法：
    python _dev/fast_download.py <url> <output_path> [threads]

ModelScope 单线程约 3.8MB/s，8 线程可到 15-30MB/s，把 27B 下载从 4 小时压到约 40 分钟。
"""
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

DEFAULT_THREADS = 8
CHUNK = 1 << 20  # 1MB 块


def get_size(url):
    # ModelScope 对 HEAD 不返回 Content-Length，用 Range GET 探测
    for method, headers in (('HEAD', {}), ('GET', {'Range': 'bytes=0-0'})):
        try:
            req = urllib.request.Request(url, method=method,
                                         headers=dict(headers, **{'User-Agent': 'Mozilla/5.0'}))
            with urllib.request.urlopen(req, timeout=30) as r:
                cr = r.headers.get('Content-Range', '')
                if cr and '/' in cr:
                    return int(cr.split('/')[-1])
                length = r.headers.get('Content-Length', 0)
                if length:
                    return int(length)
        except Exception:
            continue
    return 0


def download_range(url, path, start, end, idx, progress):
    """下载 [start, end] 字节范围到临时分块文件，断连自动续传重试。"""
    part = Path(str(path) + '.part%d' % idx)
    target = end - start + 1
    for attempt in range(12):
        existing = part.stat().st_size if part.exists() else 0
        if existing >= target:
            progress[idx] = target
            return
        actual_start = start + existing
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=%d-%d' % (actual_start, end)})
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(part, 'ab') as f:
                while True:
                    try:
                        buf = r.read(CHUNK)
                    except Exception:
                        break  # 连接中断，跳出后按已下载字节续传
                    if not buf:
                        break
                    f.write(buf)
                    progress[idx] = part.stat().st_size
            if part.stat().st_size >= target:
                progress[idx] = target
                return
        except Exception as exc:
            print('  分块%d 第%d次重试: %s' % (idx, attempt + 1, str(exc)[:50]), flush=True)
        time.sleep(2)
    raise RuntimeError('分块%d 下载失败：%d/%d' % (idx, part.stat().st_size if part.exists() else 0, target))


def download(url, path, threads=DEFAULT_THREADS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = get_size(url)
    if not total:
        raise RuntimeError('无法获取文件大小，可能 URL 不对')
    print('文件大小 %.2f GB，%d 线程' % (total / 1e9, threads), flush=True)
    if path.exists() and path.stat().st_size == total:
        print('已存在且完整，跳过', flush=True)
        return path
    block = (total + threads - 1) // threads
    ranges = [(i * block, min((i + 1) * block, total) - 1) for i in range(threads)]
    ranges = [(s, e) for s, e in ranges if s <= e]
    progress = [0] * len(ranges)
    workers = []
    for i, (s, e) in enumerate(ranges):
        t = threading.Thread(target=download_range, args=(url, path, s, e, i, progress), daemon=True)
        t.start()
        workers.append(t)
    last_report = 0
    while any(t.is_alive() for t in workers):
        time.sleep(2)
        now = time.perf_counter()
        if now - last_report >= 10:
            done = sum(progress)
            print('  进度 %.1f%% (%.2f/%.2f GB)' % (100 * done / total, done / 1e9, total / 1e9), flush=True)
            last_report = now
    for t in workers:
        t.join()
    # 合并分块
    print('合并分块...', flush=True)
    with open(path, 'wb') as out:
        for i in range(len(ranges)):
            part = Path(str(path) + '.part%d' % i)
            with open(part, 'rb') as f:
                while True:
                    buf = f.read(4 << 20)
                    if not buf:
                        break
                    out.write(buf)
            part.unlink()
    if path.stat().st_size != total:
        raise RuntimeError('下载大小不符：%d != %d' % (path.stat().st_size, total))
    print('完成：%s (%.2f GB)' % (path, path.stat().st_size / 1e9), flush=True)
    return path


if __name__ == '__main__':
    url, output = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THREADS
    download(url, output, threads)
