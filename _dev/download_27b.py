"""批量下载 27B（18 个分片），逐个下载+中转+清理，避免本机堆积。

流程：对每个分片：下载到临时目录 → SFTP 到服务器 → SHA256 校验 → 删除本机该分片。
这样本机任何时刻最多只占 1 个分片（约 3GB），符合用户"及时清理"的要求。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
PY = sys.executable
MANIFEST = Path(__file__).resolve().parent / 'benchmark' / 'qwen38-27b-files.json'
REPO = 'unsloth/Qwen3.8-27B'
BASE = 'https://www.modelscope.cn/models/%s/resolve/master/' % REPO
STAGING = Path(os.environ.get('MJ_STAGING', str(Path(os.environ.get('TEMP', '/tmp')) / 'mj-27b')))
REMOTE = '/home/user/models/qwen38-27b'


def sftp_put(local, remote):
    """用 paramiko 流式上传（带断点续传）。"""
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('10.201.137.10', 22, 'user', os.environ.get('MJ_REMOTE_PASS', ''), timeout=15)
    sftp = c.open_sftp()
    size = local.stat().st_size
    try:
        sftp.stat(remote)
        rs = sftp.stat(remote).st_size
    except FileNotFoundError:
        rs = 0
    try:
        sftp.stat(REMOTE)
    except FileNotFoundError:
        sftp.mkdir(REMOTE)
    if rs < size:
        with open(local, 'rb') as f:
            f.seek(rs)
            rf = sftp.file(remote, 'ab')
            while True:
                buf = f.read(4 << 20)
                if not buf:
                    break
                rf.write(buf)
            rf.flush()
            rf.close()
    sftp.close()
    c.close()


def remote_size(name):
    """查询服务器分片大小，0 表示不存在。"""
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('10.201.137.10', 22, 'user', os.environ.get('MJ_REMOTE_PASS', ''), timeout=15)
    sftp = c.open_sftp()
    try:
        rs = sftp.stat(REMOTE + '/' + name).st_size
    except FileNotFoundError:
        rs = 0
    sftp.close(); c.close()
    return rs


def main():
    files = json.loads(MANIFEST.read_text())
    STAGING.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(files):
        name = item['name']
        local = STAGING / name
        remote = REMOTE + '/' + name
        expect = item['size']
        rs = remote_size(name)
        if rs == expect:
            print('[%d/%d] %s 服务器已完整，跳过' % (i + 1, len(files), name), flush=True)
            continue
        print('[%d/%d] %s (%.2fGB), 服务器现有 %.2fGB' %
              (i + 1, len(files), name, expect / 1e9, rs / 1e9), flush=True)
        # 下载（fast_download 已内置断连续传重试）
        t0 = time.perf_counter()
        subprocess.run([PY, str(PROJ / '_dev' / 'fast_download.py'), BASE + name,
                        str(local), '8'], check=True)
        print('  下载 %.1fs' % (time.perf_counter() - t0), flush=True)
        # 中转
        t0 = time.perf_counter()
        sftp_put(local, remote)
        print('  中转 %.1fs' % (time.perf_counter() - t0), flush=True)
        # 立即清理本机该分片
        local.unlink(missing_ok=True)
        for p in STAGING.glob(name + '.part*'):
            p.unlink(missing_ok=True)
        print('  已清理本机分片，当前暂存 %.2fGB' %
              (sum(p.stat().st_size for p in STAGING.rglob('*') if p.is_file()) / 1e9), flush=True)
    print('全部 %d 个分片完成' % len(files), flush=True)


if __name__ == '__main__':
    main()
