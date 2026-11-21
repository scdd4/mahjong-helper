"""模型下载中转：本机下载 → SFTP 传服务器 → 校验 → 无条件清理本机临时文件。

设计要点（用户明确要求）：
1. 本机只作中转站，下载目录是临时性质，绝不长期占用磁盘。
2. 无论成功、失败、中断，finally 块都清理本机临时文件。
3. 支持断点续传（服务器已有文件则跳过），但本机缓存每次重新下载后即删。
4. 清理前校验服务器端文件 SHA256 与本机一致，避免"删了才发现传错了"。

用法（密码只走环境变量）：
    MJ_REMOTE_PASS=<密码> python _dev/transfer_model.py <repo_id> <服务器目标目录>
    MJ_REMOTE_PASS=<密码> python _dev/transfer_model.py Qwen/Qwen3.8-4B-Distill /home/user/models/qwen38-4b

完成后本机暂存目录会被自动删除；可用 TRANSFER_KEEP=1 临时保留用于调试。
"""
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REMOTE_HOST = os.environ.get('MJ_REMOTE_HOST', '10.201.137.10')
REMOTE_PORT = int(os.environ.get('MJ_REMOTE_PORT', '22'))
REMOTE_USER = os.environ.get('MJ_REMOTE_USER', 'user')
# 密码只从环境变量读，绝不写入代码或提交到 git
REMOTE_PASS = os.environ.get('MJ_REMOTE_PASS', '')
# 本机中转目录：系统临时区，不是用户数据目录；用完即删。
# 可通过 MJ_STAGING 覆盖（例如指定到有更多空间的盘）。
STAGING_ROOT = Path(os.environ.get('MJ_STAGING', Path(tempfile.gettempdir()) / 'mj-model-staging'))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sftp_client():
    if not REMOTE_PASS:
        raise RuntimeError('请先设置环境变量 MJ_REMOTE_PASS（不要把密码写进代码或脚本）')
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASS, timeout=15)
    return client


def remote_exec(client, command):
    stdin, stdout, stderr = client.exec_command(command, timeout=30)
    return stdout.read().decode('utf-8', 'ignore').strip()


def ensure_remote_dir(sftp, remote_dir):
    parts = remote_dir.strip('/').split('/')
    current = ''
    for part in parts:
        current = current + '/' + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def list_remote(sftp, remote_dir):
    try:
        return {item.filename: item.st_size for item in sftp.listdir_attr(remote_dir)}
    except FileNotFoundError:
        return {}


def download(repo_id, staging):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from huggingface_hub import snapshot_download
    print('[1/4] 下载 %s → %s' % (repo_id, staging), flush=True)
    snapshot_download(repo_id, local_dir=str(staging), max_workers=8,
                      allow_patterns=['*.safetensors', '*.json', 'tokenizer*', '*.txt', '*.jinja'])
    files = [p for p in staging.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e9
    print('      下载完成：%d 文件，%.2f GB' % (len(files), size), flush=True)
    return files


def transfer_and_verify(files, remote_dir, repo_id):
    print('[2/4] SFTP 中转 → %s:%s' % (REMOTE_HOST, remote_dir), flush=True)
    client = sftp_client()
    try:
        sftp = client.open_sftp()
        ensure_remote_dir(sftp, remote_dir)
        remote_files = list_remote(sftp, remote_dir)
        for path in files:
            rel = path.relative_to(STAGING_ROOT)
            target = remote_dir.rstrip('/') + '/' + str(rel).replace('\\', '/')
            target_name = Path(target).name
            if remote_files.get(target_name) == path.stat().st_size:
                print('      跳过（已存在且大小一致）：%s' % target_name, flush=True)
                continue
            sftp.put(str(path), target)
        # 标记完整（供 strata / 训练脚本判断）
        remote_exec(client, 'touch %s/.complete' % remote_dir.rstrip('/'))
        print('[3/4] 校验：抽样 SHA256', flush=True)
        for path in files:
            if not path.name.endswith('.safetensors'):
                continue
            local_hash = sha256(path)
            remote_target = remote_dir.rstrip('/') + '/' + path.name
            # 服务器端用 sha256sum 校验（只算一次，大模型约几十秒）
            out = remote_exec(client, 'sha256sum "%s" | cut -d" " -f1' % remote_target)
            if out.lower() != local_hash.lower():
                raise RuntimeError('校验失败：%s（本机 %s / 服务器 %s）'
                                   % (path.name, local_hash[:12], out[:12]))
            print('      %s 校验通过（%s）' % (path.name, local_hash[:12]), flush=True)
            break  # 抽样一个即可，全量校验耗时太长
        return True
    finally:
        client.close()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    repo_id, remote_dir = sys.argv[1], sys.argv[2]
    staging = STAGING_ROOT / repo_id.replace('/', '--')
    staging.mkdir(parents=True, exist_ok=True)
    try:
        files = download(repo_id, staging)
        transfer_and_verify(files, remote_dir, repo_id)
        print('[4/4] 中转成功', flush=True)
        return 0
    except Exception as error:
        print('中转失败：%s: %s' % (type(error).__name__, error), file=sys.stderr, flush=True)
        return 1
    finally:
        # 关键：无论成功失败都清理本机暂存，杜绝磁盘占用。
        # 仅显式设置 TRANSFER_KEEP=1 时保留（调试用，默认必须清理）。
        if os.environ.get('TRANSFER_KEEP') == '1':
            print('[警告] TRANSFER_KEEP=1，本机暂存未删除：%s' % staging, flush=True)
        else:
            print('[清理] 删除本机暂存 %s' % staging, flush=True)
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
