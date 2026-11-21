"""监控 4B 训练完成，自动触发评测。后台运行，完成时输出结果。"""
import os
import paramiko
import time

HOST, USER = '10.201.137.10', 'user'
PASS = os.environ.get('MJ_REMOTE_PASS', '')
LOG = '/home/user/mahjong_train/train-4b.log'
DONE = '全部完成'


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PASS, timeout=15)
    return c


def check(c, cmd):
    i, o, e = c.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', 'ignore')


def main():
    print('[监控] 等待 4B 训练完成...', flush=True)
    started = time.perf_counter()
    while True:
        try:
            c = connect()
            out = check(c, 'tail -c 4000 %s' % LOG)
            c.close()
            if DONE in out:
                print('[监控] 训练完成！耗时 %.0f 分钟' % ((time.perf_counter() - started) / 60), flush=True)
                break
            # 进度
            import re
            m = re.findall(r'(\d+)/1200 \[[\d:]+<[\d:]+,\s*[\d.]+s/it\]', out)
            if m:
                print('[监控] 进度 %s/1200' % m[-1], flush=True)
            if 'Traceback' in out:
                print('[监控] 训练出错！', flush=True)
                print(out[-2000:], flush=True)
                return
        except Exception as exc:
            print('[监控] 连接重试: %s' % str(exc)[:60], flush=True)
        time.sleep(120)

    # 自动评测
    c = connect()
    cmd = """cd /home/user/mahjong_train && setsid /home/user/miniconda3/envs/mjtrain/bin/python -u eval_server.py \
  --model /home/user/mahjong_train/out-4b/merged \
  --scenarios /home/user/mahjong_train/data/scenarios.json \
  --out /home/user/mahjong_train/eval-4b.json \
  > /home/user/mahjong_train/eval-4b.log 2>&1 < /dev/null & echo eval-launched"""
    print('[评测]', check(c, cmd).strip(), flush=True)
    c.close()

    # 等评测完成
    while True:
        try:
            c = connect()
            out = check(c, 'ls -la /home/user/mahjong_train/eval-4b.json 2>/dev/null && tail -3 /home/user/mahjong_train/eval-4b.log')
            c.close()
            if 'eval-4b.json' in out:
                print('[评测] 完成！', flush=True)
                print(out[-500:], flush=True)
                break
        except Exception:
            pass
        time.sleep(30)
    print('[完成] 可生成实验报告', flush=True)


if __name__ == '__main__':
    main()
