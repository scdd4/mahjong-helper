"""SSH 训练监控：查看远程训练日志的进度。"""
import os
import paramiko
import sys

HOST = os.environ.get('MJ_REMOTE_HOST', '10.201.137.10')
PORT = int(os.environ.get('MJ_REMOTE_PORT', '22'))
USER = os.environ.get('MJ_REMOTE_USER', 'user')
PASS = os.environ.get('MJ_REMOTE_PASS', '')


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else '/home/user/mahjong_train/train-4b.log'
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=10)
    cmd = ("echo '=== 进程 ==='; ps aux|grep train_mahjong|grep -v grep|awk '{print $2,$3,$4,$10}'\n"
           "echo '=== 日志尾部 ==='; tail -12 " + log + "\n"
           "echo '=== checkpoint ==='; ls -d /home/user/mahjong_train/out-*/checkpoint-* 2>/dev/null | tail -5\n"
           "echo '=== GPU ==='; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null\n"
           "echo '=== 磁盘 ==='; df -h /home/user | tail -1")
    i, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode('utf-8', 'ignore'))
    c.close()


if __name__ == '__main__':
    main()
