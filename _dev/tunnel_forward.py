"""本地端口转发：127.0.0.1:18080 -> Spark 127.0.0.1:8080（SSH direct-tcpip）。
仅用于 A/B 实验期间，服务只绑 127.0.0.1 时绕过 LAN 限制。用法：
  MJ_REMOTE_PASS=xxx python _dev/tunnel_forward.py
"""
import os
import socket
import threading

import paramiko

HOST = os.environ.get('MJ_REMOTE_HOST', '10.201.137.10')
USER = os.environ.get('MJ_REMOTE_USER', 'user')
PASS = os.environ.get('MJ_REMOTE_PASS', '')
LOCAL_PORT = 18080
REMOTE = ('127.0.0.1', 8080)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, 22, USER, PASS, timeout=15)
transport = c.get_transport()
print(f'tunnel {LOCAL_PORT} -> {HOST}:{REMOTE[1]} ready', flush=True)


def handle(client):
    try:
        channel = transport.open_channel('direct-tcpip', REMOTE, client.getpeername())

        def to_remote():
            try:
                while True:
                    data = client.recv(8192)
                    if not data:
                        break
                    channel.sendall(data)
            except Exception:
                pass
            finally:
                channel.close()

        def to_client():
            try:
                while True:
                    data = channel.recv(8192)
                    if not data:
                        break
                    client.sendall(data)
            except Exception:
                pass
            finally:
                client.close()

        threading.Thread(target=to_remote, daemon=True).start()
        threading.Thread(target=to_client, daemon=True).start()
    except Exception as exc:
        print('channel error:', exc, flush=True)
        client.close()


listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', LOCAL_PORT))
listener.listen(32)
while True:
    conn, _ = listener.accept()
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
