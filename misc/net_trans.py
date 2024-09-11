import socket
import threading
import argparse


def forward(source, target):
    while True:
        data = source.recv(4096)  # 从源接收数据
        if not data:
            break
        target.send(data)  # 将数据发送到目标


if __name__ == "__main__":
    # 创建套接字连接
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, default=12345, help="port to listen on"
    )
    parser.add_argument(
        "-i", "--ip", type=str, default="127.0.0.1", help="ip to trans on"
    )
    parser.add_argument(
        "-t", "--tport", type=int, default=12346, help="port to forward to"
    )

    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", args.port))  # 绑定到B的端口
    server.listen(0)

    client_a, addr_a = server.accept()  # 接受来自A的连接
    print("Connected by", addr_a)

    client_c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_c.connect((args.ip, args.tport))  # 连接到C

    # 启动转发线程
    t1 = threading.Thread(target=forward, args=(client_a, client_c))
    t2 = threading.Thread(target=forward, args=(client_c, client_a))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
