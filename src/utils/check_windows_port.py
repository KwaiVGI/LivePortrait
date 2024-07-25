import socket
import sys

if len(sys.argv) != 2:
    print("Usage: python check_port.py <port>")
    sys.exit(1)

port = int(sys.argv[1])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
result = sock.connect_ex(('127.0.0.1', port))

if result == 0:
    print("LISTENING")
else:
    print("NOT LISTENING")
sock.close
