import socket
import sys

def check_port(port: int, host: str = '127.0.0.1', timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_port.py <port>")
        sys.exit(1)

    try:
        port = int(sys.argv[1])
    except ValueError:
        print("Port must be an integer.")
        sys.exit(1)

    if check_port(port):
        print("LISTENING")
    else:
        print("NOT LISTENING")

if __name__ == "__main__":
    main()

