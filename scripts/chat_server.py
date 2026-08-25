"""Tiny web server for the local chat page (chat.html).

Fixes the two classic failure modes of the previous raw `python -m http.server`
approach:
  1. `--directory "%~dp0"` from a .bat ends with a backslash -> the trailing \"
     is parsed as an escaped quote and the directory becomes garbage -> 404 on
     every file. Here the directory is derived from __file__, always correct.
  2. On Windows two servers can silently share the same port (SO_REUSEADDR),
     so the browser could hit a broken instance. Here we probe first: if an
     instance already serves /chat.html we just open the browser on it;
     otherwise we pick the first genuinely free port (8080..8090).

The script opens the chat page in the default browser by itself, so CHAT.bat
does not need to know which port was chosen.
"""
import http.server
import os
import socket
import sys
import threading
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTS = range(8080, 8091)


def serves_chat(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/chat.html", timeout=1
        ) as r:
            return r.status == 200
    except Exception:
        return False


def port_free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)


def main() -> int:
    for p in PORTS:
        if serves_chat(p):
            url = f"http://127.0.0.1:{p}/chat.html"
            print(f"[chat-ui] already running -> {url}")
            threading.Timer(0.5, webbrowser.open, [url]).start()
            return 0

    port = next((p for p in PORTS if port_free(p)), None)
    if port is None:
        print("[chat-ui] ERROR: no free port in 8080-8090")
        return 1

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/chat.html"
    print(f"[chat-ui] serving {ROOT}")
    print(f"URL={url}")
    threading.Timer(0.8, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
