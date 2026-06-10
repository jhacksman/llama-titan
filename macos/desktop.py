#!/usr/bin/env python3
"""Run the Madrone Way dashboard as a desktop app.

Starts the FastAPI server on localhost in a background thread, then opens
a native macOS window (WKWebView via pywebview). If pywebview isn't
available, falls back to the default browser and keeps serving.
"""

import os
import socket
import sys
import threading
import time

PORT = 8742

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def port_open() -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    # If a previous copy is already serving, just open a window onto it.
    if not port_open():
        import uvicorn
        from app.main import app as fastapi_app

        threading.Thread(
            target=lambda: uvicorn.run(
                fastapi_app, host="127.0.0.1", port=PORT, log_level="warning"
            ),
            daemon=True,
        ).start()
        for _ in range(100):
            if port_open():
                break
            time.sleep(0.1)

    url = f"http://127.0.0.1:{PORT}/"
    try:
        import webview

        webview.create_window(
            "7223 NW Madrone Way", url,
            width=1240, height=900, min_size=(700, 600),
        )
        webview.start()
    except Exception:
        import webbrowser

        webbrowser.open(url)
        while True:  # keep the server alive for the browser tab
            time.sleep(3600)


if __name__ == "__main__":
    main()
