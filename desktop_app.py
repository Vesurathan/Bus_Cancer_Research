"""
desktop_app.py -- native desktop wrapper for the BUS malignancy assistant.

Boots the FastAPI backend (serve.py, which also serves the web_ui/ page) on a
background thread and opens a native window. The window shows a "starting" splash
immediately, then loads the real UI once the model is ready. One double-click,
no terminal, no browser, no npm.

    python desktop_app.py            # open the app window
    python desktop_app.py --selftest # boot, verify /health + UI, exit (no window)
"""
import os
import sys
import threading
import time
import urllib.request

# Reliable, fast defaults for a live demo: no Ollama dependency.
os.environ.setdefault("REFINE_MODE", "rule")
os.environ.setdefault("ALLOWED_ORIGINS", "*")

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8765"))
URL = f"http://{HOST}:{PORT}/"

_server_error = {}

SPLASH = """
<!doctype html><html><head><meta charset="utf-8"><style>
 html,body{height:100%;margin:0;background:#0a0f1a;color:#e8eefc;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
 .wrap{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px}
 .dot{width:16px;height:16px;border-radius:4px;background:#38bdf8;box-shadow:0 0 18px #38bdf8;
  animation:p 1.1s ease-in-out infinite}
 @keyframes p{0%,100%{opacity:.35;transform:scale(.9)}50%{opacity:1;transform:scale(1.1)}}
 h1{font-size:19px;margin:0} p{color:#8595b4;font-size:13.5px;margin:0;text-align:center;max-width:420px}
</style></head><body><div class="wrap">
 <div class="dot"></div>
 <h1>Starting the malignancy model…</h1>
 <p id="m">Loading the ViT and BiomedCLIP. The first launch can take up to a minute; after that it opens in a few seconds.</p>
</div></body></html>
"""

ERROR_HTML = """
<!doctype html><html><head><meta charset="utf-8"><style>
 html,body{height:100%;margin:0;background:#0a0f1a;color:#e8eefc;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
 .wrap{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:24px}
 h1{font-size:19px;margin:0;color:#f87171} p{color:#8595b4;font-size:13.5px;max-width:520px;text-align:center;line-height:1.5}
 code{color:#e8eefc}
</style></head><body><div class="wrap">
 <h1>Could not start the model</h1>
 <p>macOS may have blocked file access. Grant this app <b>Full Disk Access</b>
 (System Settings → Privacy &amp; Security → Full Disk Access → add “BUS Assistant”),
 then reopen. Details in <code>/tmp/bus_assistant.log</code>.</p>
</div></body></html>
"""


def _run_server():
    try:
        import uvicorn
        uvicorn.run("serve:app", host=HOST, port=PORT, log_level="warning")
    except Exception as e:  # noqa: BLE001
        _server_error["e"] = repr(e)


def _wait_healthy(timeout=180):
    """Poll /health until the model is loaded (first boot loads BiomedCLIP)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_error:
            return False
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def _prepare():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())


def selftest():
    _prepare()
    threading.Thread(target=_run_server, daemon=True).start()
    print("Starting the malignancy model …")
    if not _wait_healthy():
        print("ERROR: backend did not start.", _server_error.get("e", ""))
        sys.exit(1)
    with urllib.request.urlopen(URL, timeout=5) as r:
        html = r.read().decode("utf-8", "replace")
    ok = r.status == 200 and "Malignancy Assistant" in html
    print("UI served OK" if ok else "UI NOT served")
    sys.exit(0 if ok else 2)


def main():
    _prepare()
    threading.Thread(target=_run_server, daemon=True).start()

    import webview
    window = webview.create_window(
        "Breast-Ultrasound Malignancy Assistant",
        html=SPLASH, width=1180, height=880, min_size=(900, 640),
    )

    def _on_start():
        if _wait_healthy():
            window.load_url(URL)
        else:
            window.load_html(ERROR_HTML)

    webview.start(_on_start)   # blocks until the window closes; daemon server exits with it


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
