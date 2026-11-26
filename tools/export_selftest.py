"""Dev tool: run the workstation's real image exports in headless Edge/Chrome and save the facts (and previews) they report.

usage: python tools/export_selftest.py runs/ws.html out_dir [WxH] [full]
Writes out_dir/selftest.json plus preview PNGs (downscaled exports and a full-resolution crop) and any ZIP produced.

How it waits: headless --dump-dom fires on the page's load event, and the exports are asynchronous. The page therefore adds an <img>
pointing at a local server that holds the request open; the page releases it (GET /done) when the last export has finished.
"""
import base64
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


def _server():
    done = threading.Event()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/done"):
                done.set()
            else:
                done.wait(timeout=900)
            self.send_response(200)
            self.send_header("Content-Type", "image/gif")
            self.send_header("Content-Length", str(len(GIF)))
            self.end_headers()
            self.wfile.write(GIF)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, done


def dump(html: str, fragment: str, w: int = 1100, h: int = 700, timeout: int = 900) -> str:
    """DOM of the page after it has reported completion (the page adds #hold=PORT's <img> and calls /done when finished)."""
    exe = next(p for p in EDGE if Path(p).exists())
    srv, done = _server()
    try:
        url = Path(html).resolve().as_uri() + "#" + fragment.lstrip("#") + f"&hold={srv.server_address[1]}"
        cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", f"--window-size={w},{h}", "--dump-dom", url]
        return subprocess.run(cmd, capture_output=True, timeout=timeout, text=True, encoding="utf-8", errors="replace").stdout
    finally:
        done.set()
        srv.shutdown()


def run(html: str, w: int = 1100, h: int = 700, spec: str = "", timeout: int = 900) -> dict:
    dom = dump(html, "selftest=export" + (f"&spec={spec}" if spec else ""), w, h, timeout)
    found = re.findall(r'<pre id="selftest" style="display:\s*none;?">(\{.*?\})</pre>', dom, re.S)
    if not found:
        return {"error": "selftest element not found", "has_errbox": 'id="errbox"' in dom}
    res = json.loads(found[-1].replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    stage = re.search(r'data-xstage="([^"]*)"', dom)
    res["last_stage"] = stage.group(1) if stage else None
    return res


def save(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(res.get("cases", [])):
        for key in ("preview", "crop"):
            if c.get(key):
                (out / f"{i}_{c['scope']}_{c['dpi']}_{key}.png").write_bytes(base64.b64decode(c[key]["dataUrl"].split(",", 1)[1]))
                c[key] = {k: v for k, v in c[key].items() if k != "dataUrl"}
        if c.get("zipB64"):
            (out / f"{i}_{c['scope']}.zip").write_bytes(base64.b64decode(c.pop("zipB64")))
    (out / "selftest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    size = [int(v) for v in sys.argv[3].split("x")] if len(sys.argv) > 3 else [1100, 700]
    r = run(sys.argv[1], *size, spec=sys.argv[4] if len(sys.argv) > 4 else "")
    save(r, Path(sys.argv[2]))
    print({k: v for k, v in r.items() if k not in ("cases", "plans")})
    for c in r.get("cases", []):
        print(c["scope"], c["dpi"], c["name"], c["bytes"], c.get("png"), c["ms"], "ms", c["info"].get("warnings"))
