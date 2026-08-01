#!/usr/bin/env python3
"""Run the GBARail map locally with concurrent, pre-compressed data delivery."""
from __future__ import annotations
import argparse
import http.server
import mimetypes
import threading
import urllib.parse
import webbrowser
from pathlib import Path

class RailMapHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def __init__(self, *args, directory: str, **kwargs) -> None:
        self.root = Path(directory)
        super().__init__(*args, directory=directory, **kwargs)
    def do_GET(self) -> None:  # noqa: N802
        request_path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip("/")
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        original = (self.root / request_path).resolve()
        gz_path = Path(str(original) + ".gz")
        try: original.relative_to(self.root.resolve())
        except ValueError: return self.send_error(403)
        if accepts_gzip and gz_path.is_file() and original.is_file():
            payload = gz_path.read_bytes()
            content_type = mimetypes.guess_type(str(original))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith(("text/","application/json")) or original.suffix==".geojson" else ""))
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload); return
        super().do_GET()
    def end_headers(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path.endswith((".js",".css",".json",".geojson",".csv")):
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

class ReusableThreadingHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def main() -> None:
    parser=argparse.ArgumentParser(description="Serve the GBARail map locally")
    parser.add_argument("--port",type=int,default=8080); parser.add_argument("--no-browser",action="store_true")
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    def handler(*a,**kw): return RailMapHandler(*a,directory=str(root),**kw)
    url=f"http://127.0.0.1:{args.port}/"
    with ReusableThreadingHTTPServer(("127.0.0.1",args.port),handler) as server:
        print(f"湾区轨道地图已启动：{url}")
        print("已启用并行请求、视野分片懒加载与预压缩传输。")
        print("按 Ctrl+C 停止。")
        if not args.no_browser: threading.Timer(.2,lambda:webbrowser.open(url)).start()
        try: server.serve_forever()
        except KeyboardInterrupt: print("\n已停止。")
if __name__=="__main__": main()
