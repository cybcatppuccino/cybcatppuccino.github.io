#!/usr/bin/env python3
"""Run a small local web server and open the map in a browser."""
from __future__ import annotations

import argparse
import gzip
import http.server
import threading
import urllib.parse
import webbrowser
from pathlib import Path


class RailMapHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    """Serve files concurrently and gzip the bundled GeoJSON without changing it."""

    snapshot_bytes: bytes | None = None
    snapshot_path = "/data/rail_snapshot.geojson"

    def __init__(self, *args, directory: str, **kwargs) -> None:
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._snapshot_gzip_response = False
        request_path = urllib.parse.urlsplit(self.path).path
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        if request_path == self.snapshot_path and accepts_gzip and self.snapshot_bytes is not None:
            payload = self.snapshot_bytes
            self._snapshot_gzip_response = True
            self.send_response(200)
            self.send_header("Content-Type", "application/geo+json; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def end_headers(self) -> None:
        request_path = urllib.parse.urlsplit(self.path).path
        if not getattr(self, "_snapshot_gzip_response", False) and request_path.endswith((".js", ".css", ".json", ".geojson", ".csv")):
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()


class ReusableThreadingHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the GBA rail map locally")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    snapshot_file = root / "data" / "rail_snapshot.geojson"
    if snapshot_file.exists():
        # Compress once before opening the browser; original GeoJSON remains untouched.
        RailMapHandler.snapshot_bytes = gzip.compress(snapshot_file.read_bytes(), compresslevel=5)

    def handler(*handler_args, **handler_kwargs):
        return RailMapHandler(*handler_args, directory=str(root), **handler_kwargs)

    url = f"http://127.0.0.1:{args.port}/"
    with ReusableThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"湾区轨道地图已启动：{url}")
        print("已启用并行资源加载与 rail_snapshot.geojson 压缩传输。")
        print("按 Ctrl+C 停止。")
        if not args.no_browser:
            threading.Timer(0.2, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")


if __name__ == "__main__":
    main()
