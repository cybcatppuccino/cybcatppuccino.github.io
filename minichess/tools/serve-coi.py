#!/usr/bin/env python3
"""Local development server with headers required by pthread wasm engines.

Fairy-Stockfish's wasm build uses SharedArrayBuffer/pthreads. Modern browsers
only expose SharedArrayBuffer to cross-origin-isolated pages, so this server adds
COOP/COEP headers and serves .wasm with application/wasm.
"""
from __future__ import annotations

import functools
import http.server
import pathlib
import socketserver
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.mjs': 'text/javascript',
        '.js': 'text/javascript',
    }

    def end_headers(self) -> None:
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

if __name__ == '__main__':
    handler = functools.partial(CrossOriginIsolatedHandler, directory=str(ROOT))
    with socketserver.ThreadingTCPServer(('127.0.0.1', PORT), handler) as httpd:
        print(f'Gardner MiniChess Lab: http://127.0.0.1:{PORT}')
        print('COOP/COEP headers enabled for Fairy-Stockfish wasm.')
        httpd.serve_forever()
