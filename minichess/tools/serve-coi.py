#!/usr/bin/env python3
"""Tiny static HTTP server with COOP/COEP headers for Fairy-Stockfish.

Use this server instead of `python -m http.server` when testing the optional
Fairy-Stockfish pthread wasm kernel.  Orion JS does not require these headers,
but browsers require them before SharedArrayBuffer is exposed to wasm threads.
"""
from __future__ import annotations

import http.server
import mimetypes
import pathlib
import socketserver
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('text/javascript', '.mjs')
mimetypes.add_type('text/javascript', '.js')


class CoiRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.mjs': 'text/javascript',
        '.js': 'text/javascript',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == '__main__':
    with ReusableTCPServer(('127.0.0.1', PORT), CoiRequestHandler) as httpd:
        print(f'Gardner MiniChess Lab with COOP/COEP headers: http://127.0.0.1:{PORT}')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nServer stopped.')
