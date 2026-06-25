#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
echo "Gardner MiniChess Lab: http://localhost:8000"
python3 -m http.server 8000
