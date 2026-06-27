#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
echo "Gardner MiniChess Lab with COOP/COEP headers: http://127.0.0.1:8000"
python3 tools/serve-coi.py 8000
