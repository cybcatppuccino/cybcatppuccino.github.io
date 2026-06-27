@echo off
cd /d %~dp0
echo Gardner MiniChess Lab v16 with COOP/COEP headers: http://127.0.0.1:8000
python tools\serve-coi.py 8000
