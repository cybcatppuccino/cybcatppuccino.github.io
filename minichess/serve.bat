@echo off
cd /d "%~dp0"
echo Gardner MiniChess Lab with COOP/COEP headers: http://127.0.0.1:8000
py tools\serve-coi.py 8000
if errorlevel 1 python tools\serve-coi.py 8000
pause
