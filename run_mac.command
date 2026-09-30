#!/bin/bash
# Cardiomyocyte Analyzer — one-click start for macOS (double-click this file).
# First run: creates a Python virtual environment and installs dependencies.
# Every run: pulls the latest code (if this is a git checkout), then starts
# the server and opens http://localhost:8000 in the browser.
# Stop the server with Ctrl+C in this window (or just close the window).
set -e
cd "$(dirname "$0")"

if command -v git >/dev/null 2>&1 && [ -d .git ]; then
  echo "[1/4] 최신 코드 받는 중 (git pull)..."
  git pull --ff-only || echo "  (git pull 실패 — 인터넷 연결이나 로컬 변경을 확인하세요. 기존 코드로 계속합니다.)"
fi

cd backend
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
  echo "python3 가 없습니다. https://www.python.org/downloads/ 에서 Python 3.11 이상을 설치한 뒤 다시 실행하세요."
  read -r -p "Enter를 누르면 닫힙니다." _
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[2/4] 가상환경 만드는 중 (최초 1회)..."
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/4] 필요한 패키지 확인/설치 중..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "[4/4] 서버 시작: http://localhost:8000  (끝내려면 이 창에서 Ctrl+C)"
( sleep 3; open "http://localhost:8000" ) &
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
