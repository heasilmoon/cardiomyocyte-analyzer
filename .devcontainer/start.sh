#!/bin/bash
# Started automatically each time a Codespace is (re)attached: runs the
# server in the background on port 8000, which Codespaces forwards and
# opens in the browser (see devcontainer.json).
cd "$(dirname "$0")/../backend" || exit 1
if pgrep -f "uvicorn app.main:app" >/dev/null; then
  echo "Server already running on port 8000."
  exit 0
fi
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/cardiomyocyte-analyzer.log 2>&1 &
echo "Server starting on port 8000 (log: /tmp/cardiomyocyte-analyzer.log)."
