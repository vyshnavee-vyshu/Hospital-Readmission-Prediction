#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10 or later."
  exit 1
fi

python3 --version

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Starting Hospital Readmission Prediction at http://127.0.0.1:8000"
echo "Stop the server with Ctrl+C"
echo
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
