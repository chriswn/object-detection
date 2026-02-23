#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 not found."

  if command -v apt-get >/dev/null 2>&1; then
    if [ "${AUTO_INSTALL:-0}" = "1" ]; then
      sudo apt-get update
      sudo apt-get install -y python3.11 python3.11-venv
    else
      echo "Ubuntu/Debian detected. Run: sudo apt-get install python3.11 python3.11-venv"
      echo "Or re-run with AUTO_INSTALL=1 to install automatically."
      exit 1
    fi
  elif command -v dnf >/dev/null 2>&1; then
    if [ "${AUTO_INSTALL:-0}" = "1" ]; then
      sudo dnf install -y python3.11 python3.11-venv
    else
      echo "Fedora detected. Run: sudo dnf install python3.11 python3.11-venv"
      echo "Or re-run with AUTO_INSTALL=1 to install automatically."
      exit 1
    fi
  else
    echo "Install Python 3.11 manually, then re-run this script."
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
  echo "Removing existing .venv"
  rm -rf .venv
fi

python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
