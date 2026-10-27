#!/usr/bin/env bash
# Start the already-built AI Toolkit WebUI with the repository-local runtime.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
NPM="$SCRIPT_DIR/.node/bin/npm"
UI_DIR="$SCRIPT_DIR/ui"
UI_URL="http://127.0.0.1:8675"

die() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

if [[ ! -x "$PYTHON" ]]; then
    die "Python environment not found at $PYTHON. Create it first with: python3 -m venv --system-site-packages .venv"
fi

if [[ ! -x "$NPM" ]]; then
    die "Repository-local Node.js was not found at $NPM. Provision Node.js v20+ in .node/ before starting the WebUI."
fi

if [[ ! -d "$UI_DIR/node_modules" ]]; then
    die "UI dependencies are missing at $UI_DIR/node_modules. Install the UI dependencies before starting the WebUI."
fi

if "$PYTHON" - "$UI_URL" <<'PY'
import socket
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect((url.hostname, url.port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
then
    printf 'WebUI is already running at %s\n' "$UI_URL"
    exit 0
fi

cd "$UI_DIR"
export PATH="$SCRIPT_DIR/.node/bin:$PATH"

if [[ ! -d "$UI_DIR/.next" || ! -d "$UI_DIR/dist" ]]; then
    printf 'WebUI build is missing; running the project manager first...\n'
    exec "$PYTHON" -m manager launch --no-browser
fi

printf 'Starting WebUI at %s\n' "$UI_URL"
printf 'Press Ctrl+C to stop it.\n'
exec "$NPM" run start
