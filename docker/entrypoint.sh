#!/bin/sh
set -e

host="${MYSQL_HOST:-db}"
port="${MYSQL_PORT:-3306}"

echo "Waiting for MySQL at ${host}:${port}..."
python - <<'PY'
import os, socket, time, sys

host = os.environ.get("MYSQL_HOST", "db")
port = int(os.environ.get("MYSQL_PORT", "3306"))

for attempt in range(60):
    try:
        with socket.create_connection((host, port), timeout=2):
            print("MySQL is ready")
            sys.exit(0)
    except OSError:
        time.sleep(2)

print("Timed out waiting for MySQL", file=sys.stderr)
sys.exit(1)
PY

python -c "from app import init_db; init_db()"

exec gunicorn app:app \
  --bind 0.0.0.0:5000 \
  --workers "${GUNICORN_WORKERS:-1}" \
  --timeout 300
