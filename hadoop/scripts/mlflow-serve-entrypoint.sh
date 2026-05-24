#!/bin/bash
set -euo pipefail

MODEL_NAME="${MLFLOW_MODEL_NAME:-spells-classifier}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
PORT="${MLFLOW_SERVE_PORT:-5001}"
MAX_WAIT_SEC="${MLFLOW_SERVE_WAIT_SEC:-300}"

export MLFLOW_TRACKING_URI="$TRACKING_URI"

echo "MLflow tracking: $TRACKING_URI"
echo "Waiting for tracking server..."
python3 <<'PY'
import os
import time
import urllib.request

uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(uri, timeout=3) as resp:
            if resp.status == 200:
                break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit("MLflow tracking server not reachable")
PY

echo "Resolving model ${MODEL_NAME} (Production, else Staging)..."
MODEL_URI=$(python3 <<'PY'
import os
import sys
from mlflow.tracking import MlflowClient

name = os.environ["MLFLOW_MODEL_NAME"]
client = MlflowClient()
for stage in ("Production", "Staging"):
    versions = client.get_latest_versions(name, stages=[stage])
    if versions:
        print(f"models:/{name}/{stage}")
        sys.exit(0)
print(f"No Production/Staging version for {name}", file=sys.stderr)
sys.exit(1)
PY
)

echo "Starting model server: $MODEL_URI on port $PORT"
exec mlflow models serve \
  -m "$MODEL_URI" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --env-manager local \
  --no-conda
