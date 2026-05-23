#!/usr/bin/env bash
# Однократно при docker compose up: дождаться NameNode и выключить safe mode.
set -euo pipefail

echo "=== HDFS init: ожидание NameNode ==="
for _ in $(seq 1 90); do
  if hdfs dfsadmin -safemode get >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "=== HDFS init: safemode leave (dev) ==="
for _ in $(seq 1 40); do
  if hdfs dfsadmin -safemode get 2>&1 | grep -qi "OFF"; then
    echo "Safe mode is OFF"
    hdfs dfsadmin -report 2>/dev/null | grep -E "Live datanodes|Safe mode" || true
    exit 0
  fi
  hdfs dfsadmin -safemode leave >/dev/null 2>&1 || true
  sleep 3
done

echo "ERROR: safe mode не выключился за отведённое время" >&2
hdfs dfsadmin -safemode get >&2 || true
exit 1
