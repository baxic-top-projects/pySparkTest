"""Hive: лёгкая проверка портов + чтение Parquet той же SparkSession (без PyHive)."""
from __future__ import annotations

import os
import socket
import time


def build_hive_spark(*_args, **_kwargs):
    raise RuntimeError("build_hive_spark удалён. Restart Kernel → ячейка 1 → ячейка 2.")


def wait_hive_ports(max_wait_sec: int = 45) -> None:
    checks = [
        ("metastore", "hive-metastore", 9083),
        (
            "hiveserver2",
            os.environ.get("HIVE_SERVER_HOST", "hive-server"),
            int(os.environ.get("HIVE_SERVER_PORT", "10000")),
        ),
    ]
    for name, host, port in checks:
        print(f"Ожидание {name} {host}:{port} ...")
        ok = False
        for _ in range(0, max_wait_sec, 2):
            try:
                s = socket.create_connection((host, port), timeout=2)
                s.close()
                print(f"  {name} доступен.")
                ok = True
                break
            except OSError:
                time.sleep(2)
        if not ok:
            raise RuntimeError(f"{name} недоступен. docker compose ps")


wait_hive_server = wait_hive_ports


def read_spells_parquet(spark, hdfs_path: str):
    from spark_utils import require_spark

    return require_spark(spark).read.parquet(hdfs_path)


def hive_register_command(
    hdfs_path: str,
    hive_db: str = "pysparktest",
    hive_table: str = "spells",
) -> str:
    """Команда для регистрации таблицы на хосте (после ячейки 1)."""
    return (
        f'docker exec hive-server bash /scripts/register-spells-table.sh '
        f'-c "HIVE_DB={hive_db} HIVE_TABLE={hive_table} HDFS_PATH={hdfs_path}"'
    )
