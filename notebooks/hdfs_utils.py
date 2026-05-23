"""Ожидание HDFS и запись Parquet (только через Spark JVM, без hdfs CLI)."""
from __future__ import annotations

import time


def wait_hdfs_cluster(spark_session, hdfs_url: str, max_wait_sec: int = 60) -> None:
    """NameNode в safe mode — overwrite падает. Ждём только через Spark/Hadoop API."""
    jvm = spark_session._jvm
    conf = spark_session._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI.create(hdfs_url)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, conf)

    print("Ожидание готовности HDFS (safe mode)...")
    for elapsed in range(0, max_wait_sec, 5):
        if not fs.isInSafeMode():
            print("HDFS готов к записи.")
            return
        print(f"  safe mode ({elapsed}s)...")
        time.sleep(5)

    raise RuntimeError(
        "HDFS в safe mode. Подождите namenode (healthy) или на хосте:\n"
        "  docker exec hadoop-namenode hdfs dfsadmin -safemode leave"
    )


def write_parquet_hdfs(
    df,
    path: str,
    spark_session,
    hdfs_url: str,
    *,
    coalesce: int | None = None,
    max_attempts: int = 3,
) -> None:
    """Запись Parquet; при SafeModeException — пауза и повтор (без отдельного Java-процесса)."""
    writer = df.coalesce(coalesce) if coalesce else df

    for attempt in range(1, max_attempts + 1):
        wait_hdfs_cluster(spark_session, hdfs_url, max_wait_sec=45)
        try:
            writer.write.mode("overwrite").parquet(path)
            print("Запись в HDFS завершена.")
            return
        except Exception as exc:
            err = str(exc)
            if "SafeModeException" not in err and "safe mode" not in err.lower():
                raise
            print(f"  попытка {attempt}/{max_attempts}: safe mode, ждём 15 с...")
            time.sleep(15)

    raise RuntimeError(f"Не удалось записать Parquet в {path} (safe mode)")
