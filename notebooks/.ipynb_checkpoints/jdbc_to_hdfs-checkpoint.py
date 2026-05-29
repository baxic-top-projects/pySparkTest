"""
PostgreSQL (как в 43_spell_classifiers) → Parquet в HDFS.
Запуск: docker compose up -d, затем выполнить в Jupyter / DataSpell.
"""
import os

from hdfs_utils import write_parquet_hdfs
from spark_utils import create_spark_session, enable_hdfs

POSTGRES_JAR = os.environ.get(
    "POSTGRES_JDBC_JAR", "/opt/spark-jars/postgresql-42.7.4.jar"
)
HDFS_URL = os.environ.get("HDFS_URL", "hdfs://namenode:9000")
HDFS_PATH = f"{HDFS_URL}/data/spells"
POSTGRES_JDBC_URL = os.environ.get(
    "POSTGRES_JDBC_URL",
    "jdbc:postgresql://host.docker.internal:5432/postgres",
)

spark = create_spark_session(
    app_name="jdbc-to-hdfs",
    hdfs_url=HDFS_URL,
    postgres_jar=POSTGRES_JAR,
    driver_memory=os.environ.get("SPARK_DRIVER_MEMORY", "1g"),
)

print("=== Чтение из PostgreSQL ===")
df = (
    spark.read.format("jdbc")
    .option("url", POSTGRES_JDBC_URL)
    .option("dbtable", "spells")
    .option("user", "postgres")
    .option("password", "password")
    .option("driver", "org.postgresql.Driver")
    .option("numPartitions", "1")
    .option("connectTimeout", "10")
    .option("socketTimeout", "30")
    .load()
)

df.show()

enable_hdfs(spark, HDFS_URL)
print(f"=== Запись в HDFS: {HDFS_PATH} ===")
write_parquet_hdfs(df, HDFS_PATH, spark, HDFS_URL)

print("=== Проверка чтения из HDFS ===")
spark.read.parquet(HDFS_PATH).show()

print("Готово. В NameNode UI: Browse → /data/spells")
