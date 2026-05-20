import os

from pyspark.sql import SparkSession

POSTGRES_JAR = os.environ.get(
    "POSTGRES_JDBC_JAR", "/opt/spark-jars/postgresql-42.7.4.jar"
)

# getOrCreate() переиспользует старую сессию без jdbc-jar — остановите её
active = SparkSession.getActiveSession()
if active is not None:
    active.stop()

spark = (
    SparkSession.builder
    .appName("jdbc-postgres")
    .config("spark.jars", POSTGRES_JAR)
    .getOrCreate()
)

df = (
    spark.read.format("jdbc")
    .option("url", "jdbc:postgresql://host.docker.internal:5432/postgres")
    .option("dbtable", "spells")
    .option("user", "postgres")
    .option("password", "password")
    .option("driver", "org.postgresql.Driver")
    .load()
)

df.show()
# spark.stop()  # в ноутбуке не останавливайте, если дальше ещё ячейки со Spark
