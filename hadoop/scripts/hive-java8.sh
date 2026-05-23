#!/usr/bin/env bash
# Hive CLI: Java 8 + Guava из Hadoop 3.2 (иначе NoSuchMethodError в Preconditions)
set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-/opt/java-8}"
export HADOOP_HOME="${HADOOP_HOME:-/opt/hadoop-3.2.1}"
export HIVE_HOME="${HIVE_HOME:-/opt/hive}"
export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-/etc/hadoop}"

GUAVA_JAR="$(ls "${HADOOP_HOME}"/share/hadoop/common/lib/guava-*.jar 2>/dev/null | head -1)"
if [[ -n "${GUAVA_JAR}" ]]; then
  export HADOOP_CLASSPATH="${GUAVA_JAR}${HADOOP_CLASSPATH:+:${HADOOP_CLASSPATH}}"
fi

export PATH="${JAVA_HOME}/bin:${HADOOP_HOME}/bin:${HIVE_HOME}/bin:${PATH}"

exec "${HIVE_HOME}/bin/hive" "$@"
