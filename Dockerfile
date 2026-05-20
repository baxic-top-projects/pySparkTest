FROM eclipse-temurin:17-jre-jammy

RUN mkdir -p /opt/spark-jars \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && curl -fsSL -o /opt/spark-jars/postgresql-42.7.4.jar \
        https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pyspark==3.5.4 \
    jupyterlab==4.3.4 \
    pandas \
    matplotlib \
    scikit-learn

ENV JAVA_HOME=/opt/java/openjdk
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JUPYTER_ENABLE_LAB=yes
ENV POSTGRES_JDBC_JAR=/opt/spark-jars/postgresql-42.7.4.jar
ENV PYSPARK_SUBMIT_ARGS="--jars /opt/spark-jars/postgresql-42.7.4.jar pyspark-shell"

RUN useradd -m -s /bin/bash -u 1000 jovyan
USER jovyan
WORKDIR /home/jovyan/work

EXPOSE 8888 4040

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
