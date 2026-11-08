FROM apache/airflow:2.6.1-python3.9

USER root

ENV DEBIAN_FRONTEND=noninteractive

# Java 17 for Spark, procps for 'ps', libgomp1 (OpenMP) for XGBoost.
# Debian 11 (bullseye) is EOL: drop the PostgreSQL repo (removed upstream)
# and point apt at archive.debian.org, where bullseye packages now live
RUN rm -f /etc/apt/sources.list.d/pgdg.list && \
    sed -i -e 's|http://deb.debian.org|http://archive.debian.org|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    # Spark's scripts expect bash, not dash
    ln -sf /bin/bash /bin/sh && \
    # JAVA_HOME below is the amd64 path; on arm64 hosts (Apple Silicon) create it as a link
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    ( [ -e /usr/lib/jvm/java-17-openjdk-amd64/bin/java ] || \
      ln -s "$(readlink -f "$(which java)")" /usr/lib/jvm/java-17-openjdk-amd64/bin/java )

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

COPY requirements.txt /requirements.txt

# Python packages go to the airflow user's site-packages, like the base image's own packages
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt
