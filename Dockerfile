# =============================================================================
# Stage 1 — Builder
# Compiles dlib + installs all Python packages.
# Heavy build tools (cmake, g++, headers) stay in this stage only.
# =============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================================
# Stage 2 — Runtime
# NOTE: Java, Spark, and Hadoop are NOT installed here.
#       They are bind-mounted from the host at runtime (see docker-compose.yml).
#       This keeps the image lean and ensures the container uses the exact same
#       JDK / Spark / Hadoop versions that the host cluster is configured for.
# =============================================================================
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    # --- Kerberos (runtime auth against Hive / HDFS) ---
    krb5-user \
    libkrb5-3 \
    libkrb5-dev \
    # --- cron (kinit refresh every 6 h, matches old setup) ---
    cron \
    # --- OpenCV / dlib runtime shared libs ---
    libopenblas0 \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libx11-6 \
    # --- misc ---
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Copy compiled Python packages from builder ────────────────────────────────
COPY --from=builder /install /usr/local

# ── Environment variables ─────────────────────────────────────────────────────
# JAVA_HOME / SPARK_HOME point at host-mounted paths (same as old Dockerfile).
# PYTHONPATH mirrors what the old setup exported so PySpark can find py4j.
ENV JAVA_HOME=/usr/java/jdk1.8.0_202-amd64 \
    SPARK_HOME=/usr/yava/3.1.0.0-0000/spark2 \
    HADOOP_CONF_DIR=/etc/hadoop/conf \
    ARROW_LIBHDFS_DIR=/usr/yava/3.1.0.0-0000/hadoop/lib/native \
    PYTHONPATH=/usr/yava/3.1.0.0-0000/spark2/python:/usr/yava/3.1.0.0-0000/spark2/python/lib/py4j-0.10.9-src.zip \
    PATH=/usr/java/jdk1.8.0_202-amd64/bin:/usr/yava/3.1.0.0-0000/spark2/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYSPARK_PYTHON=python3 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Download ML models (spaCy + NLTK) ────────────────────────────────────────
RUN python -m spacy download en_core_web_sm --no-cache-dir && \
    python -c "\
import nltk, os; \
d = '/app/ml_models/nltk_data'; \
os.makedirs(d, exist_ok=True); \
nltk.download('stopwords', download_dir=d, quiet=True); \
nltk.download('wordnet',   download_dir=d, quiet=True); \
nltk.download('punkt',     download_dir=d, quiet=True); \
nltk.download('punkt_tab', download_dir=d, quiet=True); \
"

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Entrypoint: kinit cron + uvicorn ─────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p /tmp/dataprep /app/ml_models /var/log && \
    touch /var/log/kinit.log

EXPOSE 3304

ENTRYPOINT ["/entrypoint.sh"]