# Single image: FastAPI serves the API and the static vanilla-JS frontend. No Node stage needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin mlhelp

COPY backend/app ./app
COPY frontend ./frontend

# Warm the sklearn sample-dataset cache so the lessons never touch the network at runtime.
RUN python -c "from sklearn import datasets as d; d.load_iris(); d.load_wine(); d.load_breast_cancer(); d.load_digits(); d.load_diabetes()"

ENV ML_DATA_DIR=/data ML_STATIC_DIR=/app/frontend
RUN mkdir -p /data && chown mlhelp:mlhelp /data
USER mlhelp
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1", "--no-access-log", "--timeout-keep-alive", "15"]
