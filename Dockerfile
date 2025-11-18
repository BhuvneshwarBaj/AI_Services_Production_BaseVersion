
#Its a dockerfile


# FROM python:3.11-slim

# WORKDIR /app
# ENV PYTHONPATH=/app/src
# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project source
# #COPY src ./src
# COPY src/ /app/src/
# COPY artifacts ./artifacts
# COPY gunicorn.conf.py .
# COPY .env .

# # Expose the port Flask listens on
# EXPOSE 8000

# # Gunicorn entrypoint
# CMD ["gunicorn", "--config", "gunicorn.conf.py", "src.aiservices.wsgi:app"]

# syntax=docker/dockerfile:1
# FROM python:3.11-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app/src

# WORKDIR /app

# # OS deps (libgomp1 needed by xgboost)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       libgomp1 wget ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# # Install Python deps first (better layer cache)
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy project
# COPY gunicorn.conf.py .              
# COPY src/ /app/src/
# # Optional: copy artifacts into image; will be overridden by volume in compose
# COPY artifacts /app/artifacts

# # (You don't need .env inside the image when using compose env_file)
# # COPY .env .

# EXPOSE 8000

# # Healthcheck (optional)
# HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
#   CMD wget -qO- http://127.0.0.1:8000/health || exit 1

# # Gunicorn entrypoint: module path must match PYTHONPATH=/app/src
# CMD ["gunicorn", "--config", "gunicorn.conf.py", "aiservices.wsgi:app"]


FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY gunicorn.conf.py .
COPY src/ /app/src/
COPY artifacts /app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD wget -qO- http://127.0.0.1:8000/health || exit 1

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

CMD ["gunicorn", "--config", "gunicorn.conf.py", "aiservices.wsgi:app"]
