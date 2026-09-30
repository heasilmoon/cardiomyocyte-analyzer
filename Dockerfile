# Cardiomyocyte Analyzer — container image (also used by Hugging Face Spaces).
#
# Hugging Face Spaces (Docker SDK) builds this file from the repository root,
# runs the container as user 1000 and expects the app on port 7860 (see the
# `app_port` in README.md's front matter). Any other host can use it too:
#   docker build -t cardiomyocyte-analyzer .
#   docker run -p 7860:7860 cardiomyocyte-analyzer
FROM python:3.11-slim

# libgl1/libglib2.0-0 are runtime deps of opencv-python-headless for video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY frontend /frontend

RUN mkdir -p storage/uploads storage/results \
    && chown -R appuser:appuser /app /frontend
USER appuser

ENV PORT=7860
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
