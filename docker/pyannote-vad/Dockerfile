FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pyannote.audio==3.3.2" "torch>=2.0,<2.6" "torchaudio>=2.0,<2.6" "huggingface_hub<1.0" omegaconf pytorch-lightning matplotlib fastapi uvicorn[standard] python-multipart

# Create a non-root user so the model cache volume is not owned by root.
# Defaults to 1000:1000 — replace with your own UID:GID if different (see README).
RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 -s /bin/bash app
ENV HOME=/home/app
ENV HF_HOME=/home/app/.cache/huggingface
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

COPY --chown=app:app server.py /app/server.py
WORKDIR /app
ENV HF_TOKEN=""
EXPOSE 8001
USER app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
