FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir faster-whisper fastapi uvicorn[standard] python-multipart

# Create a non-root user so the model cache volume is not owned by root.
# Defaults to 1000:1000 — replace with your own UID:GID if different (see README).
RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 -s /bin/bash app
ENV HOME=/home/app
ENV HF_HOME=/home/app/.cache/huggingface
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

COPY --chown=app:app server.py /app/server.py
WORKDIR /app
ENV MODEL_SIZE=large-v3
ENV COMPUTE_TYPE=float16
EXPOSE 8002
USER app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
