FROM nvcr.io/nvidia/pytorch:25.11-py3
RUN pip install --no-cache-dir "nemo_toolkit[asr]>=2.1.0" fastapi uvicorn[standard] python-multipart librosa soundfile

# Create a non-root user so the model cache volume is not owned by root.
# Defaults to 1000:1000 — replace with your own UID:GID if different (see README).
RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 -s /bin/bash app
ENV HOME=/home/app
ENV HF_HOME=/home/app/.cache/huggingface
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

COPY --chown=app:app server.py /app/server.py
WORKDIR /app
ENV PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3
EXPOSE 8003
USER app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
