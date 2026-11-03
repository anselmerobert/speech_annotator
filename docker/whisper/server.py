import os
import tempfile

import ctranslate2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper Transcription")

model: WhisperModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    size = os.environ.get("MODEL_SIZE", "large-v3")
    device = os.environ.get("DEVICE", "cuda")
    compute_type = os.environ.get("COMPUTE_TYPE", "float16")

    # Check GPU availability via ctranslate2 (avoids torch dependency).
    if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        print("No CUDA devices found, falling back to CPU")
        device = "cpu"
        compute_type = "int8"

    model = WhisperModel(size, device=device, compute_type=compute_type)


@app.get("/health")
def health() -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str = Form(""),
) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        kwargs = {"initial_prompt": prompt} if prompt else {}
        segments, _info = model.transcribe(tmp.name, **kwargs)
        texts = []
        start = 0.0
        end = 0.0
        first = True
        for seg in segments:
            if first:
                start = float(seg.start)
                first = False
            end = float(seg.end)
            texts.append(seg.text)
        return {"text": "".join(texts).strip(), "start": start, "end": end}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
