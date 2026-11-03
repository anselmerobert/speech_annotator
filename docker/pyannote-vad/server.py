import os
import tempfile

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pyannote.audio import Pipeline

app = FastAPI(title="Pyannote VAD")

pipeline: Pipeline | None = None


@app.on_event("startup")
def load_model() -> None:
    global pipeline
    token = os.environ.get("HF_TOKEN") or None
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=token,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)


@app.get("/health")
def health() -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/vad")
async def vad(file: UploadFile = File(...)) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        output = pipeline(tmp.name)
        segments = [
            {"start": float(seg.start), "end": float(seg.end)}
            for seg in output.get_timeline().support()
        ]
        return {"segments": segments}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
