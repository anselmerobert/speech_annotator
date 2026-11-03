import os
import tempfile

import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="Parakeet Transcription")

model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    name = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    model = nemo_asr.models.ASRModel.from_pretrained(name)
    model.change_attention_model("rel_pos_local_attn", [128, 128])
    model.change_subsampling_conv_chunking_factor(1)


@app.get("/health")
def health() -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        output = model.transcribe([tmp.name])
        hyp = output[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        end = 0.0
        if hasattr(hyp, "timestamp") and hyp.timestamp:
            end = float(hyp.timestamp[-1])
        return {"text": text, "start": 0.0, "end": end}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
