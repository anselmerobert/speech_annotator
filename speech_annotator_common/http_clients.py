"""Async HTTP clients for VAD, transcription, and OCR services."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from speech_annotator_common.models import Segment

_TIMEOUT = httpx.Timeout(300.0)
_MISTRAL_TRANSCRIPTION_URL = "https://api.mistral.ai/v1/audio/transcriptions"
_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
_MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"


class VadClient:
    """Client for a Voice Activity Detection service."""

    def __init__(self, name: str, endpoint: str) -> None:
        self.name = name
        self.endpoint = endpoint.rstrip("/")

    async def health_check(self) -> bool:
        """GET /health; return True if status 200."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.get(f"{self.endpoint}/health")
            except httpx.RequestError:
                return False
            return resp.status_code == 200

    async def detect_segments(self, file_path: Path) -> list[Segment]:
        """POST /vad with the file; parse JSON response into Segments.

        Expected response shape: a list of objects with 'start' and 'end' keys
        (both floats in seconds). An optional 'text' key is ignored.
        """
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(f"{self.endpoint}/vad", files=files)

        if resp.status_code != 200:
            raise RuntimeError(
                f"VAD '{self.name}' at {self.endpoint}/vad failed "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        data = resp.json()
        # Support both {"segments": [...]} and bare [...] response shapes.
        if isinstance(data, dict) and "segments" in data:
            data = data["segments"]
        if not isinstance(data, list):
            raise RuntimeError(
                f"VAD '{self.name}' returned unexpected response: {data!r}"
            )
        segments: list[Segment] = []
        for item in data:
            if not isinstance(item, dict) or "start" not in item or "end" not in item:
                raise RuntimeError(
                    f"VAD '{self.name}' returned malformed segment: {item!r}"
                )
            segments.append(Segment(start=float(item["start"]), end=float(item["end"])))
        return segments


class TranscriptionClient:
    """Client for a local Docker transcription service."""

    def __init__(self, name: str, endpoint: str) -> None:
        self.name = name
        self.endpoint = endpoint.rstrip("/")

    async def health_check(self) -> bool:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.get(f"{self.endpoint}/health")
            except httpx.RequestError:
                return False
            return resp.status_code == 200

    async def transcribe(self, file_path: Path, prompt: str = "") -> str:
        """POST /transcribe with the file; return the transcribed text.

        Args:
            file_path: path to the audio segment WAV file.
            prompt: optional initial prompt for whisper (ignored by
                servers that don't support it).
        """
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data: dict[str, str] = {}
            if prompt:
                data["prompt"] = prompt
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.endpoint}/transcribe", files=files, data=data)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Transcription '{self.name}' at {self.endpoint}/transcribe failed "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        data = resp.json()
        if not isinstance(data, dict) or "text" not in data:
            raise RuntimeError(
                f"Transcription '{self.name}' returned malformed response: {data!r}"
            )
        return str(data["text"])


class MistralTranscriptionClient:
    """Client for the Mistral web API transcription service (Voxtral)."""

    def __init__(self, name: str, api_key: str, model: str) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model

    async def transcribe(self, file_path: Path, prompt: str = "") -> str:
        """POST /v1/audio/transcriptions with multipart file upload.

        Returns the transcribed text from the Voxtral model.
        The prompt is expected to contain space-separated context_bias
        terms (pre-selected by an LLM from slide OCR text). These are
        sent as context_bias to the Voxtral API (max 100 single tokens).
        """
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "audio/wav")}
            data: dict[str, str | list[str]] = {"model": self._model}
            if prompt:
                tokens = [w for w in prompt.split() if w][:100]
                if tokens:
                    data["context_bias"] = tokens
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            }
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _MISTRAL_TRANSCRIPTION_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Mistral transcription '{self.name}' failed "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        result = resp.json()
        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError(
                f"Mistral transcription '{self.name}' returned malformed response: {result!r}"
            )
        return str(result["text"])


class MistralOcrClient:
    """Client for the Mistral OCR API (slide text extraction)."""

    def __init__(self, api_key: str, model: str = "mistral-ocr-latest") -> None:
        self._api_key = api_key
        self._model = model

    async def ocr_image(self, image_path: Path) -> str:
        """POST /v1/ocr with a base64-encoded image; return Markdown text.

        Args:
            image_path: path to a PNG/JPEG image (e.g. a video frame).

        Returns:
            Extracted text in Markdown format.
        """
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        payload = {
            "model": self._model,
            "document": {
                "type": "image_url",
                "image_url": data_url,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(_MISTRAL_OCR_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Mistral OCR failed (HTTP {resp.status_code}): {resp.text}"
            )
        result = resp.json()
        pages = result.get("pages", [])
        if not pages:
            return ""
        return str(pages[0].get("markdown", ""))


class MistralVisionClient:
    """Client for Mistral vision chat (slide detection)."""

    def __init__(self, api_key: str, model: str = "pixtral-12b-latest") -> None:
        self._api_key = api_key
        self._model = model

    async def detect_slides(self, image_paths: list[Path]) -> bool:
        """Send sample frames to the vision model and ask if they look like
        presentation slides.

        Args:
            image_paths: list of frame image paths to examine.

        Returns:
            True if the majority of frames look like slides.
        """
        if not image_paths:
            return False

        content: list[dict] = [
            {"type": "text",
             "text": "Do these images show presentation slides (e.g. PowerPoint, "
                     "Keynote, Beamer)? Answer with a single word: 'yes' or 'no'."},
        ]
        for img_path in image_paths:
            image_bytes = img_path.read_bytes()
            suffix = img_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": f"data:{mime};base64,{b64}",
            })

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 10,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                _MISTRAL_CHAT_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Mistral Vision failed (HTTP {resp.status_code}): {resp.text}"
            )
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip().lower()
        return answer.startswith("yes")
