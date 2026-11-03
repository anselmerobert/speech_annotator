"""LLM-based correction propagation (GUI-only)."""

import asyncio
import json

import httpx

from speech_annotator_common.config import LlmConfig, LlmSettings

# Retries for transient connection errors (server unreachable): fail fast.
_CONNECT_RETRIES = 3
_CONNECT_BACKOFF_SECONDS = 5.0

# Retries for transient HTTP errors (502/503/504, e.g. "all backends
# down" while the LLM backend restarts): backend cooldowns commonly last
# a minute or more, so back off exponentially and be patient.
_HTTP_RETRIES = 5
_HTTP_BACKOFF_SECONDS = (15.0, 30.0, 60.0, 120.0)

# Total time allowed for one propagate request: generating corrections
# for ~1000 segments takes minutes.
_REQUEST_TIMEOUT = 300.0

_RETRYABLE_STATUS = {502, 503, 504}


def build_propagate_prompt(
    corrections: list[tuple[str, str]],
    texts_to_fix: list[str],
) -> str:
    """Build the LLM prompt for propagating systematic corrections.

    Args:
        corrections: list of (before, after) pairs from validated segments
            where the user corrected the merged text.
        texts_to_fix: list of merged texts for unvalidated segments to correct.

    Returns:
        JSON array of corrected texts, same length and order as texts_to_fix.
    """
    lines = [
        "You are a speech transcription correction propagator.",
        "The user has manually corrected several transcriptions. Your task is to",
        "identify the systematic corrections (e.g. spelling, punctuation, word",
        "substitutions, capitalization patterns) and apply them to new transcriptions.",
        "",
        "Here are the corrections the user made (before -> after):",
    ]
    for i, (before, after) in enumerate(corrections):
        lines.append(f"  Correction {i + 1}:")
        lines.append(f"    Before: \"{before}\"")
        lines.append(f"    After:  \"{after}\"")
    lines.append("")
    lines.append(f"Now apply the same systematic corrections to these {len(texts_to_fix)} transcriptions.")
    lines.append("Only apply corrections that match the patterns above. Do not invent new corrections.")
    lines.append("Output a JSON array of strings, one per transcription, in the same order.")
    lines.append("Do not output anything except the JSON array.")
    lines.append("")
    for i, text in enumerate(texts_to_fix):
        lines.append(f"[{i}] \"{text}\"")
    return "\n".join(lines)


async def _post_chat_completion(
    client: httpx.AsyncClient,
    settings: LlmSettings,
    payload: dict,
    on_retry=None,
) -> httpx.Response:
    """POST one chat completion, retrying transient errors.

    Connection errors fail fast after a few short retries. HTTP
    502/503/504 (e.g. "all backends down" while the LLM backend
    restarts) are retried with exponential backoff, since backend
    cooldowns commonly last a minute or more. Other HTTP errors fail
    immediately.

    on_retry, if given, is called with a short status message before
    each retry wait (used to update the GUI status bar).
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    url = f"{settings.base_url}/chat/completions"
    connect_attempts = 0
    http_attempts = 0
    while True:
        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.TransportError as exc:
            connect_attempts += 1
            if connect_attempts >= _CONNECT_RETRIES:
                raise RuntimeError(
                    f"Cannot reach the LLM server at {settings.base_url} "
                    f"after {connect_attempts} attempts ({exc}). "
                    "Check that the server is running."
                ) from exc
            wait = _CONNECT_BACKOFF_SECONDS * connect_attempts
            msg = f"LLM connection error ({exc}), retry {connect_attempts}/{_CONNECT_RETRIES - 1} in {wait:.0f}s"
        else:
            if response.status_code == 200:
                return response
            if response.status_code not in _RETRYABLE_STATUS:
                raise RuntimeError(
                    f"LLM request failed (HTTP {response.status_code}): {response.text}"
                )
            http_attempts += 1
            total_wait = sum(_HTTP_BACKOFF_SECONDS[:http_attempts])
            if http_attempts >= _HTTP_RETRIES:
                raise RuntimeError(
                    f"LLM request failed (HTTP {response.status_code}) after "
                    f"{http_attempts} attempts over {total_wait:.0f}s of retries: "
                    f"{response.text}The LLM backend was down or restarting "
                    "the whole time; try again in a few minutes."
                )
            wait = _HTTP_BACKOFF_SECONDS[min(http_attempts - 1, len(_HTTP_BACKOFF_SECONDS) - 1)]
            msg = (f"LLM transient error (HTTP {response.status_code}), retry "
                   f"{http_attempts}/{_HTTP_RETRIES - 1} in {wait:.0f}s")
        print(msg)
        if on_retry is not None:
            on_retry(msg)
        await asyncio.sleep(wait)


async def propagate_corrections(
    config: LlmConfig,
    corrections: list[tuple[str, str]],
    texts_to_fix: list[str],
    on_retry=None,
) -> list[str]:
    """Call the LLM to propagate user corrections to unvalidated segments.

    Transient errors (connection errors, HTTP 502/503/504) are retried
    with backoff; see _post_chat_completion. on_retry, if given, is
    called with a short status message before each retry wait.

    Args:
        config: LLM configuration
        corrections: list of (merged_text, validated_text) from validated segments
        texts_to_fix: list of merged texts for unvalidated segments
        on_retry: optional callback(str) for retry status updates

    Returns:
        list of corrected texts, same length and order as texts_to_fix.
    """
    prompt = build_propagate_prompt(corrections, texts_to_fix)
    settings = config.llm

    estimated_tokens = sum(len(t) // 4 + 20 for t in texts_to_fix) + len(prompt) // 4 + 200
    max_tokens = max(4096, min(estimated_tokens, 32000))

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        response = await _post_chat_completion(client, settings, {
            "model": settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }, on_retry=on_retry)
        data = response.json()
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        content = (choice["message"].get("content") or "").strip()
        if not content:
            raise RuntimeError(
                f"LLM returned no content (finish_reason={finish_reason}). "
                "The model may have exhausted the token budget on reasoning."
            )
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            if finish_reason == "length":
                raise RuntimeError(
                    f"LLM output was truncated (max_tokens={max_tokens}). "
                    f"Too many segments ({len(texts_to_fix)}) to process in one request. "
                    f"Try validating more segments first to reduce the number of unvalidated ones."
                ) from None
            raise
        if not isinstance(result, list) or len(result) != len(texts_to_fix):
            raise RuntimeError(
                f"LLM returned unexpected format: expected list of {len(texts_to_fix)} strings, "
                f"got {type(result).__name__} of length {len(result) if hasattr(result, '__len__') else '?'}"
            )
        return [str(item) for item in result]
