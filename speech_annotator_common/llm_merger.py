"""LLM-based annotation merging and confidence scoring."""

import asyncio
import itertools

import httpx

from speech_annotator_common.config import LlmConfig, LlmSettings
from speech_annotator_common.models import Segment

# Hard cap on the merge request token budget (reasoning models may need
# several retries at doubled budgets before emitting content).
_MERGE_MAX_TOKENS = 8192

# Retries and backoff for transient LLM connection errors.
_CONNECT_RETRIES = 3
_CONNECT_BACKOFF_SECONDS = 5.0

# Abort the merge after this many consecutive failed segments: the LLM server
# is presumably down, and aborting before the fusion SRT is written allows
# the run to be restarted once the server is back.
_MAX_CONSECUTIVE_FAILURES = 3


def compute_confidence(transcriptions: list[str]) -> int:
    """Compute 0-100 confidence score based on pairwise word agreement.

    Uses normalized word-level Levenshtein distance averaged across all pairs.
    Identical transcriptions → 100. Completely different → 0.
    """
    if len(transcriptions) < 2:
        return 50

    scores: list[float] = []
    for i, j in itertools.combinations(range(len(transcriptions)), 2):
        words_a = transcriptions[i].split()
        words_b = transcriptions[j].split()
        dist = _word_levenshtein(words_a, words_b)
        max_len = max(len(words_a), len(words_b))
        if max_len == 0:
            scores.append(100.0)
        else:
            scores.append(100.0 * (1.0 - dist / max_len))

    return int(sum(scores) / len(scores))


def _word_levenshtein(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    if not a and not b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def build_merge_prompt(
    transcriptions: list[str],
    slide_context: str = "",
) -> str:
    """Build the LLM prompt for merging transcriptions of one segment.

    Args:
        transcriptions: list of transcription texts from different systems.
        slide_context: optional OCR text from presentation slides shown
            during this segment, used to guide the merge.
    """
    labels = [chr(ord("A") + i) for i in range(len(transcriptions))]
    lines = [
        "You are merging multiple speech transcriptions of the same audio segment.",
        "Choose the best transcription by combining the most accurate words from each source.",
        "Only use words that appear in the provided transcriptions. Do not add new words.",
        "",
    ]
    if slide_context:
        lines.append("The speaker is presenting slides. Here is the OCR text from the slide")
        lines.append("shown during this segment, which may help disambiguate terms:")
        lines.append(f"--- SLIDE CONTEXT ---")
        lines.append(slide_context)
        lines.append("--- END SLIDE CONTEXT ---")
        lines.append("")
    for label, text in zip(labels, transcriptions, strict=True):
        lines.append(f"Source {label}: \"{text}\"")
    lines.append("")
    lines.append("Output the merged transcription as plain text, nothing else.")
    return "\n".join(lines)


async def _post_chat_completion(
    client: httpx.AsyncClient,
    settings: LlmSettings,
    prompt: str,
    max_tokens: int,
) -> httpx.Response:
    """POST one chat completion, retrying transient connection errors.

    Connection errors are retried with backoff; once the retries are
    exhausted, a RuntimeError with an actionable message is raised.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    url = f"{settings.base_url}/chat/completions"
    for attempt in range(1, _CONNECT_RETRIES + 1):
        try:
            return await client.post(url, headers=headers, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if attempt >= _CONNECT_RETRIES:
                raise RuntimeError(
                    f"Cannot reach the LLM server at {settings.base_url} "
                    f"after {attempt} attempts ({exc}). "
                    "Check that the server is running."
                ) from exc
            wait = _CONNECT_BACKOFF_SECONDS * attempt
            print(f"LLM connection error ({exc}), retry {attempt}/"
                  f"{_CONNECT_RETRIES - 1} in {wait:.0f}s")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def _extract_answer(content: str | None) -> str:
    """Extract the final answer from a completion content.

    Normally the server's reasoning parser puts the chain-of-thought in a
    separate `reasoning` field, but some configurations leave it inline in
    the content, before a `</think>` marker. Keep only what follows the
    last marker.
    """
    content = (content or "").strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1].strip()
    return content


async def merge_segment(
    config: LlmConfig,
    transcriptions: list[str],
    slide_context: str = "",
) -> str:
    """Call the LLM to merge multiple transcriptions of one segment.

    Returns the merged text.

    Reasoning models can exhaust the token budget on their reasoning before
    emitting content (finish_reason=length, content empty). In that case the
    request is retried with a doubled budget, up to _MERGE_MAX_TOKENS; if the
    LLM still returns nothing, the first transcription is used as a fallback.
    """
    prompt = build_merge_prompt(transcriptions, slide_context)
    settings = config.llm

    max_tokens = 2048
    finish_reason = ""
    async with httpx.AsyncClient(timeout=180.0) as client:
        while True:
            response = await _post_chat_completion(client, settings, prompt, max_tokens)
            if response.status_code != 200:
                raise RuntimeError(
                    f"LLM request failed (HTTP {response.status_code}): {response.text}"
                )
            data = response.json()
            content = _extract_answer(data["choices"][0]["message"].get("content"))
            if content:
                return content
            finish_reason = data["choices"][0].get("finish_reason") or ""
            if finish_reason == "length" and max_tokens < _MERGE_MAX_TOKENS:
                print(f"LLM merge hit token budget ({max_tokens}), retrying with more")
                max_tokens *= 2
                continue
            break

    print(
        f"LLM merge returned no content (finish_reason={finish_reason}), "
        "falling back to first transcription"
    )
    return transcriptions[0] if transcriptions else ""


async def merge_annotations(
    config: LlmConfig,
    segments: list[Segment],
    transcriptions_per_segment: list[list[str]],
    slide_contexts: list[str] | None = None,
    label: str = "",
    system_names: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Merge transcriptions one segment at a time.

    Each segment is merged in its own small LLM request (never all segments
    at once), using only the transcriptions of that segment. Progress is
    printed per segment (k/N).

    A segment whose LLM request still fails after connection retries falls
    back to its first transcription; _MAX_CONSECUTIVE_FAILURES failed
    segments in a row abort the merge (before the fusion SRT is written) so
    the run can simply be restarted once the LLM server is back.

    Args:
        config: LLM configuration
        segments: list of segments (for count)
        transcriptions_per_segment: for each segment, the list of transcription texts
        slide_contexts: optional list of OCR text per segment (same length as segments)
        label: optional file label used as prefix in progress messages
        system_names: optional names of the transcription systems, used to
            label the per-system transcriptions in the logs

    Returns:
        list of (merged_text, confidence_score) tuples, one per segment.
    """
    prefix = f"[{label}] " if label else ""
    n = len(transcriptions_per_segment)
    results: list[tuple[str, int]] = []
    fallbacks = 0
    consecutive_failures = 0
    for i, transcriptions in enumerate(transcriptions_per_segment):
        confidence = compute_confidence(transcriptions)
        ctx = slide_contexts[i] if slide_contexts and i < len(slide_contexts) else ""
        if len(transcriptions) >= 2:
            try:
                merged = await merge_segment(config, transcriptions, ctx)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - keep merging other segments
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"LLM merge aborted after {consecutive_failures} consecutive "
                        f"failed segments (last error: {exc})"
                    ) from exc
                fallbacks += 1
                print(f"{prefix}Segment {i + 1}/{n}: LLM merge failed ({exc}), "
                      "falling back to first transcription")
                merged = transcriptions[0]
        else:
            merged = transcriptions[0] if transcriptions else ""
        results.append((merged, confidence))
        print(f"{prefix}LLM merge: {i + 1}/{n} segments done")
        for j, text in enumerate(transcriptions):
            name = system_names[j] if system_names and j < len(system_names) else f"system {j + 1}"
            print(f"{prefix}  {name}: {text}")
        print(f"{prefix}  merged: {merged}")
    if fallbacks:
        print(f"{prefix}LLM merge finished with {fallbacks}/{n} fallback segment(s) "
              "(first transcription used)")
    return results


async def extract_context_bias_terms(
    config: LlmConfig,
    ocr_text: str,
    max_terms: int = 100,
) -> list[str]:
    """Use the LLM to extract terms from OCR text that will most help
    transcription accuracy — acronyms, proper nouns, technical terms,
    uncommon words. Returns a list of single tokens (no spaces).
    """
    if not ocr_text.strip():
        return []

    prompt = (
        "Extract the most useful terms from this slide OCR text for guiding "
        "a speech transcription model. Focus on: acronyms, proper nouns, "
        "technical terms, uncommon words, domain-specific vocabulary. "
        "Exclude common words (articles, prepositions, conjunctions). "
        f"Return at most {max_terms} terms, one per line, as plain tokens "
        "(no spaces within a token). Output ONLY the terms, nothing else.\n\n"
        f"OCR text:\n{ocr_text}"
    )
    settings = config.llm
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await _post_chat_completion(client, settings, prompt, 2048)
        if response.status_code != 200:
            raise RuntimeError(
                f"LLM request failed (HTTP {response.status_code}): {response.text}"
            )
        data = response.json()
        raw = _extract_answer(data["choices"][0]["message"].get("content"))

    terms: list[str] = []
    for line in raw.splitlines():
        term = line.strip().strip(",;:.!?\"'()")
        if term and " " not in term:
            terms.append(term)
    return terms[:max_terms]
