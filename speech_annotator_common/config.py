"""Pydantic v2 models for YAML configs (CLI and GUI)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _substitute_env(value: str) -> str:
    """Replace ${VAR} with os.environ.get(VAR, "")."""
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            missing.append(var_name)
            return ""
        return val

    result = _ENV_VAR_PATTERN.sub(_replace, value)
    if missing:
        import warnings
        warnings.warn(
            f"Environment variable(s) not set: {', '.join(missing)}. "
            f"Substituted with empty string.",
            stacklevel=2,
        )
    return result


def _substitute_tree(node: object) -> object:
    if isinstance(node, dict):
        return {k: _substitute_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_tree(v) for v in node]
    if isinstance(node, str):
        return _substitute_env(node)
    return node


class VadSystem(BaseModel):
    name: str
    endpoint: str


class TranscriptionSystem(BaseModel):
    name: str
    endpoint: str = ""
    type: str = "docker"  # "docker" for local HTTP service, "mistral" for Mistral web API
    api_key: str = ""
    model: str = ""


class MistralOcrSettings(BaseModel):
    api_key: str = ""
    ocr_model: str = "mistral-ocr-latest"
    vision_model: str = "pixtral-12b-latest"


class CliConfig(BaseModel):
    vad_systems: list[VadSystem]
    transcription_systems: list[TranscriptionSystem]
    output_format: str = "srt"
    llm: LlmSettings | None = None
    mistral_ocr: MistralOcrSettings | None = None


class LlmSettings(BaseModel):
    base_url: str
    api_key: str = ""
    model: str


class LlmConfig(BaseModel):
    llm: LlmSettings


def load_cli_config(path: Path) -> CliConfig:
    """Read YAML from path, substitute env vars, validate into CliConfig."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CliConfig.model_validate(_substitute_tree(raw))


def config_to_yaml(config: CliConfig) -> str:
    """Serialize a CliConfig back to a YAML string (used for SRT metadata)."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def load_llm_config(path: Path) -> LlmConfig:
    """Read YAML from path, substitute env vars, validate into LlmConfig."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LlmConfig.model_validate(_substitute_tree(raw))
