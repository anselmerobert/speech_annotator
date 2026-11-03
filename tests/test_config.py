"""Tests for speech_annotator_common.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from speech_annotator_common.config import (
    config_to_yaml,
    load_cli_config,
    load_llm_config,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "configs"


def test_load_cli_config() -> None:
    # Given the minimal config fixture
    # When loading it
    config = load_cli_config(_FIXTURES / "minimal.yaml")
    # Then the fields are populated correctly
    assert len(config.vad_systems) == 1
    assert config.vad_systems[0].name == "pyannote"
    assert config.vad_systems[0].endpoint == "http://localhost:8011"
    assert len(config.transcription_systems) == 2
    assert config.transcription_systems[0].name == "whisper"
    assert config.transcription_systems[1].name == "parakeet"
    assert config.output_format == "srt"


def test_load_cli_config_missing_field(tmp_path: Path) -> None:
    # Given a YAML missing a required field
    bad = tmp_path / "bad.yaml"
    bad.write_text("vad_systems: []\n", encoding="utf-8")
    # When loading it, a ValidationError is raised (transcription_systems missing)
    with pytest.raises(ValidationError):
        load_cli_config(bad)


def test_config_to_yaml() -> None:
    # Given a loaded config
    config = load_cli_config(_FIXTURES / "minimal.yaml")
    # When serializing to YAML and reloading
    yaml_str = config_to_yaml(config)
    reloaded = load_cli_config_from_string(yaml_str)
    # Then the round-trip preserves the data
    assert reloaded.vad_systems == config.vad_systems
    assert reloaded.transcription_systems == config.transcription_systems
    assert reloaded.output_format == config.output_format


def load_cli_config_from_string(yaml_str: str):
    """Helper: load a CliConfig from a YAML string via a temp file."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_str)
        path = Path(f.name)
    try:
        return load_cli_config(path)
    finally:
        path.unlink(missing_ok=True)


def test_env_var_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a YAML with ${TEST_VAR} in api_key and the env var set
    monkeypatch.setenv("TEST_VAR", "secret-from-env")
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "llm:\n  base_url: http://localhost:8080/v1\n  api_key: ${TEST_VAR}\n  model: m\n",
        encoding="utf-8",
    )
    # When loading it
    config = load_llm_config(cfg)
    # Then the env var was substituted
    assert config.llm.api_key == "secret-from-env"
    assert config.llm.base_url == "http://localhost:8080/v1"
    assert config.llm.model == "m"


def test_env_var_substitution_unset_defaults_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a YAML with ${MISSING_VAR} and the env var NOT set
    monkeypatch.delenv("MISSING_VAR", raising=False)
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "llm:\n  base_url: http://x\n  api_key: ${MISSING_VAR}\n  model: m\n",
        encoding="utf-8",
    )
    # When loading it
    config = load_llm_config(cfg)
    # Then the missing var resolves to empty string
    assert config.llm.api_key == ""
