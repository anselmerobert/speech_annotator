"""Tests for speech_annotator_common.eaf_writer."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from speech_annotator_common.models import Annotation, Segment
from speech_annotator_common.eaf_writer import write_eaf


def _sample_annotation() -> Annotation:
    return Annotation(
        audio_file="audio/test.wav",
        audio_hash="abc123",
        vad_system="pyannote",
        transcription_system="whisper",
        segments=[
            Segment(start=0.23, end=1.54, text="Hello world.", confidence=80),
            Segment(start=2.1, end=5.67, text="This is a test."),
        ],
        config_yaml="vad_systems:\n- name: pyannote\n",
    )


def test_write_eaf_creates_valid_xml(tmp_path: Path) -> None:
    # Given an annotation and an output path
    annotation = _sample_annotation()
    out = tmp_path / "out.eaf"
    # When writing the EAF
    write_eaf(annotation, out)
    # Then the file exists and parses as XML with an ANNOTATION_DOCUMENT root
    assert out.exists()
    root = ET.parse(out).getroot()
    assert root.tag == "ANNOTATION_DOCUMENT"


def test_eaf_has_tiers_and_annotations(tmp_path: Path) -> None:
    # Given an annotation with confidences on some segments only
    annotation = _sample_annotation()
    out = tmp_path / "out.eaf"
    # When writing the EAF
    write_eaf(annotation, out)
    # Then both tiers exist with the right annotation counts
    root = ET.parse(out).getroot()
    tiers = {t.get("TIER_ID"): t for t in root.findall("TIER")}
    assert set(tiers) == {"Transcription", "Confiance"}
    assert len(tiers["Transcription"].findall("ANNOTATION")) == 2
    assert len(tiers["Confiance"].findall("ANNOTATION")) == 1
    # And the transcription text and confidence value are preserved
    values = [
        a.find("ALIGNABLE_ANNOTATION/ANNOTATION_VALUE").text
        for a in tiers["Transcription"].findall("ANNOTATION")
    ]
    assert values == ["Hello world.", "This is a test."]
    conf_value = tiers["Confiance"].find("ANNOTATION/REF_ANNOTATION/ANNOTATION_VALUE").text
    assert conf_value == "80"


def test_eaf_has_time_slots_in_milliseconds(tmp_path: Path) -> None:
    # Given an annotation with two segments
    annotation = _sample_annotation()
    out = tmp_path / "out.eaf"
    # When writing the EAF
    write_eaf(annotation, out)
    # Then the time order holds 4 slots with millisecond values
    root = ET.parse(out).getroot()
    slots = root.find("TIME_ORDER").findall("TIME_SLOT")
    values = [int(s.get("TIME_VALUE")) for s in slots]
    assert values == [230, 1540, 2100, 5670]


def test_eaf_links_media_with_relative_url(tmp_path: Path) -> None:
    # Given an EAF written in a subdirectory and a media file elsewhere
    annotation = _sample_annotation()
    media = tmp_path / "media" / "test.mp4"
    media.parent.mkdir()
    media.write_bytes(b"")
    out = tmp_path / "output" / "sub" / "out.eaf"
    # When writing the EAF with a media path
    write_eaf(annotation, out, media)
    # Then the media descriptor references the file with a correct relative URL
    root = ET.parse(out).getroot()
    descriptor = root.find("HEADER/MEDIA_DESCRIPTOR")
    assert descriptor is not None
    assert descriptor.get("MIMETYPE") == "video/mp4"
    assert descriptor.get("RELATIVE_MEDIA_URL") == "../../media/test.mp4"
    assert descriptor.get("MEDIA_URL").endswith("/media/test.mp4")


def test_eaf_without_media_has_no_descriptor(tmp_path: Path) -> None:
    # Given an EAF written without a media path
    annotation = _sample_annotation()
    out = tmp_path / "out.eaf"
    # When writing the EAF without media
    write_eaf(annotation, out, media_path=None)
    # Then no media descriptor is present
    root = ET.parse(out).getroot()
    assert root.find("HEADER/MEDIA_DESCRIPTOR") is None


def test_eaf_without_confidence_has_no_confidence_tier(tmp_path: Path) -> None:
    # Given an annotation whose segments carry no confidence
    annotation = _sample_annotation()
    for seg in annotation.segments:
        seg.confidence = None
    out = tmp_path / "out.eaf"
    # When writing the EAF
    write_eaf(annotation, out)
    # Then only the transcription tier exists
    root = ET.parse(out).getroot()
    tier_ids = [t.get("TIER_ID") for t in root.findall("TIER")]
    assert tier_ids == ["Transcription"]


def test_eaf_escapes_special_characters(tmp_path: Path) -> None:
    # Given an annotation whose text contains XML special characters
    annotation = _sample_annotation()
    annotation.segments[0].text = 'Il a dit "a < b" & c > d'
    out = tmp_path / "out.eaf"
    # When writing the EAF
    write_eaf(annotation, out)
    # Then the XML parses and the text round-trips exactly
    root = ET.parse(out).getroot()
    tier = {t.get("TIER_ID"): t for t in root.findall("TIER")}["Transcription"]
    value = tier.find("ANNOTATION/ALIGNABLE_ANNOTATION/ANNOTATION_VALUE").text
    assert value == 'Il a dit "a < b" & c > d'
