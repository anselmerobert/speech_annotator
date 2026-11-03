"""Write ELAN (.eaf) annotation files from Annotation objects."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

from speech_annotator_common.models import Annotation

# MIME types for media extensions recognised when linking an .eaf to its media.
MEDIA_MIMETYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".wav": "audio/x-wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
}

TRANSCRIPTION_TIER = "Transcription"
CONFIDENCE_TIER = "Confiance"


def _ms(seconds: float) -> int:
    """Convert seconds to integer milliseconds (ELAN time unit)."""
    return round(seconds * 1000)


def build_eaf_document(
    annotation: Annotation, media_path: Path | None, eaf_path: Path
) -> ET.Element:
    """Build the EAF XML tree for an Annotation, optionally linked to a media file."""
    doc = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": "",
            "DATE": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "FORMAT": "3.0",
            "VERSION": "3.0",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan-EAFv3.xsd",
        },
    )

    header = ET.SubElement(doc, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    if media_path is not None:
        mime = MEDIA_MIMETYPES.get(media_path.suffix.lower(), "application/octet-stream")
        relative = Path(os.path.relpath(media_path, eaf_path.parent)).as_posix()
        ET.SubElement(
            header,
            "MEDIA_DESCRIPTOR",
            {
                "MEDIA_URL": media_path.resolve().as_uri(),
                "MIMETYPE": mime,
                "RELATIVE_MEDIA_URL": relative,
            },
        )
    annotation_count = len(annotation.segments) + sum(
        1 for seg in annotation.segments if seg.confidence is not None
    )
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(
        annotation_count
    )

    time_order = ET.SubElement(doc, "TIME_ORDER")
    for idx, seg in enumerate(annotation.segments, start=1):
        ET.SubElement(
            time_order,
            "TIME_SLOT",
            {"TIME_SLOT_ID": f"ts{2 * idx - 1}", "TIME_VALUE": str(_ms(seg.start))},
        )
        ET.SubElement(
            time_order,
            "TIME_SLOT",
            {"TIME_SLOT_ID": f"ts{2 * idx}", "TIME_VALUE": str(_ms(seg.end))},
        )

    ET.SubElement(
        doc,
        "TIER",
        {
            "LINGUISTIC_TYPE_REF": "Transcription",
            "TIER_ID": TRANSCRIPTION_TIER,
            "DEFAULT_LOCALE": "fr",
            "PARTICIPANT": "",
            "ANNOTATOR": "",
        },
    )
    has_confidence = any(seg.confidence is not None for seg in annotation.segments)
    if has_confidence:
        ET.SubElement(
            doc,
            "TIER",
            {
                "LINGUISTIC_TYPE_REF": "Symbole_Dependant",
                "PARENT_REF": TRANSCRIPTION_TIER,
                "TIER_ID": CONFIDENCE_TIER,
                "DEFAULT_LOCALE": "fr",
                "PARTICIPANT": "",
                "ANNOTATOR": "",
            },
        )

    transcription_tier, *extra_tiers = doc.findall("TIER")
    for idx, seg in enumerate(annotation.segments, start=1):
        ann = ET.SubElement(transcription_tier, "ANNOTATION")
        alignable = ET.SubElement(
            ann,
            "ALIGNABLE_ANNOTATION",
            {
                "ANNOTATION_ID": f"a{idx}",
                "TIME_SLOT_REF1": f"ts{2 * idx - 1}",
                "TIME_SLOT_REF2": f"ts{2 * idx}",
            },
        )
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = seg.text

        if seg.confidence is not None and has_confidence:
            conf_ann = ET.SubElement(extra_tiers[0], "ANNOTATION")
            ref = ET.SubElement(
                conf_ann,
                "REF_ANNOTATION",
                {"ANNOTATION_ID": f"conf{idx}", "ANNOTATION_REF": f"a{idx}"},
            )
            ET.SubElement(ref, "ANNOTATION_VALUE").text = str(seg.confidence)

    ET.SubElement(
        doc,
        "LINGUISTIC_TYPE",
        {
            "LINGUISTIC_TYPE_ID": "Transcription",
            "TIME_ALIGNABLE": "true",
            "GRAPHIC_REFERENCES": "false",
        },
    )
    if has_confidence:
        ET.SubElement(
            doc,
            "LINGUISTIC_TYPE",
            {
                "LINGUISTIC_TYPE_ID": "Symbole_Dependant",
                "TIME_ALIGNABLE": "false",
                "GRAPHIC_REFERENCES": "false",
                "CONSTRAINTS": "Symbolic_Association",
            },
        )
        ET.SubElement(
            doc,
            "CONSTRAINT",
            {
                "DESCRIPTION": "1-1 association with a parent annotation",
                "STEREOTYPE": "Symbolic_Association",
            },
        )
    ET.SubElement(doc, "LOCALE", {"LANGUAGE_CODE": "fr", "COUNTRY_CODE": "FR", "VARIANT": ""})
    return doc


def write_eaf(
    annotation: Annotation, output_path: Path, media_path: Path | None = None
) -> None:
    """Write an Annotation to an ELAN .eaf file, optionally linked to a media file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_eaf_document(annotation, media_path, output_path)
    raw = ET.tostring(doc, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="    ", encoding="UTF-8")
    content = "\n".join(line for line in pretty.decode("utf-8").split("\n") if line.strip())
    output_path.write_text(content + "\n", encoding="utf-8")
