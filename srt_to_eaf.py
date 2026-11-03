"""CLI : convertir des transcriptions .srt en fichiers .eaf (ELAN).

Usage :
    python srt_to_eaf.py SRT_DIR MEDIA_DIR OUTPUT_DIR

Chaque fichier SRT_DIR/<chemin>.srt est converti en OUTPUT_DIR/<chemin>.eaf.
Le média correspondant est recherché à MEDIA_DIR/<chemin>.<ext> (même chemin
relatif à part l'extension) et lié au fichier .eaf produit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from speech_annotator_common.eaf_writer import MEDIA_MIMETYPES, write_eaf
from speech_annotator_common.srt_reader import read_srt


def find_media(media_dir: Path, relative_srt: Path) -> Path | None:
    """Find the media file matching an SRT relative path, with any known extension."""
    stem = relative_srt.with_suffix("")
    for ext in sorted(MEDIA_MIMETYPES):
        candidate = media_dir / (str(stem) + ext)
        if candidate.is_file():
            return candidate
    return None


def convert_tree(srt_dir: Path, media_dir: Path, output_dir: Path) -> int:
    """Convert every .srt under srt_dir into .eaf under output_dir. Return exit code."""
    srt_files = sorted(srt_dir.rglob("*.srt"))
    if not srt_files:
        print(f"Erreur : aucun fichier .srt trouvé dans {srt_dir}")
        return 1

    converted = 0
    missing_media: list[Path] = []
    for srt_path in srt_files:
        relative = srt_path.relative_to(srt_dir)
        annotation = read_srt(srt_path)
        if not annotation.segments:
            print(f"AVERTISSEMENT : aucun segment dans {srt_path}, fichier ignoré")
            continue
        media_path = find_media(media_dir, relative)
        if media_path is None:
            missing_media.append(relative)
        output_path = output_dir / relative.with_suffix(".eaf")
        write_eaf(annotation, output_path, media_path)
        converted += 1
        media_info = media_path.name if media_path else "média introuvable"
        print(f"{srt_path} -> {output_path} [{media_info}]")

    print(f"\n{converted} fichier(s) .eaf généré(s) dans {output_dir}")
    if missing_media:
        print("AVERTISSEMENT : aucun média trouvé pour :")
        for relative in missing_media:
            print(f"  - {relative}")
        print("Les .eaf correspondants sont générés sans média lié.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="srt_to_eaf",
        description=(
            "Convertit des fichiers .srt en fichiers .eaf (ELAN), liés aux "
            "audios/vidéos correspondants par nom de fichier."
        ),
    )
    parser.add_argument("srt_dir", type=Path, help="répertoire contenant les fichiers .srt")
    parser.add_argument(
        "media_dir", type=Path, help="répertoire contenant les audios/vidéos correspondants"
    )
    parser.add_argument(
        "output_dir", type=Path, help="répertoire où générer les fichiers .eaf"
    )
    args = parser.parse_args()

    for label, path in (("srt_dir", args.srt_dir), ("media_dir", args.media_dir)):
        if not path.is_dir():
            parser.error(f"le répertoire {label} n'existe pas : {path}")

    return convert_tree(args.srt_dir, args.media_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
