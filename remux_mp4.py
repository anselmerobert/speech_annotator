"""CLI : détecter et corriger les MP4 dont la timeline audio est étirée.

Usage :
    python remux_mp4.py [--dry-run] [--min-gap SECONDS] RÉPERTOIRE

Défaut détecté : la durée déclarée de la piste audio (table stts du MP4)
dépasse la durée réelle de son contenu (nb_frames x échantillons par frame).
Les outils qui suivent les timestamps du conteneur (lecteur d'ELAN, découpe
ffmpeg -ss/-to) dérivent alors progressivement par rapport à l'audio
réellement décodé (jusqu'à ~1,6 s sur une vidéo de 80 min dans nos données
d'essai : texte ASR décalé, annotations ELAN désynchronisées).

Pour chaque fichier concerné, une version corrigée <stem>_remuxed<suffix>
est générée à côté de l'original : vidéo copiée avec timestamps recalés
(-itsscale réel/déclaré) et audio ré-encodé avec timestamps contigus
(asetpts=N/SR/TB). Les fichiers sains ne sont pas modifiés.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Échantillons par paquet pour les codecs où nb_frames (ffprobe) est fiable.
_FRAME_SAMPLES = {"aac": 1024, "mp3": 1152}

_DEFAULT_MIN_GAP = 0.25


def _probe(path: Path) -> tuple[list[dict], str | None]:
    """Retourne (flux ffprobe, durée déclarée du conteneur)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,duration,nb_frames,sample_rate"
         ":format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    format_info = data.get("format") or {}
    return data.get("streams", []), format_info.get("duration")


def _decode_audio_duration(path: Path) -> float:
    """Durée réelle de la première piste audio, par décodage complet."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    total_bytes = 0
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1 << 16)
        if not chunk:
            break
        total_bytes += len(chunk)
    proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0 or total_bytes == 0:
        raise RuntimeError("décodage audio impossible")
    return total_bytes / 2 / 16000


def analyze(
    path: Path, min_gap: float = _DEFAULT_MIN_GAP
) -> tuple[str, dict[str, object]]:
    """Analyse un MP4 et retourne (statut, infos).

    Statuts : 'ok', 'stretched' (timeline audio étirée, corrigeable),
    'shortened' (timeline plus courte que l'audio), 'no_audio', 'error'.
    """
    try:
        streams, format_duration = _probe(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return "error", {"detail": f"illisible par ffprobe : {exc}"}

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        return "no_audio", {"detail": "aucune piste audio"}

    declared_raw = audio.get("duration") or format_duration
    if not declared_raw:
        return "error", {"detail": "durée déclarée indisponible"}
    declared = float(declared_raw)

    codec = audio.get("codec_name")
    nb_frames = int(audio["nb_frames"]) if audio.get("nb_frames") else 0
    sample_rate = int(audio["sample_rate"]) if audio.get("sample_rate") else 0
    if codec in _FRAME_SAMPLES and nb_frames > 0 and sample_rate > 0:
        real = nb_frames * _FRAME_SAMPLES[codec] / sample_rate
    else:
        try:
            real = _decode_audio_duration(path)
        except RuntimeError as exc:
            return "error", {"detail": str(exc)}

    gap = declared - real
    info: dict[str, object] = {
        "declared": declared,
        "real": real,
        "gap": gap,
        "scale": real / declared,
        "has_video": any(s.get("codec_type") == "video" for s in streams),
    }
    if gap > min_gap:
        return "stretched", info
    if gap < -min_gap:
        return "shortened", info
    return "ok", info


def remux(src: Path, dst: Path, scale: float, has_video: bool) -> None:
    """Génère une copie de src avec des timestamps recalés sur l'audio réel."""
    cmd = ["ffmpeg", "-v", "error", "-itsscale", f"{scale:.8f}", "-i", str(src)]
    if has_video:
        cmd += ["-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy"]
    else:
        cmd += ["-map", "0:a:0"]
    cmd += ["-af", "asetpts=N/SR/TB", "-c:a", "aac", "-b:a", "192k", "-y", str(dst)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="remux_mp4",
        description=(
            "Détecte les MP4 dont la timeline audio déclarée est étirée par "
            "rapport à l'audio réellement décodé (dérive des découpes -ss/-to "
            "et des lecteurs type ELAN) et génère une version corrigée "
            "<nom>_remuxed.mp4."
        ),
    )
    parser.add_argument("directory", type=Path, help="répertoire à parcourir (récursif)")
    parser.add_argument(
        "--min-gap", type=float, default=_DEFAULT_MIN_GAP,
        help=f"écart minimal en secondes considéré comme un défaut "
             f"(défaut : {_DEFAULT_MIN_GAP})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="détecter et afficher sans générer les corrections",
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"le répertoire n'existe pas : {args.directory}")

    files = sorted(
        p for p in args.directory.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp4" and not p.stem.endswith("_remuxed")
    )
    if not files:
        print(f"Aucun fichier .mp4 trouvé dans {args.directory}")
        return 0

    counts = {"ok": 0, "fixed": 0, "todo": 0, "skipped": 0, "warn": 0, "error": 0}
    for path in files:
        status, info = analyze(path, args.min_gap)
        gap = info.get("gap")

        if status == "ok":
            counts["ok"] += 1
            print(f"{path.name}: OK (écart {gap:+.3f}s)")

        elif status == "stretched":
            dst = path.with_name(f"{path.stem}_remuxed{path.suffix}")
            if args.dry_run:
                counts["todo"] += 1
                print(f"{path.name}: TIMELINE ÉTIRÉE (écart {gap:+.3f}s, "
                      f"déclaré {info['declared']:.3f}s / réel {info['real']:.3f}s) "
                      f"— correction prévue : {dst.name}")
                continue
            if dst.exists():
                counts["skipped"] += 1
                print(f"{path.name}: écart {gap:+.3f}s — {dst.name} existe déjà, ignoré")
                continue
            try:
                remux(path, dst, float(info["scale"]), bool(info["has_video"]))
                new_status, new_info = analyze(dst, args.min_gap)
                if new_status == "ok":
                    counts["fixed"] += 1
                    print(f"{path.name}: écart {gap:+.3f}s → {dst.name} généré "
                          f"(écart résiduel {new_info.get('gap', 0.0):+.3f}s)")
                else:
                    counts["error"] += 1
                    print(f"{path.name}: écart {gap:+.3f}s → {dst.name} généré mais "
                          f"écart résiduel {new_info.get('gap'):+.3f}s")
            except subprocess.CalledProcessError as exc:
                counts["error"] += 1
                print(f"{path.name}: échec du re-mux : {exc}")

        elif status == "shortened":
            counts["warn"] += 1
            print(f"{path.name}: AVERTISSEMENT timeline plus courte que l'audio "
                  f"(écart {gap:+.3f}s) — non corrigé")

        else:
            key = "error" if status == "error" else "warn"
            counts[key] += 1
            print(f"{path.name}: {info.get('detail', status)}")

    print(f"\n{len(files)} fichier(s) analysé(s) : {counts['ok']} sain(s), "
          f"{counts['fixed']} corrigé(s), {counts['todo']} à corriger (dry-run), "
          f"{counts['skipped']} ignoré(s), {counts['warn']} avertissement(s), "
          f"{counts['error']} erreur(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
