# Speech Annotator

Outil d'annotation de parole à deux étages : un **CLI** génère des transcriptions automatiques via des services de VAD + transcription, puis les **fusionne via un LLM** en calculant un score de confiance par segment ; une **GUI** permet la vérification manuelle assistée par LLM (propagation de corrections).

## Architecture

```
Audio/Video
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  VAD service │───►│ Transcription│───►│   SRT files │
│ (pyannote)   │    │  services    │    │  (auto)     │
└─────────────┘    │ (whisper,    │    └──────┬──────┘
                   │  parakeet)    │           │
                   └─────────────┘           ▼
                                      ┌──────────────┐
                                      │     CLI      │
                                      │ (LLM fusion +│
                                      │  confiance)  │
                                      └──────┬───────┘
                                             │
                                      SRT (fusion)
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │     GUI      │
                                      │ (vérification │
                                      │  + propaga-  │
                                      │  tion LLM)   │
                                      └──────┬───────┘
                                             ▼
                                       SRT (verified)
```

### Étape 1 — CLI : transcription automatique + fusion LLM

Le CLI découpe l'audio en segments (VAD), envoie chaque segment aux services de transcription, puis :

1. Écrit un fichier SRT par combinaison (VAD × système de transcription)
2. **Fusionne** les multiples transcriptions de chaque segment via un LLM
3. **Calcule un score de confiance** (0-100) pour chaque segment, basé sur l'accord entre les transcriptions (distance de Levenshtein moyennée sur toutes les paires)
4. Écrit un fichier SRT de **fusion** (`fusion/`) contenant le texte fusionné et le score de confiance par segment

La fusion et le calcul de confiance s'exécutent uniquement si la section `llm` est présente dans la config CLI.

#### Détection de slides et OCR (optionnel)

Si la section `mistral_ocr` est présente dans la config CLI et que le fichier traité est une **vidéo**, le CLI :

1. Extrait 5 frames réparties dans la vidéo et les envoie à l'API Mistral Vision (modèle Pixtral) pour détecter s'il s'agit d'une présentation avec slides
2. Si des slides sont détectées, extrait une frame au milieu de chaque segment VAD et la passe à l'API OCR de Mistral (`mistral-ocr-latest`) pour en extraire le texte
3. Le texte OCR de chaque segment est utilisé comme contexte pour :
   - **Whisper** : passé en tant que `initial_prompt` (aide à reconnaître les termes techniques affichés sur les slides)
   - **Fusion LLM** : inclus dans le prompt de fusion pour guider le choix entre les transcriptions

Parakeet ne supporte pas de prompt initial, donc seul Whisper bénéficie du contexte OCR au niveau de la transcription. La fusion LLM, en revanche, utilise le contexte OCR pour tous les systèmes.

### Étape 2 — GUI : vérification manuelle

La GUI charge les annotations de fusion (pré-calculées par le CLI), présente les segments par ordre de confiance croissante (les moins fiables en premier), et permet la correction, la validation et la propagation de corrections via LLM. Les statistiques de correction (pourcentage de mots modifiés par segment) sont affichées en temps réel.

## Installation

### Prérequis

- Python ≥ 3.11
- `uv` (gestionnaire de paquets) ou `pip`
- `ffmpeg` (extraction audio, conversion)
- GPU NVIDIA + Docker (pour les services de VAD/transcription)

### Setup du projet

```bash
uv sync          # crée le .venv et installe les dépendances
```

Ou avec pip :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Démarrer les services Docker

Les services de VAD et transcription tournent dans des conteneurs Docker :

```bash
cd docker
export HF_TOKEN=<votre-huggingface-token>    # requis pour pyannote
docker compose up -d
```

Services exposés :

| Service       | Port  | Modèle                            |
|---------------|-------|-----------------------------------|
| pyannote-vad  | 8001  | pyannote/voice-activity-detection  |
| whisper       | 8002  | faster-whisper (large-v3)         |
| parakeet      | 8003  | nvidia/parakeet-tdt-0.6b-v3       |

En plus des services Docker locaux, le CLI supporte la transcription via l'**API web de Mistral** (modèle Voxtral) sans nécessiter de conteneur Docker. Voir la section Configuration ci-dessous.

Vérifier que les services Docker sont opérationnels :

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

#### Utilisateur non-root dans les conteneurs

Les conteneurs s'exécutent avec un utilisateur non-root `app` (UID/GID `1000:1000` par défaut, le premier utilisateur standard sur la plupart des Linux) : le volume de cache des modèles HF (`model-cache`) n'est ainsi pas rempli de fichiers appartenant à root.

Si votre UID/GID est différent (vérifiez avec `id -u` et `id -g`), pensez à remplacer `1000` aux deux endroits suivants :

1. la ligne `user: "1000:1000"` de chaque service dans `docker/docker-compose.yml` ;
2. la ligne `RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 ...` de chaque `Dockerfile`.

Puis reconstruisez les images :

```bash
docker compose up -d --build
```

## Configuration

### Config CLI (`configs/example_config.yaml`)

La config CLI inclut les services de VAD/transcription **et** la config LLM utilisée pour la fusion :

```yaml
vad_systems:
  - name: pyannote
    endpoint: http://localhost:8001

transcription_systems:
  - name: whisper
    endpoint: http://localhost:8002
  - name: parakeet
    endpoint: http://localhost:8003
  - name: voxtral
    type: mistral
    model: voxtral-mini-latest
    api_key: ${MISTRAL_API_KEY}

output_format: srt

llm:
  base_url: https://api.mistral.ai/v1
  api_key: ${MISTRAL_API_KEY}
  model: glm-5-2

mistral_ocr:
  api_key: ${MISTRAL_API_KEY}
  ocr_model: mistral-ocr-latest
  vision_model: pixtral-12b-latest
```

Les systèmes de transcription peuvent être de deux types :
- **`docker`** (par défaut) : service HTTP local avec `endpoint` (whisper, parakeet)
- **`mistral`** : API web de Mistral avec `model` et `api_key` (Voxtral)

La section `mistral_ocr` (optionnelle) active la détection de slides et l'OCR sur les vidéos via les API web de Mistral (Vision pour la détection, OCR pour l'extraction de texte).

Les variables d'environnement `${VAR}` sont substituées automatiquement. Définissez la clé avant de lancer le CLI :

```bash
export MISTRAL_API_KEY=<votre-clé>
```

### Config LLM pour la GUI (`configs/example_llm_config.yaml`)

La GUI utilise un LLM pour la propagation de corrections (appliquer les corrections d'un segment validé aux segments similaires non encore validés) :

```yaml
llm:
  base_url: https://api.mistral.ai/v1
  api_key: ${MISTRAL_API_KEY}
  model: glm-5-2
```

## Utilisation

### CLI — transcription automatique + fusion

```bash
export MISTRAL_API_KEY=<votre-clé>

python -m speech_annotator_cli \
    --config configs/example_config.yaml \
    --input /chemin/vers/audio/ \
    --output /chemin/vers/annotations_auto/
```

Options :

| Option       | Description                                          |
|--------------|------------------------------------------------------|
| `--config`   | Fichier YAML de configuration (VAD + transcription + LLM)  |
| `--input`    | Fichier ou répertoire audio/vidéo                    |
| `--output`   | Répertoire de sortie pour les fichiers SRT           |

Formats audio/vidéo supportés : `.wav`, `.mp3`, `.m4a`, `.flac`, `.webm`, `.ogg`, `.avi`, `.mp4`, `.mkv`, `.mov`

Les fichiers SRT sont organisés comme :
- `{output}/{vad}_{transcription}/{chemin_relatif}/{stem}.srt` — une transcription par système
- `{output}/fusion/{chemin_relatif}/{stem}.srt` — fusion LLM avec scores de confiance

### GUI — vérification manuelle

```bash
export MISTRAL_API_KEY=<votre-clé>

python -m speech_annotator_gui \
    --config configs/example_llm_config.yaml \
    --audio-dir /chemin/vers/audio/ \
    --auto-annotations /chemin/vers/annotations_auto/ \
    --output /chemin/vers/annotations_verifiees/
```

Options :

| Option               | Description                                    |
|----------------------|------------------------------------------------|
| `--config`           | Fichier YAML de config LLM (pour la propagation) |
| `--audio-dir`        | Répertoire des fichiers audio/vidéo           |
| `--auto-annotations` | Répertoire des SRT générés par le CLI         |
| `--output`           | Répertoire de sortie pour les SRT vérifiés     |

#### Workflow de vérification

1. Sélectionner un fichier dans la liste de gauche — la GUI charge l'annotation de fusion (pré-calculée par le CLI)
2. Les segments sont triés par ordre de confiance croissante — les moins fiables en premier
3. Pour chaque segment :
   - Écouter le segment (bouton play)
   - Corriger le texte si nécessaire
   - Valider (bouton ✓ Validate)
4. Les segments validés sont sauvegardés dans le SRT de sortie à chaque validation
5. Bouton "Validate all remaining" : valide d'un coup tous les segments non encore validés
6. Bouton "Propagate corrections" : utilise le LLM pour propager les corrections validées vers les segments non validés similaires
7. La courbe "Words Corrected %" affiche le pourcentage de mots corrigés par segment (ordonnés par confiance) :
   - Barres **vertes** : segments validés (pourcentage de mots corrigés)
   - Barres **bleues** : segments non validés, surmontées d'une **flèche bleue vers le bas** pointant la barre
   - **Rectangle rouge** : segment courant
   - Le titre du diagramme affiche la progression (ex. `42/1076 validated`)
   - **Cliquer sur une barre** (ou sa flèche) navigue directement vers ce segment

#### En-tête SRT vérifié

Les fichiers SRT vérifiés contiennent des métadonnées en commentaire :

```
; audio_hash=<sha256>
; vad_system=<systèmes VAD utilisés>
; transcription_system=<systèmes de transcription utilisés>
; verifier=<login utilisateur>
; gui_name=speech-annotator-gui
; gui_version=0.1.0
; <config CLI d'origine>
; --- LLM config ---
; <config LLM (clé API masquée)>
```

## Scripts utilitaires

### `srt_to_eaf.py` — conversion SRT → ELAN

```bash
python srt_to_eaf.py <srt_dir> <media_dir> <output_dir>
```

Convertit chaque `.srt` de `srt_dir` (récursivement) en `.eaf` dans `output_dir`, en liant le média correspondant cherché dans `media_dir` (même chemin relatif à part l'extension : `.mp4`, `.wav`, ...).

### `remux_mp4.py` — correction de MP4 à timeline audio étirée

```bash
python remux_mp4.py [--dry-run] [--min-gap SECONDES] <répertoire>
```

Détecte les MP4 dont la durée audio déclarée (table `stts` du conteneur) dépasse la durée réelle de leur contenu — la dérive qui décale les découpes `-ss`/`-to` de ffmpeg et les lecteurs type ELAN — et génère une version corrigée `<nom>_remuxed.mp4` (timestamps vidéo recalés avec `-itsscale`, audio ré-encodé à timestamps contigus).

## Tests

```bash
uv run pytest
```

## Structure du projet

```
speech_annotator/
├── speech_annotator_cli/       # CLI : VAD + transcription + fusion LLM
│   ├── __main__.py             # Point d'entrée CLI
│   └── pipeline.py             # Orchestration (VAD, transcription, fusion)
├── speech_annotator_common/    # Code commun (CLI + GUI)
│   ├── audio_utils.py          # Hash, extraction ffmpeg
│   ├── config.py               # Modèles Pydantic (CLI + LLM)
│   ├── http_clients.py         # Clients HTTP VAD/transcription/OCR (Docker + Mistral API)
│   ├── llm_merger.py           # Fusion LLM + scoring de confiance + contexte OCR
│   ├── models.py               # Segment, Annotation
│   ├── eaf_writer.py           # Écriture de fichiers ELAN (.eaf)
│   ├── srt_writer.py           # Écriture SRT + métadonnées
│   ├── srt_reader.py           # Lecture SRT + métadonnées
│   └── format_registry.py      # Registry des formats de sortie
├── speech_annotator_gui/       # GUI : vérification manuelle
│   ├── __main__.py             # Point d'entrée GUI
│   ├── main_window.py          # Fenêtre principale
│   ├── annotation_store.py     # Index des annotations auto/fusion/vérifiées
│   ├── llm_merger.py           # Propagation de corrections via LLM
│   ├── segment_state.py        # État par segment
│   └── widgets/                # Composants UI
│       ├── word_corrections_chart.py
│       ├── file_list_panel.py
│       ├── navigation_bar.py
│       ├── segment_editor.py
│       ├── segment_player.py
│       └── waveform_view.py
├── docker/                     # Services Docker
│   ├── docker-compose.yml
│   ├── pyannote-vad/           # VAD (pyannote.audio)
│   ├── whisper/               # Transcription (faster-whisper)
│   └── parakeet/              # Transcription (NVIDIA NeMo)
├── configs/                    # Configs d'exemple
│   ├── example_config.yaml     # Config CLI (VAD + transcription + LLM)
│   └── example_llm_config.yaml # Config LLM pour la GUI (propagation)
├── srt_to_eaf.py               # CLI utilitaire : conversion SRT → ELAN (.eaf)
├── remux_mp4.py                # CLI utilitaire : correction des MP4 à timeline audio étirée
├── tests/                      # Tests pytest
├── pyproject.toml
└── uv.lock
```
