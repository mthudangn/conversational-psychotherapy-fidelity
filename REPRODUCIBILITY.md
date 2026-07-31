# Reproducibility Guide

## Environment

Recommended Python version:

```text
Python 3.11 or 3.12
```

Install Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install FFmpeg on macOS:

```bash
brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

## Hugging Face Access

The pyannote diarization model requires:

1. A Hugging Face account
2. Accepted access conditions for `pyannote/speaker-diarization-community-1`
3. A read-only Hugging Face access token

Never commit the token.

Use an environment variable:

```bash
export HF_TOKEN="hf_..."
```

or an interactive hidden prompt inside the notebook.

## Main Notebook

Primary notebook:

```text
psychotherapy_pipeline_code.ipynb
```

## Recommended Execution Order

1. Install dependencies
2. Import libraries
3. Set random seeds
4. Load AnnoMI Full
5. Build the consensus dataset
6. Build the unique audio manifest
7. Download and standardise audio
8. Run VAD
9. Run Faster-Whisper ASR
10. Run pyannote speaker diarization
11. Align ASR text with speaker timestamps
12. Reconstruct utterances
13. Match ASR utterances to gold utterances
14. Train therapist and client text models
15. Generate out-of-fold predictions
16. Build session-level features
17. Train the session-level classifier
18. Evaluate the gold-transcript and speech conditions
19. Export figures and result tables

## Leakage Prevention

The pipeline uses:

- Transcript-level splitting
- Group-based cross-validation
- Out-of-fold training predictions
- Held-out session evaluation

## Randomness

Use fixed random seeds for:

- NumPy
- Python `random`
- PyTorch
- scikit-learn estimators
- Transformer training

## Cached Artefacts

The following may be cached locally but should not be committed:

- Audio files
- Hugging Face model cache
- Transformer checkpoints
- Fold-specific model directories
- Temporary ASR outputs
- Local environments

## Expected Public Artefacts

Recommended public outputs include:

- Clean notebook
- `requirements.txt`
- `README.md`
- Model card
- Data statement
- Ethical considerations
- Reproducibility guide
- Selected de-identified figures
- Aggregate evaluation tables

## Failure Logging

Speech-processing failures should be retained in a manifest with fields such as:

```text
transcript_id
video_url
pipeline_status
error_message
```

Do not silently discard unavailable or failed sessions.
