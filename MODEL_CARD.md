# Model Card

## Project

**End-to-End Speech AI and LLM-Based Psychotherapy Fidelity Assessment**

This repository contains a hierarchical research pipeline for analysing psychotherapy sessions, with a focus on Motivational Interviewing fidelity.

## Intended Use

The models are intended for research on:

- Speech-to-text processing of psychotherapy-session audio
- Speaker diarization and therapist/client role assignment
- Therapist behaviour classification
- Client talk-type classification
- Session-level fidelity assessment
- Error-propagation analysis across speech and language components

The system is intended as a research and decision-support framework only.

## Out-of-Scope Use

The models must not be used to:

- Replace qualified therapists
- Diagnose mental-health conditions
- Make autonomous treatment decisions
- Evaluate individuals without informed consent
- Produce high-stakes clinical decisions without human oversight
- Infer protected or sensitive personal characteristics

## Speech AI Components

### Voice Activity Detection

Primary model:

- Silero VAD

Reference alternative:

- WebRTC VAD

### Automatic Speech Recognition

Primary model:

- Faster-Whisper `large-v3`

The ASR component produces timestamped transcript segments and may include word-level timestamps and confidence-related metadata.

### Speaker Diarization

Primary model:

- `pyannote/speaker-diarization-community-1`

The diarization model produces anonymous speaker labels such as `SPEAKER_00` and `SPEAKER_01`. A separate role-mapping stage assigns therapist and client roles.

## Language Models

### Therapist Behaviour Classification

Target classes:

- Reflection
- Question
- Input
- Other

Model families:

- TF-IDF + LinearSVC
- SBERT `all-MiniLM-L6-v2` + Logistic Regression
- DistilBERT with conversational context
- RoBERTa
- FLAN-T5 benchmark

### Client Talk-Type Classification

Target classes:

- Change Talk
- Sustain Talk
- Neutral Talk

The same model families are evaluated separately for client utterances.

## Session-Level Model

The session-level model uses standardised, interpretable features with Logistic Regression and balanced class weights.

Feature groups include:

- Static behavioural ratios
- Behaviour-signal features
- Subtype-completeness features
- Interaction features
- Therapist-to-client transitions
- Early-to-late trajectories
- Speech-derived structural features

## Evaluation

Utterance-level metrics include:

- Macro F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrices

Session-level metrics include:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1
- Probability of high-quality MI

Speech metrics may include:

- Word Error Rate
- Character Error Rate
- Diarization Error Rate
- Speaker-attribution accuracy
- VAD precision, recall, and F1

## Limitations

Known limitations include:

- Domain mismatch between general speech models and psychotherapy dialogue
- Low-volume or emotional speech
- Accent and demographic bias
- Overlapping speech
- Speaker diarization errors
- Missing or unavailable source videos
- Propagation of ASR errors into downstream behavioural models
- Limited external clinical validation

## Human Oversight

All outputs require expert interpretation. Model predictions should be treated as supporting evidence, not definitive clinical judgments.
