# Data Statement

## Primary Dataset

The behavioural modelling component uses the **AnnoMI Full Dataset**.

The dataset contains annotated Motivational Interviewing transcripts with fields including:

- `transcript_id`
- `video_url`
- `video_title`
- `topic`
- `utterance_id`
- `interlocutor`
- `timestamp`
- `utterance_text`
- `mi_quality`
- `main_therapist_behaviour`
- `client_talk_type`
- therapist behaviour existence indicators
- therapist behaviour subtypes
- annotator metadata

## Why the Full Dataset Is Used

The Full version is used because subtype and existence-indicator fields support richer hierarchical and session-level feature engineering.

## Audio Source

Where available and permitted, session audio is derived from the dataset's `video_url` field.

The pipeline builds one audio record per unique `transcript_id` and records failures such as:

- Deleted video
- Private video
- Missing audio stream
- Download failure
- Validation failure
- ASR failure
- Diarization failure

## Gold and ASR Conditions

The project maintains two parallel representations:

1. Human-annotated transcript utterances
2. ASR-generated speech utterances

Human transcripts are used as the gold-standard comparison condition. ASR text remains the input in the end-to-end speech condition.

## Data Splitting

Splitting is performed at the transcript level to prevent utterances from the same session appearing in both training and test sets.

The methodology uses:

- Transcript-level stratified train/test splitting
- Group-based cross-validation
- Grouping by `transcript_id`
- Out-of-fold predictions for training-side session features

## Privacy

Raw psychotherapy audio and transcripts may contain sensitive information.

The public repository should not include:

- Raw audio
- Personally identifiable information
- Private participant data
- Access tokens
- Restricted source material
- Unredacted clinical content

## Data Availability

Data access remains subject to the original dataset terms, source-platform conditions, and applicable ethical requirements.

This repository provides code and documentation rather than redistributing restricted or sensitive source data.
