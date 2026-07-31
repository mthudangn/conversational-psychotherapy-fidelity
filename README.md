# End-to-End Speech AI and LLM-Based Psychotherapy Fidelity Assessment

A research and engineering repository for automated psychotherapy intervention fidelity assessment using **Speech AI**, **Natural Language Processing**, **Transformer-based language models**, and **interpretable machine learning**, with a primary focus on **Motivational Interviewing (MI)**.

The system connects raw psychotherapy-session audio with behavioural and session-level fidelity assessment through a modular hierarchical pipeline:

```text
YouTube Session Video
        ↓
Audio Extraction and Validation
        ↓
Audio Standardisation
        ↓
Voice Activity Detection
        ↓
Automatic Speech Recognition
        ↓
Speaker Diarization
        ↓
Therapist–Client Role Assignment
        ↓
ASR–Speaker Timestamp Alignment
        ↓
Utterance Reconstruction
        ↓
Therapist Behaviour Classification
        ↓
Client Talk-Type Classification
        ↓
Behavioural, Transition, and Trajectory Features
        ↓
Session-Level MI Fidelity Assessment
        ↓
Interpretable Fidelity Evidence
```

This repository supports a long-term research programme investigating whether modern Speech AI and language models can capture therapist–client behavioural dynamics associated with psychotherapy quality while preserving methodological transparency, leakage control, interpretability, and reproducibility.

---

## Project Scope

Psychotherapy intervention fidelity refers to how closely a therapy session follows the principles, behavioural techniques, and delivery standards of a specific therapeutic approach.

Traditional fidelity assessment often requires trained human coders to review full therapy recordings or transcripts. Although clinically valuable, manual assessment is:

- Time-consuming
- Expensive
- Difficult to scale
- Dependent on specialist training
- Sensitive to subjective interpretation
- Hard to deploy continuously across large services

This project investigates an automated and interpretable alternative combining:

- Audio acquisition and standardisation
- Voice Activity Detection
- Automatic Speech Recognition
- Speaker diarization
- Therapist/client role mapping
- Utterance-level behavioural classification
- Conversational context modelling
- Session-level behavioural aggregation
- Transition and trajectory analysis
- Interpretable fidelity prediction
- End-to-end error-propagation analysis

Rather than directly mapping raw audio or transcript text to a final quality label, the framework decomposes the problem into clinically meaningful and technically measurable stages.

---

# Research Questions

The project examines whether modern Speech AI and Transformer-based models can:

1. Extract valid audio from psychotherapy-session videos.
2. Detect speech and silence reliably in long-form conversations.
3. Transcribe conversational psychotherapy audio accurately.
4. Separate therapist and client speech through speaker diarization.
5. Reconstruct speaker-labelled utterances from ASR and diarization outputs.
6. Match ASR-generated utterances with human-annotated utterances.
7. Classify therapist behaviour at the utterance level.
8. Classify client motivational language at the utterance level.
9. Capture local conversational context and turn-taking dynamics.
10. Model therapist-to-client behavioural transitions.
11. Measure early-to-late behavioural movement within sessions.
12. Predict session-level Motivational Interviewing quality.
13. Quantify how upstream speech-processing errors affect downstream behavioural and fidelity predictions.
14. Produce interpretable evidence rather than only opaque model outputs.

---

# End-to-End Architecture

## Stage 1 — Dataset and Audio Manifest

The dataset contains one or more rows per psychotherapy utterance and includes a `video_url` field.

The speech pipeline first builds a unique audio manifest using:

```text
transcript_id + video_url
```

Only one video is processed for each unique psychotherapy session.

The audio manifest records:

```text
transcript_id
video_url
audio_path
download_status
pipeline_status
error_message
```

This prevents duplicated downloads and creates an auditable record of successful and failed sessions.

## Stage 2 — YouTube Audio Acquisition

Audio is extracted from the session-level `video_url` using `yt-dlp`.

The download module:

- Downloads audio-only content
- Prevents playlist expansion
- Reuses cached files where available
- Records unavailable, private, or deleted videos
- Associates every audio file with its original `transcript_id`
- Avoids silently dropping failed sessions

## Stage 3 — Audio Validation and Standardisation

YouTube audio may use different codecs, sample rates, and channel layouts.

The pipeline uses `FFmpeg` and `FFprobe` to convert each recording into:

```text
WAV
Mono
16 kHz
PCM 16-bit
```

The standardisation stage includes:

- Audio stream validation
- Duration validation
- Stereo-to-mono conversion
- Resampling
- Codec conversion
- Timestamp preservation
- File-integrity checks

---

# Speech AI Pipeline

## 1. Voice Activity Detection

The primary Voice Activity Detection model is **Silero VAD**.

It is used to:

- Detect speech onset and offset
- Remove long silence regions
- Preserve speech timestamps
- Merge fragmented speech regions
- Apply minimum speech-duration thresholds
- Apply minimum silence-duration thresholds
- Add padding around speech boundaries
- Reduce unnecessary ASR computation

VAD is especially important for psychotherapy recordings containing reflective pauses, emotional silence, hesitations, low-volume speech, background noise, short acknowledgements, interrupted turns, and long gaps between responses.

VAD does **not** identify the speaker. It only determines whether speech is present.

Alternative VAD approaches represented in the architecture include:

- WebRTC VAD
- pyannote segmentation
- Energy-based detection
- Threshold-based frame analysis

## 2. Automatic Speech Recognition

The primary ASR model is **Faster-Whisper large-v3**.

Faster-Whisper is an optimised implementation of the Whisper architecture using CTranslate2. It is responsible for converting speech into transcript text.

The ASR stage supports:

- Long-form conversational transcription
- English-language restriction
- Automatic language detection where enabled
- Segment-level timestamps
- Word-level timestamps
- Punctuation generation
- Confidence-related metadata
- Batched inference
- Voice activity filtering
- Reproducible transcript export

Primary configuration:

```text
Model: large-v3
Task: transcribe
Language: English
Word timestamps: enabled
Audio format: mono 16 kHz WAV
```

Alternative ASR families considered:

- OpenAI Whisper
- Faster-Whisper
- Wav2Vec 2.0
- HuBERT
- NVIDIA NeMo ASR
- Parakeet
- Cloud-based speech-recognition systems

ASR evaluation metrics include:

- Word Error Rate
- Character Error Rate
- Substitution rate
- Deletion rate
- Insertion rate
- Real-Time Factor
- Timestamp-alignment error
- Downstream behaviour-classification degradation

## 3. Speaker Diarization

The primary speaker diarization model is **pyannote speaker-diarization-community-1**.

Speaker diarization answers: **Who spoke when?**

The pyannote pipeline includes:

- Speaker-change detection
- Speaker embedding extraction
- Speaker clustering
- Segment-level speaker attribution
- Speaker-count estimation
- Overlapping-speech handling
- Exclusive speaker diarization
- Timestamp generation

The model initially produces anonymous labels such as:

```text
SPEAKER_00
SPEAKER_01
```

It does not inherently know which speaker is the therapist or client.

Alternative diarization approaches represented in the architecture include:

- pyannote.audio
- NVIDIA NeMo MSDD
- SpeechBrain ECAPA-TDNN
- Speaker-embedding clustering
- Speaker-recognition systems with voice enrolment

Diarization evaluation metrics include:

- Diarization Error Rate
- Speaker confusion
- Missed speech
- False-alarm speech
- Therapist/client attribution accuracy

## 4. Speaker Recognition and Role Assignment

A separate role-mapping stage determines:

```text
SPEAKER_00 → Therapist
SPEAKER_01 → Client
```

For the research dataset, role assignment can use:

- Temporal overlap with annotated therapist/client turns
- Timestamp proximity
- Session-level majority assignment
- Speaker consistency
- Lexical correspondence
- Turn-order constraints

For real-world deployment without gold labels, role assignment could use:

- A short manually verified calibration segment
- Enrolled therapist voice samples
- Speaker-recognition embeddings
- Session metadata
- Manual review

The system deliberately avoids using perceived gender, accent, or pitch to infer therapeutic role.

## 5. ASR–Diarization Alignment

Whisper and pyannote produce independent timestamp sequences.

The alignment layer combines:

- ASR segments
- Word-level timestamps
- VAD intervals
- Diarization intervals

Each word or ASR segment is assigned to the speaker with the strongest temporal overlap.

Primary alignment strategies include:

- Maximum temporal overlap
- Word-level speaker assignment
- Segment midpoint matching
- Boundary smoothing
- Adjacent same-speaker merging
- Minimum-overlap thresholds

## 6. Overlapping Speech

Psychotherapy conversations often include short interruptions, simultaneous responses, and backchannel acknowledgements such as “yeah,” “mm-hmm,” and “right.”

The pipeline records overlap-related information where available and uses boundary-aware alignment to reduce attribution errors.

## 7. Utterance Reconstruction

ASR segments do not always correspond to meaningful conversational utterances.

The reconstruction stage uses:

- Speaker identity
- Sentence boundaries
- Punctuation
- Pause duration
- Semantic completeness
- Maximum token length
- Short acknowledgement rules
- Consecutive same-speaker logic

The reconstructed speech table contains fields such as:

```text
transcript_id
speech_utterance_id
start
end
duration
diarized_speaker
interlocutor
utterance_text
avg_logprob
no_speech_prob
speaker_overlap_ratio
```

## 8. Matching ASR Utterances to Human-Annotated Utterances

The system maintains two parallel representations:

```text
Gold transcript utterances
ASR-generated speech utterances
```

ASR utterances are matched to annotated utterances using:

- Timestamp overlap
- Timestamp proximity
- Normalised lexical similarity
- Speaker compatibility
- Sequence order

The aligned output contains:

```text
speech_utterance_id
matched_gold_utterance_id
```

This supports direct comparison between gold-transcript and end-to-end speech conditions.

---

# NLP and Language Model Pipeline

## Therapist Behaviour Classification

Therapist utterances are classified into:

- Reflection
- Question
- Input
- Other

## Client Talk-Type Classification

Client utterances are classified into:

- Change Talk
- Sustain Talk
- Neutral Talk

---

# Language Model Families

## 1. TF-IDF + LinearSVC

The baseline uses unigram and bigram TF-IDF features with English stop-word filtering and a LinearSVC classifier.

It provides:

- A strong classical NLP baseline
- Fast training and inference
- Sparse lexical representations
- Interpretable feature weights
- A benchmark for Transformer-based models

## 2. SBERT + Logistic Regression

The semantic embedding model uses:

```text
Sentence-BERT
all-MiniLM-L6-v2
```

Each utterance is encoded as a dense sentence embedding and classified using Logistic Regression.

## 3. DistilBERT with Conversational Context

The context-aware model uses DistilBERT with:

- Previous utterance
- Current utterance
- Speaker markers

This helps distinguish context-dependent therapist behaviours such as reflections.

## 4. RoBERTa

RoBERTa is fine-tuned as a strong utterance-level Transformer classifier using the current utterance without additional previous-turn context.

## 5. Google FLAN-T5

FLAN-T5 is included as an additional generative benchmark, formulating classification as text generation rather than discriminative classification.

---

# Conversational Context Modelling

The framework evaluates:

- Current utterance only
- Previous + current utterance
- Speaker markers
- Therapist–client turn order
- Contextual input construction

Context is particularly useful for identifying reflections, short questions, client motivational responses, backchannel acknowledgements, and therapist responses to Change Talk or Sustain Talk.

---

# Leakage-Aware Experimental Design

Complete psychotherapy sessions are assigned to either training or testing partitions so that utterances from the same session cannot appear in both.

The design includes:

- Transcript-level stratified train/test splitting
- Group-based cross-validation
- Grouping by `transcript_id`
- Out-of-fold utterance predictions
- Independent held-out test evaluation

Out-of-fold predictions are used to construct training-side session features and avoid downstream leakage.

---

# Session-Level Fidelity Assessment

Utterance predictions are aggregated into interpretable session-level features.

The session-level classifier uses:

```text
Logistic Regression
Standardised features
Balanced class weights
```

It produces:

- Binary MI-quality prediction
- Probability of high-quality MI

---

# Session-Level Feature Groups

## Static Therapist Features

- Reflection ratio
- Question ratio
- Input ratio
- Other ratio

## Static Client Features

- Change Talk ratio
- Sustain Talk ratio
- Neutral Talk ratio

## Behaviour Signal Features

- Question signal rate
- Reflection signal rate
- Input signal rate
- Mean signal count
- No-signal rate
- Multi-signal rate
- All-signal rate

## Behaviour Subtype Features

- Open-question rate
- Closed-question rate
- Simple-reflection rate
- Complex-reflection rate
- Advice or information-input rate
- Missing question-subtype rate
- Missing reflection-subtype rate
- Missing input-subtype rate
- Existence–subtype mismatch rate

## Interaction Features

- Reflection × Change Talk
- Input × Sustain Talk
- Question × Change Talk
- Reflection-to-question ratio

## Therapist-to-Client Transition Features

Examples include:

- Reflection → Change Talk
- Reflection → Sustain Talk
- Reflection → Neutral Talk
- Question → Change Talk
- Question → Sustain Talk
- Input → Sustain Talk

## Early-to-Late Trajectory Features

Trajectory is represented as:

```text
Late-session behaviour rate
minus
Early-session behaviour rate
```

Examples include Change Talk, Sustain Talk, Reflection, Question, and Input trajectories.

## Speech-Derived Structural Features

- Therapist speaking-time ratio
- Client speaking-time ratio
- Mean therapist turn duration
- Mean client turn duration
- Silence proportion
- Mean pause duration
- Response latency
- Speaker-switch frequency
- Overlapping-speech rate
- Interruption count
- Speech-rate estimate
- Mean ASR confidence
- Low-confidence transcription rate
- Mean no-speech probability

---

# Evaluation

## Voice Activity Detection

- Precision
- Recall
- F1-score
- False-alarm duration
- Missed-speech duration
- Detection Error Rate

## Automatic Speech Recognition

- Word Error Rate
- Character Error Rate
- Substitution rate
- Deletion rate
- Insertion rate
- Real-Time Factor
- Timestamp-alignment error

## Speaker Diarization

- Diarization Error Rate
- Speaker confusion
- Missed speech
- False-alarm speech
- Therapist/client attribution accuracy

## Utterance-Level Behaviour Classification

- Macro F1-score
- Per-class precision
- Per-class recall
- Per-class F1-score
- Accuracy
- Confusion matrix

## Session-Level Fidelity Assessment

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Predicted probability of high-quality MI

---

# End-to-End Evaluation

The system compares:

```text
Human transcript
→ Behaviour classifiers
→ Session features
→ MI fidelity prediction
```

against:

```text
YouTube audio
→ VAD
→ ASR
→ Diarization
→ Speaker-labelled ASR utterances
→ Behaviour classifiers
→ Session features
→ MI fidelity prediction
```

This measures how speech-processing errors affect utterance classification, transition features, trajectory features, session-level ROC-AUC, and fidelity probability estimates.

---

# Error-Propagation Analysis

```text
VAD error
    ↓
Missing or fragmented speech
    ↓
ASR transcription error
    ↓
Incorrect utterance representation
    ↓
Diarization or role-assignment error
    ↓
Behaviour classification error
    ↓
Distorted session features
    ↓
Incorrect fidelity assessment
```

Example failure modes include:

- VAD removing quiet client speech
- Whisper misrecognising clinically meaningful words
- Diarization assigning therapist speech to the client
- Punctuation changing a question into a statement
- A reflection being split into several fragments
- Short acknowledgements being removed
- Overlapping speech being duplicated
- Incorrect role mapping reversing therapist and client labels

---

# Interpretability

Interpretability outputs include:

- Logistic regression coefficients
- Behaviour-ratio comparisons
- Therapist-to-client transition patterns
- Early-to-late trajectories
- Timestamp-linked supporting utterances
- Low-confidence ASR regions
- Speaker-attribution information
- Speech-processing diagnostics
- Static versus trajectory comparisons
- ROC curve comparisons

---

# Dataset

The behavioural modelling component uses the **AnnoMI Full Dataset**.

The dataset contains:

- Session-level MI quality
- Transcript identifiers
- YouTube video URLs
- Video titles
- Topic metadata
- Utterance identifiers
- Speaker labels
- Timestamps
- Utterance text
- Therapist behaviour labels
- Client talk labels
- Behaviour existence indicators
- Question subtypes
- Reflection subtypes
- Therapist input subtypes
- Annotator information

The Full version is used because its subtype and behaviour-signal columns support richer hierarchical and session-level feature engineering.

---

# Current Model Stack

## Speech AI

```text
Audio acquisition:
yt-dlp

Audio processing:
FFmpeg
FFprobe

Voice Activity Detection:
Silero VAD
WebRTC VAD reference implementation

Automatic Speech Recognition:
Faster-Whisper large-v3

Speaker Diarization:
pyannote speaker-diarization-community-1

Speech evaluation:
jiwer
```

## NLP and Language Models

```text
TF-IDF + LinearSVC
SBERT all-MiniLM-L6-v2 + Logistic Regression
DistilBERT with conversational context
RoBERTa
Google FLAN-T5
```

## Session-Level Modelling

```text
StandardScaler
Logistic Regression
Balanced class weights
Static behavioural features
Interaction features
Transition features
Trajectory features
Speech-derived structural features
```

---

# Technology Stack

## Programming

- Python
- SQL
- JavaScript

## Data Processing

- pandas
- NumPy

## Audio and Speech Processing

- yt-dlp
- FFmpeg
- FFprobe
- Silero VAD
- WebRTC VAD
- Faster-Whisper
- pyannote.audio
- librosa
- soundfile
- jiwer

## Machine Learning

- scikit-learn
- LinearSVC
- Logistic Regression
- GroupKFold
- StandardScaler

## NLP and Transformers

- Hugging Face Transformers
- Sentence Transformers
- DistilBERT
- RoBERTa
- FLAN-T5
- PyTorch

## Development

- Jupyter Notebook
- Google Colab
- Cursor
- Git
- GitHub

## Deployment and Infrastructure

- Docker
- Google Cloud Platform
- Google Cloud Run

---

# Repository Structure

```text
llm-therapy-fidelity/
├── README.md
├── psychotherapy_pipeline_code.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── annotations/
│   ├── audio_manifest/
│   └── sample_data/
│
├── speech_outputs/
│   ├── speech_run_manifest.csv
│   ├── speech_utterances_all.csv
│   ├── vad_segments/
│   ├── asr_segments/
│   └── diarization_segments/
│
├── models/
│   ├── therapist/
│   ├── client/
│   └── session/
│
├── figures/
│
├── src/
│   ├── audio/
│   ├── nlp/
│   ├── fidelity/
│   └── evaluation/
│
└── docs/
    ├── methodology.md
    ├── model_card.md
    ├── data_statement.md
    └── ethical_considerations.md
```

---

# Reproducibility

The repository supports reproducible experimentation through:

- Fixed transcript-level splits
- Group-based cross-validation
- Saved out-of-fold predictions
- Cached audio files
- Timestamp-preserving ASR output
- Saved diarization artefacts
- Saved speech-run manifests
- Deterministic feature engineering
- Versioned model configurations
- Exported evaluation tables
- Saved figures
- Dependency documentation

Failed sessions are recorded rather than silently discarded.

Example statuses:

```text
success
download_failed
audio_validation_failed
vad_failed
asr_failed
diarization_failed
alignment_failed
```

---

# Recommended Repository Files

## Required

```text
README.md
psychotherapy_pipeline_code.ipynb
requirements.txt
.gitignore
LICENSE
```

## Strongly Recommended

```text
MODEL_CARD.md
DATA_STATEMENT.md
ETHICAL_CONSIDERATIONS.md
REPRODUCIBILITY.md
```

## Optional

```text
CONTRIBUTING.md
CITATION.cff
CHANGELOG.md
```

Do not upload:

- Raw psychotherapy audio
- Hugging Face tokens
- API keys
- Private patient information
- Large model checkpoints unless explicitly intended
- Temporary notebook cache
- Local Python environments
- LaTeX auxiliary files
- Personal submission documents

---

# Ethical and Privacy Considerations

Psychotherapy audio and transcripts may contain highly sensitive information.

The system must prioritise:

- Informed consent
- Secure storage
- Encryption
- Access control
- De-identification
- Personally identifiable information removal
- Dataset governance
- Restricted model outputs
- Human professional oversight

The project is intended as a research and decision-support framework.

It is not designed to:

- Replace qualified therapists
- Make clinical diagnoses
- Provide autonomous treatment decisions
- Evaluate individuals without consent
- Operate as a standalone clinical assessment system

Raw audio requires stronger protection than derived and de-identified behavioural features.

---

# Limitations

Important limitations include:

- Limited availability of public psychotherapy audio
- Privacy restrictions around clinical recordings
- Missing, private, or deleted YouTube videos
- ASR errors in emotional or low-volume speech
- Accent and demographic bias
- Speaker diarization errors
- Overlapping speech
- Domain mismatch between general ASR data and psychotherapy dialogue
- Annotation uncertainty
- Limited session-level ground truth
- Propagation of upstream speech errors
- High computational requirements for long recordings
- Need for external clinical validation

Automated fidelity outputs should be interpreted as decision-support evidence rather than definitive clinical judgments.

---

# Research Contribution

The main contribution of this work is an interpretable, leakage-aware, hierarchical framework connecting psychotherapy audio and dialogue with session-level fidelity assessment.

The framework:

- Extends fidelity assessment from transcripts to raw session audio
- Integrates VAD, ASR, diarization, role assignment, and NLP
- Models therapist and client behaviours separately
- Preserves the hierarchical structure of psychotherapy assessment
- Uses out-of-fold predictions for downstream session modelling
- Combines static, interaction, transition, trajectory, and speech-derived features
- Evaluates error propagation from audio processing to fidelity scoring
- Produces timestamp-linked and behaviour-linked evidence
- Supports categorical and probability-based fidelity assessment
- Prioritises interpretability over opaque end-to-end scoring

The objective is not merely to classify a session as high or low quality. The system is designed to provide transparent evidence describing how speech, language, behaviour, and conversational dynamics contribute to the assessment.

---

# Project Status

The transcript-based NLP and session-level modelling components have been implemented.

The repository is currently being extended and consolidated with an end-to-end Speech AI front end covering:

- YouTube audio acquisition
- Audio validation and standardisation
- Voice Activity Detection
- Faster-Whisper transcription
- pyannote speaker diarization
- Therapist/client role assignment
- ASR-to-gold utterance matching
- Speech-derived session features
- End-to-end error-propagation analysis

Source code, model configurations, evaluation outputs, figures, and reproducibility documentation are being organised for public presentation.

---

# Author

**Minnie Thu Dang**

AI/ML Engineer · Data Scientist · NLP and Speech AI Researcher

Master of Data Science  
Monash University

Research and engineering interests:

- Speech AI
- Automatic Speech Recognition
- Voice Activity Detection
- Speaker Diarization
- Speaker Recognition
- Natural Language Processing
- Large Language Models
- Transformer-Based Systems
- Conversational AI
- Computational Mental Health
- Responsible and Interpretable AI
