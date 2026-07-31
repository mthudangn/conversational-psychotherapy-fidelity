# End-to-End Speech and LLM-Based Psychotherapy Fidelity Assessment

An end-to-end **Speech AI, Natural Language Processing, and Machine Learning research system** for automatically assessing intervention fidelity in psychotherapy sessions, with a primary focus on **Motivational Interviewing (MI)**.

The system transforms raw psychotherapy audio into interpretable behavioural and session-level fidelity evidence through a hierarchical pipeline:

> Raw Audio → Voice Activity Detection → Automatic Speech Recognition → Speaker Diarization → Utterance Segmentation → Behaviour Classification → Session-Level Fidelity Assessment

This repository supports a research paper investigating whether modern Speech AI, Transformer-based language models, and interpretable machine learning can capture therapist–client behavioural dynamics associated with psychotherapy quality.

---

## Research Overview

Psychotherapy intervention fidelity refers to how closely a therapy session follows the principles, behaviours, and delivery standards of a particular therapeutic intervention.

Traditional fidelity assessment depends on trained human coders manually reviewing complete therapy sessions. Although clinically valuable, this process is:

- Time-consuming
- Expensive
- Difficult to scale
- Sensitive to subjective interpretation
- Dependent on specialist training
- Challenging to apply continuously across large clinical services

This research investigates an automated alternative that integrates **speech processing**, **conversational NLP**, and **hierarchical machine learning**.

Rather than directly predicting session quality from raw audio or transcript text, the proposed system decomposes the task into interpretable stages.

The pipeline first identifies when speech occurs, determines who is speaking, transcribes the audio, segments the conversation into meaningful utterances, classifies therapist and client behaviours, and finally aggregates these predictions into session-level fidelity indicators.

---

## End-to-End System Architecture

```text
Psychotherapy Session Audio
            │
            ▼
Voice Activity Detection
            │
            ▼
Speech Segmentation
            │
            ▼
Automatic Speech Recognition
            │
            ▼
Speaker Diarization
            │
            ▼
Therapist–Client Speaker Mapping
            │
            ▼
Utterance Segmentation and Alignment
            │
            ▼
Transcript Cleaning and Normalisation
            │
            ▼
Utterance-Level Behaviour Classification
            │
            ▼
Behavioural, Transition, and Trajectory Features
            │
            ▼
Session-Level MI Fidelity Assessment
            │
            ▼
Interpretable Fidelity Report
