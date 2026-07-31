# Ethical Considerations

## Purpose

This project investigates automated psychotherapy intervention fidelity assessment using Speech AI, NLP, and interpretable machine learning.

It is intended for research and decision support.

## Clinical Boundaries

The system is not designed to:

- Replace therapists
- Diagnose mental-health conditions
- Select treatments autonomously
- Evaluate individuals without consent
- Make disciplinary or employment decisions
- Operate without qualified human oversight

## Privacy and Confidentiality

Psychotherapy recordings may contain highly sensitive information.

Required safeguards include:

- Informed consent
- Secure storage
- Encryption
- Role-based access control
- De-identification
- Personally identifiable information removal
- Restricted model outputs
- Data-retention controls
- Audit logging

Raw audio should receive stronger protection than aggregated, de-identified features.

## Bias and Fairness

Potential sources of bias include:

- Accent
- Dialect
- Age
- Gender presentation
- Speaking style
- Recording quality
- Cultural communication norms
- Domain mismatch
- Annotation practices

Performance should be examined across relevant subgroups whenever ethically and statistically feasible.

## Speaker Role Assignment

Therapist/client roles must not be inferred from perceived gender, pitch, accent, or other sensitive characteristics.

Preferred approaches include:

- Timestamp alignment with known labels
- A manually verified calibration segment
- Enrolled therapist voice samples
- Session metadata
- Human verification

## Explainability

The system prioritises interpretable behavioural evidence, including:

- Behaviour ratios
- Model coefficients
- Therapist-to-client transitions
- Early-to-late trajectories
- Timestamp-linked utterances
- Speech-processing confidence indicators

## Human Oversight

Qualified professionals must review model outputs before any real-world interpretation.

Low-confidence ASR, diarization uncertainty, and conflicting behavioural evidence should be surfaced rather than hidden.

## Responsible Release

The public repository must exclude:

- Raw psychotherapy audio
- Private transcripts
- Access tokens
- API keys
- Personal information
- Restricted model files
- Sensitive session-level outputs
