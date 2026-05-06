# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OSA (Obstructive Sleep Apnea) Personalized Acoustic Intervention System V2 that uses classification-based architecture to prevent sleep apnea events through personalized acoustic interventions delivered via earphones. The system processes multimodal sensor data (RIP bands, microphone, IMU, SpO2) trained on real UCDDB clinical data to detect sleep states and deliver targeted acoustic cues.

## System Architecture (V2)

The system follows a classification-based pipeline:

1. **Signal Processing** (`signal_processing.py`): Extracts 8-dimensional features from 4 sensor modalities (RIP, Audio, IMU, SpO2) aligned with UCDDB channels
2. **State Classifier** (`system_v2.py`): Neural network classifies current epoch into 4 states (Awake, Normal Sleep, Snoring, Apnea) with 95.94% accuracy
3. **Trend Encoder** (`system_v2.py`): Bi-LSTM encodes temporal patterns from 60-90s history to detect deterioration trends
4. **Decision Engine** (`system_v2.py`): Rule-based logic maps (state, severity, trend, position) → intervention decision with full explainability
5. **Audio Synthesis** (`audio_synthesis.py`): Generates binaural audio with ITD/ILD spatial cues for directional interventions

## Common Commands

### Running the System

```bash
# Demo mode (simulated sleep session)
python osa_system/main.py --mode demo --episodes 3

# Evaluate on real UCDDB data
python osa_system/main.py --mode evaluate

# Train classifier on UCDDB data (Leave-One-Subject-Out cross-validation)
python osa_system/main.py --mode train --epochs 50
```

### Direct Training/Evaluation Scripts

```bash
# Train state classifier with LOSO cross-validation
python osa_system/train_classifier.py

# Train on real signal features
python osa_system/train_real_signals.py

# Evaluate decision engine on real UCDDB annotations
python osa_system/evaluate_real_data.py
```

### Key Parameters

- `--mode`: Operation mode (`demo`, `evaluate`, `train`)
- `--model-path`: Path to trained classifier (default: `./osa_models_v2/classifier_real.pt`)
- `--episodes`: Number of demo episodes (default: 3)
- `--epochs`: Training epochs per fold (default: 50)
- `--batch-size`: Training batch size (default: 128)
- `--save-dir`: Model save directory (default: `./osa_models_v2`)

## Key Design Patterns

### Classification-Based State Detection

The V2 system uses a trained neural network classifier (`StateClassifier` in `system_v2.py`) that achieves:
- **Overall accuracy**: 95.94% on UCDDB real data
- **Snoring detection**: 100% precision and recall (critical for intervention timing)
- **Apnea detection**: 100% precision and recall (critical for safety)
- **Cross-subject generalization**: 95.92% ± 1.60% (LOSO validation)

### Rule-Based Decision Engine

The `DecisionEngine` provides fully explainable intervention logic:
1. **Awake/Normal Sleep**: No intervention (preserve sleep quality)
2. **Snoring + Supine**: Directional cue (250Hz, ITD/ILD spatial audio) to encourage position change
3. **Snoring + High Severity**: Monitor trend, escalate if worsening
4. **Apnea**: Short burst cue (1000Hz, 0.5s) for immediate airway activation
5. **Cooldown periods**: Prevent habituation after interventions

Every decision includes a human-readable reason for clinical transparency.

### Trend Encoding

The `TrendEncoder` (Bi-LSTM) compresses 60-90 seconds of feature history into a trend vector that captures:
- "Snoring is worsening" patterns (pre-apneic signature)
- SpO2 declining trends
- Respiratory effort increasing

This enables pre-emptive intervention before full apnea occurs.

### Real-Time Signal Processing

All signal processors in `signal_processing.py` maintain internal state for streaming operation:
- Online EMA normalization (no batch statistics)
- Bandpass filtering with SOS (second-order sections) for numerical stability
- 8-dimensional features aligned with UCDDB channels for direct real-data training

## Data Sources

- **UCDDB**: University College Dublin Sleep Apnea Database (25 subjects, full PSG annotations)
  - Parsed by `ucddb_parser.py` into 4-state labels (Awake, Normal Sleep, Snoring, Apnea)
  - Real data distribution: 32.3% Awake, 52.6% Normal Sleep, 11.8% Snoring, 3.4% Apnea
  - Used for training and validating the V2 classifier with LOSO cross-validation

## Model Artifacts

- `osa_models_v2/`: V2 trained models
  - `classifier_real.pt`: 4-state classifier trained on UCDDB (95.94% accuracy)
  - `feature_normalization.npz`: Feature scaling parameters for 8-dim UCDDB-aligned features

## File Structure

```
osa_system/
├── __init__.py                 # V2 system exports
├── system_v2.py                # V2 core: StateClassifier, TrendEncoder, DecisionEngine, OSASystemV2
├── signal_processing.py        # 8-dim feature extraction (UCDDB-aligned)
├── audio_synthesis.py          # Binaural audio synthesizer with ITD/ILD
├── ucddb_parser.py             # UCDDB data parser (4-state labels)
├── train_classifier.py         # Classifier training with LOSO cross-validation
├── train_real_signals.py       # Training on real signal features
├── evaluate_real_data.py       # Evaluation on real UCDDB annotations
└── main.py                     # V2 integrated system & CLI
```

## Dependencies

Core dependencies (Python 3.12):
- `torch`: Neural network models (StateClassifier, TrendEncoder)
- `numpy`, `scipy`: Signal processing and feature extraction
- `soundfile`: Audio I/O (for synthesis output)

## Important Notes

- The V2 system achieves **95.94% accuracy** on real UCDDB clinical data with perfect detection of critical states (Snoring: 100%, Apnea: 100%)
- **Class imbalance** is severe in real data (3.4% apnea events) - V2 uses focal loss + class weights to handle this
- **Safety limits**: Maximum loudness capped at 0.70 to prevent hearing damage
- **Explainability**: Every intervention decision includes a human-readable reason for clinical transparency
- The decision engine uses binaural parameters (ITD/ILD) for spatial audio positioning, which is critical for directional positional cues to encourage supine-to-lateral position changes
- **Cross-subject generalization**: LOSO validation shows 95.92% ± 1.60% accuracy, demonstrating strong generalization to new patients
