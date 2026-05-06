# OSA Personalized Acoustic Intervention System V2
# ==================================================
# Classification-based system for preventing obstructive sleep apnea
# events through personalized acoustic interventions delivered via earphones.
#
# Architecture:
#   1. Signal Processing Pipeline (RIP, Audio, IMU, SpO2) → 8-dim features
#   2. State Classifier → 4 states (Awake, Normal Sleep, Snoring, Apnea)
#   3. Trend Encoder → Temporal pattern analysis (Bi-LSTM)
#   4. Decision Engine → Rule-based intervention logic
#   5. Audio Synthesis → Binaural spatial audio with ITD/ILD

__version__ = "2.0.0"

# V2 Core Components
from osa_system.system_v2 import (
    OSASystemV2,
    StateClassifier,
    DecisionEngine,
    TrendEncoder,
    OSAState,
    STATE_NAMES,
    InterventionDecision,
    FocalLoss,
)

# Shared Components
from osa_system.signal_processing import (
    SignalConfig,
    MultimodalFeatureExtractor,
)

from osa_system.audio_synthesis import (
    AudioSynthesizer,
    AudioConfig,
)

from osa_system.ucddb_parser import (
    build_ucddb_dataset,
    SleepState,
    EpochLabel,
)

__all__ = [
    # V2 System
    'OSASystemV2',
    'StateClassifier',
    'DecisionEngine',
    'TrendEncoder',
    'OSAState',
    'STATE_NAMES',
    'InterventionDecision',
    'FocalLoss',

    # Signal Processing
    'SignalConfig',
    'MultimodalFeatureExtractor',

    # Audio Synthesis
    'AudioSynthesizer',
    'AudioConfig',

    # UCDDB Data
    'build_ucddb_dataset',
    'SleepState',
    'EpochLabel',
]
