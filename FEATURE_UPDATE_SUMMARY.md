# Feature Dimension Update: 33 → 8 (UCDDB-Aligned)

## Summary

Successfully updated the OSA Intervention System to use 8 UCDDB-aligned features instead of the original 33-dimensional feature vector. This change aligns the system with the actual UCDDB data channels available.

## Feature Mapping

The new 8-dimensional feature vector maps directly to UCDDB channels:

| Index | Feature Name | UCDDB Channel | Description |
|-------|-------------|---------------|-------------|
| 0 | RIP chest amplitude | `ribcage` | Thoracic respiratory inductance plethysmography |
| 1 | RIP abdominal amplitude | `abdo` | Abdominal respiratory inductance plethysmography |
| 2 | Respiratory rate | `Sum` / `ribcage` | Breaths per minute (normalized to [0,1]) |
| 3 | Chest-abdomen phase difference | `ribcage` + `abdo` | Phase angle via Hilbert transform (normalized to [-1,1]) |
| 4 | Snoring RMS | `Sound` | Root mean square of audio signal |
| 5 | Body position/supine detection | `BodyPos` | Binary: 1 if supine, 0 otherwise |
| 6 | SpO2 value | `SpO2` | Blood oxygen saturation (normalized to [0,1]) |
| 7 | SpO2 decline slope | `SpO2` difference | Rate of oxygen desaturation |

## Files Modified

### 1. `osa_system/system.py`
- **StateClassifier**: Changed `input_dim` default from 33 → 8
- **TrendEncoder**: Changed `input_dim` default from 33 → 8
- **OSASystem**: Updated to use 8-dimensional features
- Updated documentation to reflect UCDDB-aligned features

### 2. `osa_system/signal_processing.py`
- **MultimodalFeatureExtractor.FEATURE_DIM**: Changed from 33 → 8
- **extract_all()**: Simplified to output 8 features matching UCDDB channels
- Updated feature vector layout documentation

### 3. `osa_system/ucddb_parser.py`
- Added new function `extract_ucddb_features()` to extract 8 features from raw UCDDB signals
- Implements proper signal processing for each UCDDB channel
- Uses Hilbert transform for phase difference calculation
- Handles body position encoding (supine detection)

### 4. `osa_system/train_classifier.py`
- Updated `epoch_label_to_features()` to generate 8-dimensional synthetic features
- Modified all helper functions:
  - `_awake_features()`: 8 features
  - `_normal_sleep_features()`: 8 features
  - `_snoring_features()`: 8 features
  - `_apnea_features()`: 8 features
- Updated `train_classifier()` to use `input_dim=8`

### 5. `osa_system/train_real_signals.py`
- Updated `extract_epoch_features()` to extract 8 features from real EDF signals
- Simplified feature extraction to match UCDDB channels
- Updated feature statistics display
- Changed model initialization to `input_dim=8`

### 6. `osa_system/main.py`
- Updated system architecture diagram: "33-dim" → "8-dim"
- Updated documentation to reflect UCDDB-aligned features

## Verification

All models now correctly use 8-dimensional input:

```python
StateClassifier().input_dim = 8
TrendEncoder().input_dim = 8
MultimodalFeatureExtractor.FEATURE_DIM = 8
OSASystem().classifier.input_dim = 8
OSASystem().trend_encoder.input_dim = 8
```

## Feature Extraction Functions

### From Real UCDDB Signals
```python
from osa_system.ucddb_parser import extract_ucddb_features

features = extract_ucddb_features(
    ribcage=ribcage_signal,    # np.ndarray
    abdo=abdo_signal,          # np.ndarray
    sound=sound_signal,        # np.ndarray
    body_pos=bodypos_signal,   # np.ndarray
    spo2=spo2_signal,          # np.ndarray
    fs=128                     # Sampling frequency
)
# Returns: np.ndarray of shape (8,)
```

### From Annotations (Synthetic)
```python
from osa_system.train_classifier import epoch_label_to_features

features = epoch_label_to_features(label, rng)
# Returns: np.ndarray of shape (8,)
```

## Training Pipeline

The training pipeline remains the same, but now uses 8 features:

1. **Synthetic Training** (from annotations):
   ```bash
   python osa_system/train_classifier.py
   ```

2. **Real Signal Training** (from EDF files):
   ```bash
   python osa_system/train_real_signals.py
   ```

3. **Evaluation** (on real data):
   ```bash
   python osa_system/evaluate_real_data.py
   ```

## Benefits

1. **Direct UCDDB Alignment**: Features map 1:1 with available UCDDB channels
2. **Simplified Architecture**: Reduced from 33 to 8 dimensions
3. **Faster Training**: Smaller input dimension = faster convergence
4. **Better Interpretability**: Each feature has clear clinical meaning
5. **Real Data Ready**: Can directly process UCDDB EDF files

## Backward Compatibility

⚠️ **Breaking Change**: Models trained with 33-dimensional features are NOT compatible with this update. You will need to retrain all models.

## Next Steps

1. Retrain the state classifier on UCDDB data with 8 features
2. Update saved model files in `osa_models/`
3. Verify performance metrics match or exceed previous 33-dim results
4. Update any external scripts that depend on feature dimensions

## Testing

Run the following to verify the changes:

```bash
# Test feature extraction
python3 -c "
from osa_system.ucddb_parser import extract_ucddb_features
import numpy as np
features = extract_ucddb_features(
    np.random.randn(3840), np.random.randn(3840),
    np.random.randn(3840), np.ones(3840)*4, np.ones(3840)*95
)
print('Features shape:', features.shape)
assert features.shape == (8,), 'Expected 8 features'
print('✓ Feature extraction working correctly')
"

# Test model dimensions
python3 -c "
from osa_system.system import OSASystem
system = OSASystem()
assert system.classifier.input_dim == 8
assert system.trend_encoder.input_dim == 8
print('✓ Model dimensions correct')
"
```

## Date
2026-04-28
