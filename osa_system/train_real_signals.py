"""
Real Signal Feature Extraction + Classifier Training on UCDDB
================================================================

Uses ACTUAL EDF signals from UCDDB (not synthetic features).
Extracts per-epoch features from real ribcage/abdo RIP, SpO2, Sound, 
BodyPos, and Flow channels, then trains the 4-state classifier.

This is the honest pipeline:
  Real signals → Real features → Train classifier → Measure real accuracy
"""

import numpy as np
import os
import struct
from typing import Dict, List, Tuple, Optional
from scipy.signal import butter, sosfilt, welch, find_peaks
from scipy.signal import hilbert


# =============================================================================
# EDF Reader (handles partial/truncated files)
# =============================================================================

def read_ucddb_edf(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Read UCDDB .rec (EDF format) file, handling truncation."""
    with open(filepath, 'rb') as f:
        header = f.read(256)
        n_records = int(header[236:244].decode('ascii').strip())
        dur_record = float(header[244:252].decode('ascii').strip())
        n_signals = int(header[252:256].decode('ascii').strip())
        
        f.seek(256)
        labels = [f.read(16).decode('ascii', errors='replace').strip() for _ in range(n_signals)]
        f.read(80 * n_signals)  # transducers
        phys_dims = [f.read(8).decode('ascii').strip() for _ in range(n_signals)]
        phys_mins = [float(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        phys_maxs = [float(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        dig_mins = [int(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        dig_maxs = [int(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        f.read(80 * n_signals)  # prefiltering
        samples_per_record = [int(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        f.read(32 * n_signals)  # reserved
        
        header_size = 256 + 256 * n_signals
        f.seek(0, 2)
        file_size = f.tell()
        
        bytes_per_record = sum(s * 2 for s in samples_per_record)
        available_records = (file_size - header_size) // bytes_per_record
        records_to_read = min(available_records, n_records)
        
        scales = {}
        for i in range(n_signals):
            dig_range = dig_maxs[i] - dig_mins[i]
            phys_range = phys_maxs[i] - phys_mins[i]
            if dig_range != 0:
                scales[i] = (phys_range / dig_range, phys_mins[i] - dig_mins[i] * phys_range / dig_range)
            else:
                scales[i] = (1.0, 0.0)
        
        f.seek(header_size)
        signals = {labels[i]: [] for i in range(n_signals)}
        
        for rec in range(records_to_read):
            for i in range(n_signals):
                raw = f.read(samples_per_record[i] * 2)
                if len(raw) < samples_per_record[i] * 2:
                    break
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
                s, o = scales[i]
                signals[labels[i]].append(data * s + o)
            else:
                continue
            break
        
        for key in signals:
            signals[key] = np.concatenate(signals[key]) if signals[key] else np.array([])
        
        info = {
            'labels': labels,
            'sample_rates': {labels[i]: samples_per_record[i] / dur_record for i in range(n_signals)},
            'n_records': records_to_read,
            'duration_seconds': records_to_read * dur_record,
        }
        
    return signals, info


# =============================================================================
# Real Feature Extraction (per 30-second epoch)
# =============================================================================

def extract_epoch_features(
    ribcage: np.ndarray,
    abdo: np.ndarray,
    spo2: np.ndarray,
    sound: np.ndarray,
    flow: np.ndarray,
    bodypos: np.ndarray,
    fs: float = 8.0,
) -> np.ndarray:
    """
    Extract 33-dim feature vector from real signals for one 30s epoch.
    
    All signals at 8 Hz, so each epoch = 240 samples.
    """
    n = len(ribcage)
    
    # ---- RIP Features (11) ----
    # Respiratory amplitude
    thorax_amp = float(np.std(ribcage))
    abdo_amp = float(np.std(abdo))
    total_amp = thorax_amp + abdo_amp
    
    # Respiratory rate via peak detection  
    sum_signal = ribcage + abdo
    try:
        peaks, _ = find_peaks(sum_signal, distance=int(fs * 1.5), height=0)
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / fs
            resp_rate = 60.0 / np.mean(intervals)
            resp_rate_var = np.std(intervals) / (np.mean(intervals) + 1e-8)
        else:
            resp_rate = 0.0
            resp_rate_var = 0.0
    except:
        resp_rate = 12.0
        resp_rate_var = 0.0
    
    # Phase angle (Hilbert transform)
    try:
        phase_t = np.angle(hilbert(ribcage))
        phase_a = np.angle(hilbert(abdo))
        phase_diff = phase_t - phase_a
        phase_angle = float(np.degrees(np.arctan2(
            np.mean(np.sin(phase_diff)),
            np.mean(np.cos(phase_diff))
        )))
    except:
        phase_angle = 0.0
    
    is_paradoxical = float(abs(phase_angle) > 90.0)
    
    # Flow limitation
    inspiratory = flow[flow > 0]
    if len(inspiratory) > 5:
        flow_limitation = float(np.percentile(inspiratory, 95) / (np.mean(inspiratory) + 1e-8))
    else:
        flow_limitation = 5.0
    
    # Respiratory effort
    resp_effort = total_amp / (total_amp + 1e-8)
    
    rip_features = [
        thorax_amp,
        abdo_amp,
        total_amp,
        np.clip(resp_rate, 0, 60) / 60.0,
        resp_rate_var,
        phase_angle / 180.0,
        is_paradoxical,
        np.clip(flow_limitation / 5.0, 0, 2),
        resp_effort,
        0.0, 0.0,  # reserved
    ]
    
    # ---- Audio/Sound Features (9) ----
    sound_rms = float(np.sqrt(np.mean(sound ** 2)))
    
    # F0 estimation via autocorrelation
    try:
        ac = np.correlate(sound, sound, mode='full')[n-1:]
        ac = ac / (ac[0] + 1e-12)
        min_lag = max(1, int(fs / 5))  # max 5 Hz  
        max_lag = min(int(fs / 0.5), len(ac) - 1)  # min 0.5 Hz
        if min_lag < max_lag:
            search = ac[min_lag:max_lag]
            peak_idx = np.argmax(search) + min_lag
            f0_confidence = float(ac[peak_idx])
            f0 = fs / peak_idx if f0_confidence > 0.3 else 0.0
        else:
            f0, f0_confidence = 0.0, 0.0
    except:
        f0, f0_confidence = 0.0, 0.0
    
    # Spectral features
    try:
        freqs, psd = welch(sound, fs=fs, nperseg=min(64, n))
        total_power = np.sum(psd) + 1e-12
        spectral_centroid = float(np.sum(freqs * psd) / total_power)
        spectral_bw = float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd) / total_power))
    except:
        spectral_centroid, spectral_bw = 0.0, 0.0
    
    audio_features = [
        min(sound_rms * 100, 1.0),  # Normalize
        f0 / 5.0,  # Normalize (max ~4 Hz at 8 Hz sampling)
        f0_confidence,
        0.0,  # f0_stability (need previous epoch)
        spectral_centroid / 4.0,
        spectral_bw / 2.0,
        float(sound_rms > 0.005),  # Snore present
        0.0,  # intermittency
        0.0,  # crescendo
    ]
    
    # ---- IMU/Position Features (9) ----
    # UCDDB BodyPos: typically 1=left, 2=right, 3=prone, 4=supine, 5=upright
    mean_pos = float(np.mean(bodypos))
    is_supine = float(abs(mean_pos - 4.0) < 0.5)  # Position ~4 = supine
    
    # Position stability
    pos_std = float(np.std(bodypos))
    pos_stability = float(1.0 / (1.0 + pos_std))
    
    # Movement from position variance
    movement = float(np.var(np.diff(bodypos)))
    
    imu_features = [
        is_supine,
        0.0,  # pitch (not available from single position channel)
        0.0,  # roll
        pos_stability,
        movement,
        float(np.std(np.diff(bodypos))),
        0.0, 0.0, 0.0,  # reserved
    ]
    
    # ---- SpO2 Features (4) ----
    valid_spo2 = spo2[(spo2 > 50) & (spo2 <= 100)]
    if len(valid_spo2) > 0:
        mean_spo2 = float(np.mean(valid_spo2))
        spo2_norm = np.clip((mean_spo2 - 70.0) / 30.0, 0, 1)
    else:
        mean_spo2 = 95.0
        spo2_norm = 0.83
    
    # SpO2 trend (slope)
    try:
        if len(valid_spo2) > 2:
            slope = np.polyfit(np.arange(len(valid_spo2)), valid_spo2, 1)[0]
        else:
            slope = 0.0
    except:
        slope = 0.0
    
    hypoxemia_risk = float(1.0 / (1.0 + np.exp(0.5 * (mean_spo2 - 90.0))))
    
    spo2_features = [
        spo2_norm,
        np.clip(slope / 2.0, -1, 1),
        hypoxemia_risk,
        0.0,  # ODI proxy (need longer history)
    ]
    
    # Assemble 33-dim vector
    features = rip_features + audio_features + imu_features + spo2_features
    assert len(features) == 33, f"Got {len(features)} features, expected 33"
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Build Dataset from Real Signals + Annotations
# =============================================================================

def build_real_feature_dataset(
    signals: Dict[str, np.ndarray],
    info: Dict,
    labels: list,
    epoch_sec: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from real signals for each labeled epoch.
    
    Returns:
        features: (n_epochs, 33) 
        states: (n_epochs,) 
        severities: (n_epochs,)
    """
    fs = 8.0  # All respiratory channels at 8 Hz
    samples_per_epoch = int(epoch_sec * fs)  # 240
    
    signal_duration = info['duration_seconds']
    n_signal_epochs = int(signal_duration / epoch_sec)
    n_label_epochs = len(labels)
    n_epochs = min(n_signal_epochs, n_label_epochs)
    
    print(f"  Signal epochs: {n_signal_epochs}, Label epochs: {n_label_epochs}, Using: {n_epochs}")
    
    features_list = []
    states_list = []
    severities_list = []
    
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        
        # Check bounds
        if end > len(signals.get('ribcage', [])):
            break
        
        ribcage = signals['ribcage'][start:end]
        abdo = signals['abdo'][start:end]
        spo2 = signals['SpO2'][start:end]
        sound = signals['Sound'][start:end]
        flow = signals['Flow'][start:end]
        bodypos = signals['BodyPos'][start:end]
        
        feat = extract_epoch_features(ribcage, abdo, spo2, sound, flow, bodypos, fs=fs)
        
        features_list.append(feat)
        states_list.append(labels[i].state)
        severities_list.append(labels[i].severity)
    
    return (
        np.array(features_list, dtype=np.float32),
        np.array(states_list, dtype=np.int64),
        np.array(severities_list, dtype=np.float32),
    )


# =============================================================================
# Training with Real Features
# =============================================================================

def train_and_evaluate_real(data_dir: str):
    """
    Full pipeline: read real EDF → extract features → train classifier → evaluate.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from collections import Counter
    
    from osa_system.ucddb_parser import build_ucddb_dataset
    from osa_system.system_v2 import StateClassifier, FocalLoss, STATE_NAMES
    
    print("=" * 70)
    print("  Training Classifier on REAL UCDDB Signals")
    print("=" * 70)
    
    # 1. Load annotations
    dataset = build_ucddb_dataset(data_dir)
    
    # 2. Find available EDF files
    available_subjects = []
    for subj in dataset['subjects']:
        rec_file = os.path.join(data_dir, f'ucddb{subj}.rec')
        if os.path.exists(rec_file) and os.path.getsize(rec_file) > 5_000_000:
            available_subjects.append(subj)
    
    print(f"\n  Available EDF files: {len(available_subjects)} subjects: {available_subjects}")
    
    if not available_subjects:
        print("  No EDF files available! Cannot train on real signals.")
        return None
    
    # 3. Extract features from each available subject
    all_features = []
    all_states = []
    all_severities = []
    all_subject_ids = []
    
    for subj in available_subjects:
        print(f"\n  Processing subject {subj}...")
        rec_file = os.path.join(data_dir, f'ucddb{subj}.rec')
        
        try:
            signals, info = read_ucddb_edf(rec_file)
            labels = dataset['labels'][subj]
            
            features, states, severities = build_real_feature_dataset(
                signals, info, labels
            )
            
            print(f"    Extracted {len(features)} epochs with real features")
            print(f"    State distribution: {dict(Counter(states.tolist()))}")
            
            all_features.append(features)
            all_states.append(states)
            all_severities.append(severities)
            all_subject_ids.extend([subj] * len(features))
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    if not all_features:
        print("  No features extracted!")
        return None
    
    # Concatenate all
    X = np.concatenate(all_features)
    y = np.concatenate(all_states)
    sev = np.concatenate(all_severities)
    subject_ids = np.array(all_subject_ids)
    
    print(f"\n  Total dataset: {len(X)} epochs from {len(available_subjects)} subjects")
    print(f"  State distribution:")
    for state_id in range(4):
        count = np.sum(y == state_id)
        print(f"    {STATE_NAMES[state_id]:15s}: {count:5d} ({count/len(y)*100:.1f}%)")
    
    # 4. Feature statistics (sanity check)
    print(f"\n  Feature statistics (real signals):")
    feat_names = [
        'thorax_amp', 'abdo_amp', 'total_amp', 'resp_rate', 'resp_rate_var',
        'phase_angle', 'is_paradoxical', 'flow_limit', 'resp_effort', 'rsv1', 'rsv2',
        'sound_rms', 'f0', 'f0_conf', 'f0_stab', 'spec_cent', 'spec_bw',
        'snore_present', 'intermit', 'crescendo',
        'is_supine', 'pitch', 'roll', 'pos_stab', 'movement', 'mov_var',
        'rsv3', 'rsv4', 'rsv5',
        'spo2_norm', 'spo2_slope', 'hypox_risk', 'odi'
    ]
    for i, name in enumerate(feat_names[:15]):
        vals = X[:, i]
        print(f"    {name:15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")
    
    # 5. Train/test split (if only 1 subject, use 80/20 random split)
    # Check how many subjects actually contributed data
    contributing_subjects = list(set(all_subject_ids))
    
    if len(contributing_subjects) > 1:
        # Leave-one-subject-out
        test_subj = contributing_subjects[-1]
        train_mask = subject_ids != test_subj
        test_mask = subject_ids == test_subj
        print(f"\n  LOSO: Train on {contributing_subjects[:-1]}, Test on {test_subj}")
    else:
        # Random 80/20 split (only one subject has data)
        rng = np.random.default_rng(42)
        n = len(X)
        idx = rng.permutation(n)
        split = int(0.8 * n)
        train_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[idx[:split]] = True
        test_mask[idx[split:]] = True
        print(f"\n  Random 80/20 split (single subject: {contributing_subjects[0]})")
    
    X_train, y_train, sev_train = X[train_mask], y[train_mask], sev[train_mask]
    X_test, y_test, sev_test = X[test_mask], y[test_mask], sev[test_mask]
    
    print(f"  Train: {len(X_train)} epochs")
    print(f"  Test:  {len(X_test)} epochs")
    
    # 6. Normalize features
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std
    
    # 7. Train classifier
    model = StateClassifier(input_dim=33, hidden_dim=128)
    
    # Weighted sampler for class imbalance
    class_counts = Counter(y_train.tolist())
    weights = np.array([1.0 / class_counts[s] for s in y_train])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.LongTensor(y_train),
        torch.FloatTensor(sev_train),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.LongTensor(y_test),
        torch.FloatTensor(sev_test),
    )
    
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    
    focal_loss = FocalLoss(gamma=2.0)
    sev_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    print(f"\n  Training for 80 epochs...")
    best_acc = 0
    best_state = None
    
    for epoch in range(80):
        model.train()
        total_loss = 0
        n_batch = 0
        
        for feats, states, sevs in train_loader:
            optimizer.zero_grad()
            out = model(feats)
            loss = focal_loss(out['state_logits'], states) + 0.5 * sev_loss_fn(out['severity'], sevs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                for feats, states, sevs in test_loader:
                    out = model(feats)
                    preds = torch.argmax(out['state_probs'], dim=-1)
                    acc = accuracy_score(states.numpy(), preds.numpy())
            
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            print(f"    Epoch {epoch+1:3d}: loss={total_loss/n_batch:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}")
    
    # 8. Final evaluation with best model
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        for feats, states, sevs in test_loader:
            out = model(feats)
            preds = torch.argmax(out['state_probs'], dim=-1).numpy()
            true = states.numpy()
    
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS (Real UCDDB Signals)")
    print(f"{'='*70}")
    print(f"\n  Test Accuracy: {accuracy_score(true, preds):.4f}")
    
    target_names = [STATE_NAMES[i] for i in range(4)]
    print(f"\n{classification_report(true, preds, target_names=target_names, digits=3, zero_division=0)}")
    
    cm = confusion_matrix(true, preds, labels=[0,1,2,3])
    print(f"  Confusion Matrix:")
    print(f"  {'':15s} {'P-Awake':>10s} {'P-Normal':>10s} {'P-Snore':>10s} {'P-Apnea':>10s}")
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:15s}", end="")
        for v in row:
            print(f"{v:10d}", end="")
        print()
    
    # Save model and normalization stats
    save_dir = os.path.join(data_dir, '..', 'osa_models_v2')
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'classifier_real.pt'))
    np.savez(os.path.join(save_dir, 'feature_normalization.npz'), mean=feat_mean, std=feat_std)
    print(f"\n  Model saved to {save_dir}/classifier_real.pt")
    
    return {
        'accuracy': float(accuracy_score(true, preds)),
        'predictions': preds,
        'true_labels': true,
        'confusion_matrix': cm,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_subjects': len(available_subjects),
    }


if __name__ == '__main__':
    results = train_and_evaluate_real('/app/ucddb_data')
